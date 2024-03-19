import time
import tqdm
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss

from lift.rl.utils import (
    make_replay_buffer, 
    get_activation, 
    make_collector, 
    log_metrics,
)

class SAC:
    """Soft actor critic trainer"""
    def __init__(self, config, train_env, eval_env):
        self.config = config
        self.train_env = train_env
        self.eval_env = eval_env
        self.device = torch.device(config.device)

        self.replay_buffer = make_replay_buffer(
            batch_size=self.config.batch_size,
            prioritize=self.config.prioritize,
            buffer_size=self.config.replay_buffer_size,
            scratch_dir=self.config.scratch_dir,
            device="cpu",
        )
        self._init_policy()
        self._init_loss_module()
        self._init_optimizer()
    
    """TODO: remove lazy layer from first layer. manually specify input dims"""
    def _init_policy(self):
        # Define Actor Network
        in_keys = ["observation"]
        action_spec = self.train_env.action_spec
        if self.train_env.batch_size:
            action_spec = action_spec[(0,) * len(self.train_env.batch_size)]
        actor_net_kwargs = {
            "num_cells": self.config.hidden_sizes,
            "out_features": 2 * action_spec.shape[-1],
            "activation_class": get_activation(self.config.activation),
        }

        actor_net = MLP(**actor_net_kwargs)

        dist_class = TanhNormal
        dist_kwargs = {
            "min": action_spec.space.low,
            "max": action_spec.space.high,
            "tanh_loc": False,
        }

        actor_extractor = NormalParamExtractor(
            scale_mapping=f"biased_softplus_{self.config.default_policy_scale}",
            scale_lb=self.config.scale_lb,
        )
        actor_net = nn.Sequential(actor_net, actor_extractor)

        in_keys_actor = in_keys
        actor_module = TensorDictModule(
            actor_net,
            in_keys=in_keys_actor,
            out_keys=[
                "loc",
                "scale",
            ],
        )
        actor = ProbabilisticActor(
            spec=action_spec,
            in_keys=["loc", "scale"],
            module=actor_module,
            distribution_class=dist_class,
            distribution_kwargs=dist_kwargs,
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=False,
        )

        # Define Critic Network
        qvalue_net_kwargs = {
            "num_cells": self.config.hidden_sizes,
            "out_features": 1,
            "activation_class": get_activation(self.config.activation),
        }

        qvalue_net = MLP(
            **qvalue_net_kwargs,
        )

        qvalue = ValueOperator(
            in_keys=["action"] + in_keys,
            module=qvalue_net,
        )

        self.model = nn.ModuleDict({
            "policy": actor,
            "value": qvalue
        }).to(self.device)

        # init input dims
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            td = self.train_env.reset()
            td = td.to(self.device)
            for net in self.model.values():
                net(td)
        del td
        self.train_env.close()
    
    def _init_loss_module(self):
        # Create SAC loss
        self.loss_module = SACLoss(
            actor_network=self.model["policy"],
            qvalue_network=self.model["value"],
            num_qvalue_nets=2,
            loss_function=self.config.loss_function,
            delay_actor=False,
            delay_qvalue=True,
            alpha_init=self.config.alpha_init,
        )
        self.loss_module.make_value_estimator(gamma=self.config.gamma)

        # Define Target Network Updater
        self.target_net_updater = SoftUpdate(
            self.loss_module, eps=self.config.target_update_polyak
        )
    
    def _init_optimizer(self):
        critic_params = list(self.loss_module.qvalue_network_params.flatten_keys().values())
        actor_params = list(self.loss_module.actor_network_params.flatten_keys().values())
        
        self.optimizers = {}
        self.optimizers["actor"] = torch.optim.Adam(
            actor_params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_eps,
        )
        self.optimizers["critic"] = torch.optim.Adam(
            critic_params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_eps,
        )
        self.optimizers["alpha"] = torch.optim.Adam(
            [self.loss_module.log_alpha],
            lr=3.0e-4,
        )
    
    def train(self, logger=None):
        # Create off-policy collector
        collector = make_collector(self.config, self.train_env, self.model["policy"])

        # Main loop
        start_time = time.time()
        collected_frames = 0
        pbar = tqdm.tqdm(total=self.config.total_frames)

        init_random_frames = self.config.init_random_frames
        num_updates = int(
            self.config.env_per_collector
            * self.config.frames_per_batch
            * self.config.utd_ratio
        )
        prioritize = self.config.prioritize
        eval_iter = self.config.eval_iter
        frames_per_batch = self.config.frames_per_batch
        eval_rollout_steps = self.config.max_eps_steps

        sampling_start = time.time()
        for i, tensordict in enumerate(collector):
            sampling_time = time.time() - sampling_start

            # Update weights of the inference policy
            collector.update_policy_weights_()

            pbar.update(tensordict.numel())

            tensordict = tensordict.reshape(-1)
            current_frames = tensordict.numel()
            # Add to replay buffer
            self.replay_buffer.extend(tensordict.cpu())
            collected_frames += current_frames

            # Optimization steps
            training_start = time.time()
            if collected_frames >= init_random_frames:
                losses = TensorDict({}, batch_size=[num_updates])
                for i in range(num_updates):
                    # Sample from replay buffer
                    sampled_tensordict = self.replay_buffer.sample()
                    if sampled_tensordict.device != self.device:
                        sampled_tensordict = sampled_tensordict.to(
                            self.device, non_blocking=True
                        )
                    else:
                        sampled_tensordict = sampled_tensordict.clone()

                    # Compute loss
                    loss_td = self.loss_module(sampled_tensordict)

                    actor_loss = loss_td["loss_actor"]
                    q_loss = loss_td["loss_qvalue"]
                    alpha_loss = loss_td["loss_alpha"]

                    # Update actor
                    self.optimizers["actor"].zero_grad()
                    actor_loss.backward()
                    self.optimizers["actor"].step()

                    # Update critic
                    self.optimizers["critic"].zero_grad()
                    q_loss.backward()
                    self.optimizers["critic"].step()

                    # Update alpha
                    self.optimizers["alpha"].zero_grad()
                    alpha_loss.backward()
                    self.optimizers["alpha"].step()

                    losses[i] = loss_td.select(
                        "loss_actor", "loss_qvalue", "loss_alpha"
                    ).detach()

                    # Update qnet_target params
                    self.target_net_updater.step()

                    # Update priority
                    if prioritize:
                        self.replay_buffer.update_priority(sampled_tensordict)

            training_time = time.time() - training_start
            episode_end = (
                tensordict["next", "done"]
                if tensordict["next", "done"].any()
                else tensordict["next", "truncated"]
            )
            episode_rewards = tensordict["next", "episode_reward"][episode_end]

            # Logging
            metrics_to_log = {}
            if len(episode_rewards) > 0:
                episode_length = tensordict["next", "step_count"][episode_end]
                metrics_to_log["train/reward"] = episode_rewards.mean().item()
                metrics_to_log["train/episode_length"] = episode_length.sum().item() / len(
                    episode_length
                )
            if collected_frames >= init_random_frames:
                metrics_to_log["train/q_loss"] = losses.get("loss_qvalue").mean().item()
                metrics_to_log["train/actor_loss"] = losses.get("loss_actor").mean().item()
                metrics_to_log["train/alpha_loss"] = losses.get("loss_alpha").mean().item()
                metrics_to_log["train/alpha"] = loss_td["alpha"].item()
                metrics_to_log["train/entropy"] = loss_td["entropy"].item()
                metrics_to_log["train/sampling_time"] = sampling_time
                metrics_to_log["train/training_time"] = training_time

            # Evaluation
            if abs(collected_frames % eval_iter) < frames_per_batch:
                with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                    eval_start = time.time()
                    eval_rollout = self.eval_env.rollout(
                        eval_rollout_steps,
                        self.model["policy"],
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                    )
                    eval_time = time.time() - eval_start
                    eval_reward = eval_rollout["next", "reward"].mean(-2).mean().item()
                    metrics_to_log["eval/reward"] = eval_reward
                    metrics_to_log["eval/time"] = eval_time
                    print("mean reward", eval_reward)
            if logger is not None:
                log_metrics(logger, metrics_to_log, collected_frames)
            sampling_start = time.time()

        collector.shutdown()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Training took {execution_time:.2f} seconds to finish")
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)


if __name__ == '__main__':
    from lift.rl.utils import gym_env_maker, apply_env_transforms
    from configs import TeacherConfig

    train_env = apply_env_transforms(gym_env_maker('FetchReachDense-v2'))
    eval_env = apply_env_transforms(gym_env_maker('FetchReachDense-v2'))

    config = TeacherConfig()
    model = SAC(config, train_env, eval_env)

    logger = None
    model.train(logger=logger)

    model.save('sac.pth')