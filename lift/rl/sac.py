import time
import tqdm
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives.sac import SACLoss

from lift.rl.algo import AlgoBase
from lift.rl.utils import (
    make_collector, 
    log_metrics,
    make_replay_buffer
)

class SAC(AlgoBase):
    """Soft actor critic trainer"""
    def __init__(self, config, train_env, eval_env):
        super().__init__(config, train_env, eval_env)
            # maybe init replay buffer per algo
        self.replay_buffer = make_replay_buffer(
            batch_size=self.config.batch_size,
            prioritize=self.config.prioritize,
            buffer_size=self.config.replay_buffer_size,
            scratch_dir=self.config.scratch_dir,
            device="cpu",
        )
    
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
    
    # def train(self, logger=None):
    def train_agent(self, logger=None):
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

                    grad_clip = self.config.grad_clip
                    # Update actor
                    actor_loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.optimizers["actor"].param_groups[0]["params"], grad_clip
                        )
                    self.optimizers["actor"].step()
                    self.optimizers["actor"].zero_grad() 

                    # Update critic
                    q_loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.optimizers["critic"].param_groups[0]["params"], grad_clip
                        )
                    self.optimizers["critic"].step()
                    self.optimizers["critic"].zero_grad()                    

                    # Update alpha
                    alpha_loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.optimizers["alpha"].param_groups[0]["params"], grad_clip
                        )
                    self.optimizers["alpha"].step()
                    self.optimizers["alpha"].zero_grad()

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


if __name__ == '__main__':
    from lift.rl.utils import gym_env_maker, apply_env_transforms
    from configs import TeacherConfig

    train_env = apply_env_transforms(gym_env_maker('FetchReachDense-v2'))
    eval_env = apply_env_transforms(gym_env_maker('FetchReachDense-v2'))

    config = TeacherConfig()
    model = SAC(config, train_env, eval_env)

    logger = None
    model.train_agent(logger=logger)

    model.save('sac.pth')