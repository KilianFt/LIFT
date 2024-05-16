import time

import wandb
import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.objectives.cql import CQLLoss
from torchrl.collectors import SyncDataCollector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.data import ReplayBuffer, LazyTensorStorage, SliceSampler

from lift.rl.algo import AlgoBase
from lift.rl.sac import SAC
from lift.rl.utils import gym_env_maker, apply_env_transforms, log_metrics
from configs import OfflineRLConfig


def record_buffer(env, policy):
    collector = SyncDataCollector(env, policy, frames_per_batch=10, total_frames=-1)
    rb = ReplayBuffer(
        storage=LazyTensorStorage(50_000),
        sampler=SliceSampler(num_slices=8, traj_key=("collector", "traj_ids"),
                            truncated_key=None, strict_length=False),
                            batch_size=64)
    
    for i, data in enumerate(collector):
        rb.extend(data)
        if i >= 5_000:
            break

    return rb


class CQL(AlgoBase):
    def __init__(self, config, replay_buffer, eval_env, encoder=None):
        in_keys = ["emg"]
        super().__init__(config, eval_env, eval_env, in_keys=in_keys)

        self.replay_buffer = replay_buffer
        self.bellman_scaling = torch.tensor(config.bellman_scaling).to(self.device)
        self.bc_regularization = torch.tensor(config.bc_regularization).to(self.device)
        if encoder is not None:
            # replace policy with encoder
            actor_extractor = NormalParamExtractor(
                scale_mapping=f"biased_softplus_{config.default_policy_scale}",
                scale_lb=config.scale_lb,
            )
            actor_net = nn.Sequential(encoder.base.mlp, actor_extractor)
            encoder_actor_module = TensorDictModule(
                actor_net,
                in_keys=in_keys,
                out_keys=[
                    "loc",
                    "scale",
                ],
            )
            self.model.policy.module[0].module = encoder_actor_module

    def _init_loss_module(self):
        self.loss_module = CQLLoss(
            actor_network=self.model["policy"],
            qvalue_network=self.model["value"],
            loss_function=self.config.loss_function,
            delay_actor=False,
            delay_qvalue=True,
            alpha_init=self.config.alpha_init,
            action_spec=self.eval_env.action_spec,
        )

    def train(self, logger=None, use_wandb=False):
        for train_step in range(self.config.num_updates):
            batch = self.replay_buffer.sample()
            loss = self.loss_module(batch)
            # entropy

            # choose to add bc reqularization or not
            actor_loss = loss["loss_actor"]# + loss["loss_actor_bc"]
            q_loss = loss["loss_cql"] + self.bellman_scaling * loss["loss_qvalue"]
            alpha_loss = loss["loss_alpha"]

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

            self.target_net_updater.step()

            metrics_to_log = loss.to_dict()

            # Evaluation
            if train_step % self.config.eval_iter == 0:
                with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                    eval_start = time.time()
                    eval_rollout = self.eval_env.rollout(
                        self.config.eval_rollout_steps,
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
                log_metrics(logger, metrics_to_log, train_step)

            if use_wandb:
                wandb.log(metrics_to_log)


if __name__ == '__main__':
    config = OfflineRLConfig()

    train_env = apply_env_transforms(gym_env_maker(config.env_name))
    eval_env = apply_env_transforms(gym_env_maker(config.env_name))

    collector_model = SAC(config, train_env, eval_env)
    collector_model.load('sac.pth')
    policy = collector_model.model["policy"]
    # policy = RandomPolicy(train_env.action_spec)

    replay_buffer = record_buffer(train_env, policy)

    model = CQL(config, replay_buffer, eval_env)
    model.train()