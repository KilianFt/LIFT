import time

import torch
import wandb

from tensordict import TensorDict
from torchrl.objectives.cql import CQLLoss
from torchrl.collectors import SyncDataCollector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.data import ReplayBuffer, LazyTensorStorage, SliceSampler

from lift.rl.algo import AlgoBase
from lift.rl.sac import SAC
from lift.rl.utils import gym_env_maker, apply_env_transforms, log_metrics
from configs import TeacherConfig


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
        super().__init__(config, eval_env, eval_env)

        self.replay_buffer = replay_buffer

        # if encoder is not None:
        #     # replace policy with encoder


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
        # TODO make param
        num_updates = 10_000
        for train_step in range(num_updates):
            batch = self.replay_buffer.sample()
            loss = self.loss_module(batch)
            # entropy

            actor_loss = loss["loss_actor"] + loss["loss_actor_bc"]
            q_loss = loss["loss_qvalue"] + loss["loss_cql"]
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

            # losses[i] = loss.select(
            #     "loss_actor", "loss_actor_bc", "loss_qvalue", "loss_cql", "loss_alpha"
            # ).detach()

            self.target_net_updater.step()

            metrics_to_log = loss.to_dict()

            # Evaluation
            # TODO make param
            eval_iter = 99
            if train_step % eval_iter == 0:
                # TODO make param
                eval_rollout_steps = 100
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
                log_metrics(logger, metrics_to_log, train_step)

            if use_wandb:
                wandb.log(metrics_to_log)


if __name__ == '__main__':
    # TODO create own config
    config = TeacherConfig()

    train_env = apply_env_transforms(gym_env_maker(config.env_name))
    eval_env = apply_env_transforms(gym_env_maker(config.env_name))

    collector_model = SAC(config, train_env, eval_env)
    collector_model.load('sac.pth')
    policy = collector_model.model["policy"]
    # policy = RandomPolicy(train_env.action_spec)

    replay_buffer = record_buffer(train_env, policy)

    # logger = 
    model = CQL(config, replay_buffer, eval_env)
    model.train()