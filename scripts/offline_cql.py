import torch
from torch import nn

from tensordict import TensorDict
from torchrl.objectives.cql import CQLLoss
from torchrl.collectors import SyncDataCollector, RandomPolicy
from torchrl.data import ReplayBuffer, LazyTensorStorage, SliceSampler
from torchrl.objectives import SoftUpdate

from lift.rl.algo import AlgoBase
from lift.rl.utils import gym_env_maker, apply_env_transforms
from configs import TeacherConfig


def get_buffer(env, policy):
    collector = SyncDataCollector(env, policy, frames_per_batch=10, total_frames=-1)
    rb = ReplayBuffer(
        storage=LazyTensorStorage(10_000),
        sampler=SliceSampler(num_slices=8, traj_key=("collector", "traj_ids"),
                            truncated_key=None, strict_length=False),
                            batch_size=64)
    
    for i, data in enumerate(collector):
        rb.extend(data)
        if i >= 1_000:
            break

    return rb


class CQL(AlgoBase):
    def __init__(self, config, train_env, eval_env):
        super().__init__(config, train_env, eval_env)

        # TODO combine with make_replay_buffer
        policy = RandomPolicy(train_env.action_spec)
        self.replay_buffer = get_buffer(train_env, policy)

    def _init_loss_module(self):
        self.loss_module = CQLLoss(
            actor_network=self.model["policy"],
            qvalue_network=self.model["value"],
            loss_function=self.config.loss_function,
            delay_actor=False,
            delay_qvalue=True,
            alpha_init=self.config.alpha_init,
            action_spec=self.train_env.action_spec,
        )

    def train(self, logger=None):
        # TODO make param
        num_updates = 10_000
        losses = TensorDict({}, batch_size=[num_updates])
        for i in range(num_updates):
            batch = self.replay_buffer.sample()
            loss = self.loss_module(batch)
            # entropy

            actor_loss = loss["loss_actor"] + loss["loss_actor_bc"]
            q_loss = loss["loss_qvalue"] + loss["loss_cql"]
            alpha_loss = loss["loss_alpha"]
            if i % 99 == 0:
                print("actor_loss", actor_loss)
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

            losses[i] = loss.select(
                "loss_actor", "loss_actor_bc", "loss_qvalue", "loss_cql", "loss_alpha"
            ).detach()

            self.target_net_updater.step()


def main():
    train_env = apply_env_transforms(gym_env_maker('FetchReachDense-v2'))
    eval_env = apply_env_transforms(gym_env_maker('FetchReachDense-v2'))

    # TODO create own config
    config = TeacherConfig()
    model = CQL(config, train_env, eval_env)

    model.train()
    # n_obs = train_env.observation_spec["observation"].shape[-1]
    # n_act = train_env.action_spec.shape[-1]


if __name__ == '__main__':
    main()
