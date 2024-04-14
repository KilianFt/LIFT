import torch
import torch.nn as nn
from tensordict import TensorDictBase
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs import (
    CatTensors,
    Compose,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    TransformedEnv,
)
from torchrl.envs.transforms import (
    InitTracker, 
    RewardSum, 
    StepCounter,
)
from torchrl.envs import Transform
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage


class ExpandAction(Transform):
    """ Transforms action from 3D to 4D by adding a zero column

    Args:
        in_keys: keys that are considered for transform
        out_keys: keys after transform

    Example:
        >>> t_act = ExpandAction(in_keys=["action"], out_keys=["action"])
        >>> env.append_transform(t_act)
        >>> check_env_specs(env)
        >>> data = env.reset()
        >>> print(data["emg])

    """
    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        return next_tensordict

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] == 3:
            env_action = torch.cat((action, torch.zeros(*action.shape[:-1], 1)), dim=-1)
        else:
            env_action = action

        return env_action
    
    @_apply_to_composite
    def transform_action_spec(self, action_spec):
        return UnboundedContinuousTensorSpec(
            shape=(3,),
            dtype=torch.float32,
            device=action_spec.device,
        )

def gym_env_maker(env_name, cat_obs=True, cat_keys=None, device="cpu"):
    with set_gym_backend("gymnasium"):
        raw_env = GymEnv(env_name, device=device)
        in_keys = ["observation"]
        if cat_obs:
            in_keys = raw_env.observation_spec.keys() if cat_keys is None else cat_keys
            assert all([k in raw_env.observation_spec.keys() for k in in_keys])
        
        env = TransformedEnv(
            raw_env, 
            CatTensors(in_keys=in_keys, out_key="observation")
        )
    return env

def apply_env_transforms(env, max_episode_steps=1000):
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_episode_steps),
            DoubleToFloat(),
            RewardSum(),
            ExpandAction(in_keys_inv=["action"], out_keys_inv=["action"]),
        ),
    )
    return transformed_env

def parallel_env_maker(
        env_name, 
        cat_obs=True, 
        cat_keys=None,
        env_per_collector=1,
        max_eps_steps=1000,
        device="cpu",
    ):
    parallel_env = ParallelEnv(
        env_per_collector,
        EnvCreator(lambda : gym_env_maker(env_name, cat_obs, cat_keys, device)),
        serial_for_single=True,
    )
    parallel_env = apply_env_transforms(parallel_env, max_eps_steps)
    return parallel_env

def make_collector(config, train_env, actor_model_explore):
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=config.init_random_frames,
        frames_per_batch=config.frames_per_batch,
        total_frames=config.total_frames,
        device=config.device,
    )
    collector.set_seed(config.seed)
    return collector

"""TODO: maybe unify offline and online replay buffer"""
def make_replay_buffer(
    batch_size,
    prioritize=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    if prioritize:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    return replay_buffer

def get_activation(activation):
    if activation == "relu":
        return nn.ReLU
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError
    
def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)