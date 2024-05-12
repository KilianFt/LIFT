from typing import Sequence, Tuple
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
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage

from configs import BaseConfig
from lift.environments.teacher_envs import TeacherTransform


class ResizeActionSpace(Transform):
    """ Transforms size of action space by reducing from the last dimension or zero padding

    Args:
        in_keys: keys that are considered for transform
        out_keys: keys after transform
        new_size: target action size

    Example:
        >>> t_act = ExpandAction(in_keys=["action"], out_keys=["action"], new_size=4)
        >>> env.append_transform(t_act)
        >>> check_env_specs(env)
        >>> data = env.reset()
        >>> print(data["emg])

    """
    def __init__(
        self, 
        old_dim: int,
        new_dim: int,
        in_keys: Sequence[str | Tuple[str, ...]] = None, 
        out_keys: Sequence[str | Tuple[str, ...]] | None = None, 
        in_keys_inv: Sequence[str | Tuple[str, ...]] | None = None, 
        out_keys_inv: Sequence[str | Tuple[str, ...]] | None = None,        
    ):
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
        self.old_dim = old_dim
        self.new_dim = new_dim

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] < self.old_dim:
            diff_dim = self.old_dim - action.shape[-1]
            env_action = torch.cat((action, torch.zeros(*action.shape[:-1], diff_dim)), dim=-1)
        elif action.shape[-1] > self.old_dim:
            env_action = action[..., :self.old_dim]
        else:
            env_action = action

        return env_action
    
    def transform_input_spec(self, input_spec):
        full_action_spec = input_spec["full_action_spec"]
        action_dim = full_action_spec["action"].shape
        if action_dim[-1] < self.new_dim:
            diff_dim = self.new_dim - action_dim[-1]
            new_action_dim = list(action_dim)
            new_action_dim[-1] = self.new_dim
            full_action_spec["action"].shape = torch.Size(new_action_dim)
            full_action_spec["action"].space.low = torch.cat(
                [
                    full_action_spec["action"].space.low, 
                    torch.zeros(*action_dim[:-1], diff_dim)
                ], dim=-1,
            )
            full_action_spec["action"].space.high = torch.cat(
                [
                    full_action_spec["action"].space.low, 
                    torch.zeros(*action_dim[:-1], diff_dim)
                ], dim=-1,
            )
        elif action_dim[-1] > self.new_dim:
            new_action_dim = list(action_dim)
            new_action_dim[-1] = self.new_dim
            full_action_spec["action"].shape = torch.Size(new_action_dim)
            full_action_spec["action"].space.low = full_action_spec["action"].space.low[..., :self.new_dim]
            full_action_spec["action"].space.high = full_action_spec["action"].space.high[..., :self.new_dim]
        input_spec["full_action_spec"] = full_action_spec
        return input_spec


"""TODO: maybe figure out better way to pipe configs"""
def gym_env_maker(env_name, config: BaseConfig | None = None, meta=False, cat_obs=True, cat_keys=None, device="cpu"):
    with set_gym_backend("gymnasium"):
        raw_env = GymEnv(env_name, device=device)
        in_keys = ["observation"]
        if cat_obs:
            in_keys = raw_env.observation_spec.keys() if cat_keys is None else cat_keys
            assert all([k in raw_env.observation_spec.keys() for k in in_keys])
        
        transforms = [CatTensors(in_keys=in_keys, out_key="observation")]
        if env_name == "FetchReachDense-v2":
            transforms.append(
                ResizeActionSpace(
                    old_dim=raw_env.action_space.shape[0],
                    new_dim=raw_env.action_space.shape[0] - 1,
                    in_keys_inv=["action"], 
                    out_keys_inv=["action"], 
                )
            )
        
        if config is not None and meta:
            transforms.append(
                TeacherTransform(
                    noise_range=config.noise_range,
                    alpha_range=config.alpha_range,
                )
            )
        env = TransformedEnv(raw_env, Compose(*transforms))
    return env

def apply_env_transforms(env, max_episode_steps=1000):
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_episode_steps),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env

def parallel_env_maker(
        env_name, 
        config=None,
        meta=False,
        cat_obs=True, 
        cat_keys=None,
        env_per_collector=1,
        max_eps_steps=1000,
        device="cpu",
    ):
    parallel_env = ParallelEnv(
        env_per_collector,
        EnvCreator(lambda : gym_env_maker(env_name, config, meta, cat_obs, cat_keys, device)),
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