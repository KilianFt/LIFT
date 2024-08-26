from typing import Sequence
import copy
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import Transform
from torchrl.envs import Compose
from torchrl.modules import TanhNormal

import gymnasium as gym
import numpy as np

from lift.environments.gym_envs import NpGymEnv, resize_gym_box_space
from lift.rl.sac import SAC
from lift.rl.sac_meta import MetaSAC

def compute_noise_scale(action: np.ndarray | torch.Tensor, base_noise: float, slope: float):
    """Linear scale noise from base_noise up to 2"""
    abs_action = np.abs(action) if isinstance(action, np.ndarray) else action.abs()
    noise = base_noise + abs_action * slope 
    return noise.clip(0., 2.)

def compute_alpha_scale(obs: np.ndarray, max_alpha: float, apply_range: list[float]):
    """Linear scale alpha from 1 to max_alpha"""
    cur_pos = obs[..., :3]
    goal = obs[..., 10:13]
    dist_goal = np.abs(cur_pos - goal).clip(0., apply_range[1]) - apply_range[0]
    alpha = 1 + (max_alpha - 1) / (apply_range[1] - apply_range[0]) * dist_goal
    alpha = alpha.clip(1., max_alpha)
    return alpha

def apply_gaussian_drift(z: np.ndarray, offset: float, std: float, range=[-np.inf, np.inf]):
    """Apply gaussian drift to variable z and clip to bound"""
    drift = np.random.normal(offset, std)
    z_new = np.clip(z + drift, range[0], range[1])
    return z_new


class TeacherEnv(gym.Wrapper):
    """Environment used to meta train teacher

    - Observations: game observation corrupted by emg policy
    - Actions: game actions generated by teacher
    """
    def __init__(
        self, 
        env: NpGymEnv, 
        noise_range: list[float] | None = [0., 1.], 
        noise_slope_range: list[float] | None = [0., 1.], 
        append_obs: bool = True,
    ):
        """
        Args:
            noise_range (list[float] | None): range of noise applied to corrupt teacher actions
            noise_slope_range (list[float] | None): range of slope of action magnitude dependent noise
            append_obs (bool): whether to append noise level as observation
        """
        super().__init__(env)
        self.noise_range = noise_range
        self.noise_slope_range = noise_slope_range
        self.append_obs = append_obs

        # add meta variables to observation space
        new_obs_dim = self.observation_space["observation"].shape[-1] 
        new_obs_dim += sum([noise_range != None, noise_slope_range != None])
        self.observation_space["observation"] = resize_gym_box_space(
            self.observation_space["observation"], new_obs_dim
        )

    def reset(self):
        obs = self.env.reset()

        # sample meta vars
        self.noise = None
        if self.noise_range is not None:
            noise = np.random.uniform(self.noise_range[0], self.noise_range[1])
            self.noise = np.array([noise], dtype=self.observation_space["observation"].dtype)
        
        self.noise_slope = 0.
        if self.noise_slope_range is not None:
            noise_slope = np.random.uniform(self.noise_slope_range[0], self.noise_slope_range[1])
            self.noise_slope = np.array([noise_slope], dtype=self.observation_space["observation"].dtype)
        
        if self.noise_range is not None and self.append_obs:
            obs["observation"] = np.concatenate([obs["observation"], self.noise])

        if self.noise_slope_range is not None and self.append_obs:
            obs["observation"] = np.concatenate([obs["observation"], self.noise_slope])
        return obs

    def step(self, action: np.ndarray):
        if self.noise_range is not None:
            noise = compute_noise_scale(action, self.noise, self.noise_slope)
            eps = np.random.normal(size=action.shape) * noise
            decoded_action = action + eps
        else:
            decoded_action = action.copy()

        obs, rwd, done, info = self.env.step(decoded_action)
        info["noise"] = self.noise
        info["noise_slope"] = self.noise_slope

        if self.noise is not None and self.append_obs:
            obs["observation"] = np.concatenate([obs["observation"], self.noise])
        if self.noise_slope is not None and self.append_obs:
            obs["observation"] = np.concatenate([obs["observation"], self.noise_slope])
        return obs, rwd, done, info


class TeacherTransform(Transform):
    """Transforms observation to add meta teacher variables"""
    def __init__(
        self,
        noise_range: list[float] | None,
        noise_slope_range: list[float] | None,
        in_keys: Sequence[str] | None = None,
        out_keys: Sequence[str] | None = None,
        in_keys_inv: Sequence[str] | None = None,
        out_keys_inv: Sequence[str] | None = None,
    ):
        """
        Args:
            noise_range (list[float] | None): range of noise applied to corrupt teacher actions
            noise_slope_range (list[float] | None): range of slope of action magnitude dependent noise
        """
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
        self.noise_range = noise_range
        self.noise_slope_range = noise_slope_range

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        new_obs = obs.clone()
        if self.noise is not None:
            new_obs = torch.cat([new_obs, self.noise], dim=-1)
        if self.noise_slope is not None:
            new_obs = torch.cat([new_obs, self.noise_slope], dim=-1)
        return new_obs
    
    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        if self.noise_range is not None:
            noise = compute_noise_scale(action, self.noise, self.noise_slope)
            eps = torch.randn_like(action) * noise
            decoded_action = action + eps
        else:
            decoded_action = action.clone()
        return decoded_action
    
    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        next_tensordict["observation"] = self._apply_transform(next_tensordict["observation"])
        return next_tensordict
    
    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        self.noise = None
        if self.noise_range is not None:
            self.noise = torch.rand(1).uniform_(self.noise_range[0], self.noise_range[1])

        self.noise_slope = torch.zeros(1)
        if self.noise_slope_range is not None:
            self.noise_slope = torch.rand(1).uniform_(self.noise_slope_range[0], self.noise_slope_range[1])
        
        if self.noise_range is not None:
            tensordict_reset["observation"] = torch.cat([tensordict_reset["observation"], self.noise], dim=-1)

        if self.noise_slope_range is not None:
            tensordict_reset["observation"] = torch.cat([tensordict_reset["observation"],
                                                self.noise_slope], dim=-1)
        return tensordict_reset
    
    def transform_observation_spec(self, observation_spec):
        new_obs_dim = list(observation_spec["observation"].shape)
        new_obs_dim[-1] += sum([self.noise_range != None, self.noise_slope_range != None])
        observation_spec["observation"].shape = torch.Size(new_obs_dim)
        return observation_spec


class ConditionedTeacher:
    """Wrapper to reset teacher with random meta variables for simulated trajectories
    Should be used with raw environment
    """
    def __init__(
        self, 
        teacher: SAC | MetaSAC, 
        noise_range: list[float] | None = [0., 1.], 
        noise_slope_range: list[float] | None = [0., 1.], 
        alpha_range: list[float] | None = [0., 1.], 
        alpha_apply_range: list[float] | None = [1., 3.],
        user_bias: float | None = None,
    ):
        """
        Args:
            noise_range (list[float] | None): range of base noise to condition the teacher on
            noise_slope_range (list[float] | None): range of slope of action magnitude dependent noise
            alpha_range (list[float] | None): range of base alpha to modify teacher action sampling
            alpha_apply_range (list[float] | None): range of goal distance to apply alpha scaling
        """
        self.noise_range = noise_range
        self.noise_slope_range = noise_slope_range
        self.alpha_range = alpha_range
        self.alpha_apply_range = alpha_apply_range
        self.user_bias = user_bias

        self.teacher = teacher

    def reset(self):
        self.noise = None
        self.noise_slope = None
        self.alpha = None
        if self.noise_range is not None:
            noise = np.random.uniform(self.noise_range[0], self.noise_range[1])
            self.noise = np.array([noise])
        if self.noise_slope_range is not None:
            noise_slope = np.random.uniform(self.noise_slope_range[0], self.noise_slope_range[1])
            self.noise_slope = np.array([noise_slope])
        if self.alpha_range is not None:
            self.alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
    
    def get_meta_vars(self):
        return dict(noise=self.noise, noise_slope=self.noise_slope, alpha=self.alpha)
    
    def set_meta_vars(self, meta_vars):
        self.noise = meta_vars["noise"]
        if not isinstance(meta_vars["noise"], np.ndarray):
            self.noise = np.array([self.noise]) 
        self.noise_slope = meta_vars["noise_slope"]
        if not isinstance(meta_vars["noise_slope"], np.ndarray):
            self.noise_slope = np.array([self.noise_slope])
        self.alpha = meta_vars["alpha"]
        if not isinstance(meta_vars["alpha"], np.ndarray):
            self.alpha = np.array(self.alpha)

    def get_action_dist(self, obs):
        obs_ = copy.deepcopy(obs)

        if self.noise is not None:
            noise = np.ones_like(obs_["observation"][..., :1]) * self.noise
            obs_["observation"] = np.concatenate([obs_["observation"], noise], axis=-1)
        if self.noise_slope is not None:
            noise_slope = np.ones_like(obs_["observation"][..., :1]) * self.noise_slope
            obs_["observation"] = np.concatenate([obs_["observation"], noise_slope], axis=-1)
        act_dist = self.teacher.get_action_dist(obs_)

        if self.alpha is not None:
            alpha = compute_alpha_scale(obs["observation"], self.alpha, self.alpha_apply_range)
            new_scale = act_dist.scale * alpha
        else:
            new_scale = act_dist.scale

        new_loc = act_dist.loc
        if self.user_bias is not None:
            new_loc += self.user_bias
        act_dist = TanhNormal(loc=new_loc, scale=new_scale,
                              upscale=act_dist.upscale, min=act_dist.min, max=act_dist.max)
        return act_dist
    
    def sample_action(self, obs, sample_mean=False):
        act_dist = self.get_action_dist(obs)
        
        if sample_mean:
            act = act_dist.mode
        else:
            act = act_dist.sample()
        return act.numpy()


if __name__ == "__main__":
    from torchrl.envs import TransformedEnv
    from lift.rl.env_utils import gym_env_maker, apply_env_transforms
    from configs import BaseConfig
    np.random.seed(0)
    torch.manual_seed(0)

    # test np env
    env = NpGymEnv("FetchReachDense-v2")
    noise_range = [0.001, 1.]
    noise_slope_range = [0.001, 1.]
    obs_dim = env.observation_space["observation"].shape[-1] + 2
    env = TeacherEnv(env, noise_range=noise_range, noise_slope_range=noise_slope_range)
    
    obs = env.reset()
    act = env.action_space.sample()
    next_obs, rwd, done, info = env.step(act)
    print(next_obs["observation"].shape)
    assert env.observation_space["observation"].shape == (obs_dim,)
    assert obs["observation"].shape == (obs_dim,)
    assert next_obs["observation"].shape == (obs_dim,)

    # test torchrl transform
    env = gym_env_maker("FetchReachDense-v2")
    env = TransformedEnv(env, Compose(TeacherTransform(
        noise_range, 
        noise_slope_range,
        in_keys=["observation"], 
        out_keys=["observation"], 
        in_keys_inv=["action"], 
        out_keys_inv=["action"],
    )))
    
    action = TensorDict({"action": torch.randn(3)})
    obs1 = env.reset().clone()
    next_obs1 = env.step(action).clone()
    obs2 = env.reset().clone()
    next_obs2 = env.step(action).clone()
    assert obs1["observation"].shape == (obs_dim,)
    assert next_obs1["next"]["observation"].shape == (obs_dim,)
    assert obs2["observation"].shape == (obs_dim,)
    assert next_obs2["next"]["observation"].shape == (obs_dim,)
    assert (obs1["observation"][-2:] == next_obs1["next"]["observation"][-2:]).all()
    assert (obs2["observation"][-2:] == next_obs2["next"]["observation"][-2:]).all()
    assert (obs1["observation"][-2:] != obs2["observation"][-2:]).all()

    # test conditional teacher
    config = BaseConfig()
    env = NpGymEnv("FetchReachDense-v2")
    torchrl_env = apply_env_transforms(gym_env_maker("FetchReachDense-v2", config, meta=True))
    teacher = SAC(config.teacher, torchrl_env, torchrl_env)

    noise_range = [0.001, 1.]
    noise_slope_range = [0.001, 1.]
    alpha_range = [1., 3.]
    alpha_apply_range = [1., 3.]
    conditioned_teacher = ConditionedTeacher(
        teacher, noise_range, noise_slope_range, alpha_range, alpha_apply_range
    )

    obs = env.reset()
    conditioned_teacher.reset()
    act = conditioned_teacher.sample_action(obs)
    assert act.shape == (3,)