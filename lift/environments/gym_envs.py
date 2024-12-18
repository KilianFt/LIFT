import numpy as np
import gymnasium as gym
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.box import Box
from gymnasium.utils.step_api_compatibility import step_api_compatibility

class NpGymEnv(gym.Wrapper):
    """Numpy gym environment wrapper for version compatibility"""
    def __init__(self, env_name, cat_obs=True, cat_keys=None, **kwargs):
        """
        Args:
            env_name (str): gym environment name used for gym.make()
            cat_obs (bool): whether to concatenate dict observations
            cat_keys (list, none): list of keys to concatenate. 
                observations will be concatenated in the order of supplied keys.
                if none and cat_obs=True, use sorted dict keys.
        """
        super().__init__(gym.make(env_name, **kwargs))
        self.env_name = env_name
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.cat_obs = cat_obs
        if not isinstance(self.observation_space, Dict):
            self.cat_obs = False
        
        if self.cat_obs:
            obs_keys = sorted(list(self.observation_space.spaces.keys()))
            self.cat_keys = cat_keys if cat_keys is not None else obs_keys
            assert all([k in obs_keys for k in self.cat_keys]), "Observation keys to concat not in env key set"

            # modify observation space
            obs_dim = sum([self.observation_space[k].shape[0] for k in self.cat_keys]) # assume vector observations
            self.observation_space = Dict(
                {
                    "observation": gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
                    )
                }
            )

        # special handle action space
        if env_name == "FetchReachDense-v2":
            new_act_dim = self.action_space.shape[-1] - 1
            self.action_space = resize_gym_box_space(self.action_space, new_act_dim)
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = self.concat_obs_dict(obs)
        return obs
    
    def step(self, action):
        # special handle action space
        if self.env_name == "FetchReachDense-v2":
            action = np.concatenate([action, np.zeros_like(action[..., :1])], axis=-1)
            
        next_obs, rwd, done, info = step_api_compatibility(
            self.env.step(action), 
            output_truncation_bool=False
        )
        next_obs = self.concat_obs_dict(next_obs)
        return next_obs, rwd, done, info
    
    def concat_obs_dict(self, obs):
        if self.cat_obs:
            obs = {"observation": np.concatenate([obs[k] for k in self.cat_keys], axis=-1)}
        return obs


def resize_gym_box_space(space: Box, new_dim: int):
    new_shape = list(space.shape)
    new_shape[-1] = new_dim
    if new_dim < space.shape[-1]:
        new_space = Box(
            low=space.low[..., :-1], 
            high=space.high[..., :-1], 
            shape=tuple(new_shape),
            dtype=space.dtype,
        )
    elif new_dim > space.shape[-1]:
        diff_dim = new_dim - space.shape[-1]
        box_shape = list(space.low.shape[:-1]) + [diff_dim]
        new_space = Box(
            low=np.concatenate([space.low, np.ones(box_shape) * -np.inf]), 
            high=np.concatenate([space.high, np.ones(box_shape) * np.inf]), 
            shape=tuple(new_shape),
            dtype=space.dtype,
        )
    else:
        new_space = space

    return new_space

if __name__ == "__main__":
    # test basic compatibility
    env = NpGymEnv("FetchReachDense-v2", cat_obs=False)
    obs = env.reset()
    act = env.action_space.sample()
    next_obs, rwd, done, info = env.step(act)
    assert list(obs.keys()) == ["observation", "achieved_goal", "desired_goal"]
    assert list(next_obs.keys()) == ["observation", "achieved_goal", "desired_goal"]

    # test naive cat obs
    env = NpGymEnv("FetchReachDense-v2", cat_obs=True)
    obs = env.reset()
    act = env.action_space.sample()
    next_obs, rwd, done, info = env.step(act)
    assert list(obs.keys()) == ["observation"]
    assert list(next_obs.keys()) == ["observation"]
    assert env.observation_space["observation"].shape == (16,)
    assert obs["observation"].shape == (16,)
    assert next_obs["observation"].shape == (16,)

    # test custom cat obs
    env = NpGymEnv("FetchReachDense-v2", cat_obs=True, cat_keys=["observation", "desired_goal"])
    obs = env.reset()
    act = env.action_space.sample()
    next_obs, rwd, done, info = env.step(act)
    assert list(obs.keys()) == ["observation"]
    assert list(next_obs.keys()) == ["observation"]
    assert env.observation_space["observation"].shape == (13,)
    assert obs["observation"].shape == (13,)
    assert next_obs["observation"].shape == (13,)