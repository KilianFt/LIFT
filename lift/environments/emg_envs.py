import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import Transform
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs.transforms.transforms import _apply_to_composite

import gymnasium as gym
import numpy as np

from lift.environments.gym_envs import NpGymEnv
from lift.environments.simulator import Simulator, SimulatorFactory
from lift.rl.sac import SAC
                         

class EMGEnv(gym.Wrapper):
    """Environment controlled by emg policy

    - Observations: emg signals generated by emg_simulator-transformed teacher actions
    - Actions: game actions generated by emg policy
    """
    def __init__(self, env: NpGymEnv, teacher: SAC, emg_simulator: Simulator):
        super().__init__(env)
        self.teacher = teacher
        self.emg_simulator = emg_simulator

        # TODO fix this, values can be > 1 and < -1
        if emg_simulator.return_features:
            self.observation_space["emg_observation"] = gym.spaces.Box(
                low=-10,
                high=10,
                shape=(emg_simulator.num_channels * 4,),
                dtype=np.float64
            )
        else:
            self.observation_space["emg_observation"] = gym.spaces.Box(
                low=-3, 
                high=3, 
                shape=(emg_simulator.num_channels, emg_simulator.window_size),
                dtype=np.float64
            )

    def _obs_to_emg(self, obs):
        teacher_action = self.teacher.sample_action(obs).reshape(1, -1)
        emg = self.emg_simulator(teacher_action)[0].numpy() # last action entry not used in fetch env
        return emg

    def reset(self):
        obs = self.env.reset()
        obs["emg_observation"] = self._obs_to_emg(obs)
        return obs

    def step(self, action):
        obs, rwd, done, info = self.env.step(action)
        obs["emg_observation"] = self._obs_to_emg(obs)
        return obs, rwd, done, info
    
    # def get_ideal_action(self, state):
    #     short_state = {key: value for key, value in state.items() if key != "emg_observation"}
    #     return self.teacher.predict(short_state)
    

class EMGTransform(Transform):
    """ Transforms observation to simulated EMG

    Args:
        teacher: teacher model that predicts "ideal" action
        simulator: create EMG signals for each action
        in_keys: keys that are considered for transform
        out_keys: keys after transform

    Example:
        >>> env = apply_env_transforms(gym_env_maker('FetchReachDense-v2'))
        >>> teacher = SAC(...)
        >>> sim = WindowSimulator(...)
        >>> t_emg = EMGTransform(teacher, sim, in_keys=["observation"], out_keys=["emg"])
        >>> env.append_transform(t_emg)
        >>> check_env_specs(env)
        >>> data = env.reset()
        >>> print(data["emg])

    """
    def __init__(self, teacher, simulator: Simulator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.simulator = simulator

    def _apply_transform(self, obs: torch.Tensor) -> None:
        with torch.no_grad():
            loc, scale, action = self.teacher.model.policy(obs)
        emg = self.simulator(action.view(1,-1))
        return emg.squeeze()
    
    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)
    
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return UnboundedContinuousTensorSpec(
            shape=(32,),
            dtype=torch.float32,
            device=observation_spec.device,
        )

if __name__ == '__main__':
    from torchrl.envs.utils import check_env_specs, step_mdp
    from configs import TeacherConfig, BaseConfig
    from lift.rl.utils import gym_env_maker, apply_env_transforms

    config = BaseConfig()
    teach_config = TeacherConfig()

    torchrl_env = apply_env_transforms(gym_env_maker("FetchReachDense-v2"))
    teacher = SAC(teach_config, torchrl_env, torchrl_env)
    
    data_path = (config.mad_data_path / "Female0"/ "training0").as_posix()
    sim = SimualtorFactory(
        data_path,
        config,
        return_features=True,
    )
    
    # test numpy emg env
    env = NpGymEnv("FetchReachDense-v2")
    env = EMGEnv(env, teacher, sim)
    obs = env.reset()
    act = env.action_space.sample()
    next_obs, rwd, done, info = env.step(act)
    assert list(obs.keys()) == ["observation", "emg_observation"]
    assert list(next_obs.keys()) == ["observation", "emg_observation"]
    assert obs["emg_observation"].shape == env.observation_space["emg_observation"].shape
    assert next_obs["emg_observation"].shape == env.observation_space["emg_observation"].shape


    # t_emg = EMGTransform(teacher, sim, in_keys=["observation"], out_keys=["emg"])
    # env.append_transform(t_emg)

    # check_env_specs(env)

    # data = env.reset()
    # for _ in range(5):
    #     action = teacher.model.policy(data)
    #     data = env.step(action)
    #     data = step_mdp(data, keep_other=True)
    #     print(data['step_count'])
    #     print(data['emg'].shape)