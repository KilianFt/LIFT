import gymnasium as gym
import numpy as np
import torch

from lift.simulator.simulator import WindowSimulator
from lift.utils import compute_features

"""TODO: handle different gym versions more cleanly, maybe don't use agent.get_env()"""

class EMGWrapper(gym.Wrapper):
    def __init__(self, teacher, emg_simulator):
        super().__init__(teacher.get_env())
        self.teacher = teacher
        self.emg_simulator = emg_simulator

        # TODO fix this, values can be > 1 and < -1
        self.observation_space["observation"] = gym.spaces.Box(
            low=-3, 
            high=3, 
            shape=(emg_simulator.num_channels, emg_simulator.window_size),
            dtype=np.float64
        )

    def _obs_to_emg(self, state):
        ideal_action, _ = self.teacher.predict(state)
        # last action entry not used in fetch env
        out = self.emg_simulator(ideal_action)
        return out

    def reset(self):
        state = self.env.reset()
        state["emg_observation"] = self._obs_to_emg(state)
        return state

    def step(self, action):
        state, reward, terminated, info = self.env.step(action)
        state["emg_observation"] = self._obs_to_emg(state)
        return state, reward, terminated, info
    
    def get_ideal_action(self, state):
        short_state = {key: value for key, value in state.items() if key != "emg_observation"}
        return self.teacher.predict(short_state)


"""TODO: unify with EMGWrapper"""
class UserSimulator(gym.Wrapper):
    """Simulator maps intended action to emg and back to decoded action"""
    def __init__(self, env, decoder, config):
        super().__init__(env)
        self.decoder = decoder
        self.emg_simulator = WindowSimulator(num_actions=6,#FIXME make param
                                             num_bursts=config.n_bursts,
                                             num_channels=config.n_channels,
                                             window_size=config.window_size,
                                             noise=config.noise)

    def reset(self, seed=0):
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        emg = self.emg_simulator(action[None])
        decoded_action = self.decoder.sample_action({'emg_observation': emg})[-1]
        state, reward, terminated, truncated, info = self.env.step(decoded_action)
        return state, reward, terminated, truncated, info
