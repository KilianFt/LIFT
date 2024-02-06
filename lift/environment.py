import gymnasium as gym
import numpy as np
import torch

from lift.simulator.simulator import WindowSimulator

"""TODO: handle different gym versions more cleanly, maybe don't use agent.get_env()"""

class EMGWrapper(gym.Wrapper):
    def __init__(self, teacher, config):
        super().__init__(teacher.get_env())
        self.teacher = teacher
        data_path = './datasets/MyoArmbandDataset/PreTrainingDataset/Female0/training0/'
        sim = WindowSimulator(num_actions=6, num_bursts=1, num_channels=8, window_size=200)
        sim.fit_params_to_mad_sample(data_path, desired_labels=[1, 2, 3, 4, 5, 6])

        # TODO fix this, values can be > 1 and < -1
        self.observation_space["observation"] = gym.spaces.Box(low=-1, high=1,
                                                               shape=(config.n_channels, config.window_size),
                                                               dtype=np.float64)

    def _obs_to_emg(self, state):
        ideal_action, _ = self.teacher.predict(state)
        return self.emg_simulator(ideal_action)

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


class UserSimulator(gym.Wrapper):
    """Simulator maps intended action to emg and back to decoded action"""
    def __init__(self, env, decoder, config):
        super().__init__(env)
        self.decoder = decoder
        self.emg_simulator = FakeSimulator(
            action_size=config.action_size, 
            features_per_action=config.n_channels, 
            noise=config.noise
        )

    def reset(self, seed=0):
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        emg = self.emg_simulator(action[None])
        decoded_action = self.decoder.sample_action({'emg_observation': emg})[-1]
        state, reward, terminated, truncated, info = self.env.step(decoded_action)
        return state, reward, terminated, truncated, info
