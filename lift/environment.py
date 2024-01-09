import gymnasium as gym
import numpy as np

from lift.simulator.simulator import EMGSimulator, FakeSimulator


class EMGWrapper(gym.Wrapper):
    def __init__(self, teacher, config):
        super().__init__(teacher.get_env())
        self.teacher = teacher
        # self.emg_simulator = EMGSimulator(n_channels=config.n_channels, window_size=config.window_size)
        self.emg_simulator = FakeSimulator(num_values=config.n_channels * config.window_size, noise=config.noise)
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
