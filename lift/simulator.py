import random

import numpy as np
import matplotlib.pyplot as plt

# TODO
# - how to handle action transitions
# - simultaneous actions should be superposition of single ones


def get_action_params(action, num_values):
    seed = hash(tuple(action))
    random.seed(seed)
    params = [random.random() for _ in range(num_values)]
    return params


class ParametrizedSin:
    def __call__(self, x, action):
        height, amplitude, phase_shift, period = get_action_params(action, 4)
        return height + amplitude * np.sin(phase_shift + period * x)


class EMGSimulator:
    def __init__(self, n_channels: int):
        self.channels = [ParametrizedSin() for _ in range(n_channels)]
        self.timestep = 0
        self.step_size = 0.1
        self.sampling_frequency = 20

    def sample(self, action, n_timesteps: int):
        start_time = self.timestep
        end_time = start_time + self.step_size * n_timesteps
        ts = np.arange(start_time, end_time, step=self.step_size)

        self.timestep = end_time
        window = np.stack([ch(ts, action+ch_id) for ch_id, ch in enumerate(self.channels)])

        return window
    

if __name__ == '__main__':
    emg_sim = EMGSimulator(4)
    emgs = [emg_sim.sample(np.array([1,0,1]), 100) for _ in range(3)]
    emgs += [emg_sim.sample(np.array([1,0,1,1]), 100) for _ in range(3)]
    emgs += [emg_sim.sample(np.array([5,2,0,1,1]), 100) for _ in range(3)]
    emg_series = np.concatenate(emgs, axis=1)

    for emg_channel in emg_series:
        plt.plot(emg_channel)
    plt.show()
