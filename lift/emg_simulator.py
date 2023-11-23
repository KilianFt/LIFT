import numpy as np
import matplotlib.pyplot as plt


class ParametrizedSin:
    def __init__(self, amplitude: float = .1, phase_shift: float = 0., period: float = 1., height: float = 0.):
        self.amplitude = amplitude
        self.phase_shift = phase_shift
        self.period = period
        self.height = height

    def __call__(self, x):
        return self.height + self.amplitude * np.sin(self.phase_shift + self.period * x)
    

# TODO
# - parameterize depeding on action
# - how to handle action transitions
# - what is the input of sample? a sequence of actions?

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

        return np.stack([ch(ts) for ch in self.channels])
    

if __name__ == '__main__':
    emg_sim = EMGSimulator(2)
    emgs = [emg_sim.sample(None, 10) for _ in range(3)]
    emg_series = np.concatenate(emgs, axis=1)
    plt.plot(emg_series[1,:])
    plt.show()
