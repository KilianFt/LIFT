import random
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.distributions as torch_dist

from libemg.feature_extractor import FeatureExtractor

# TODO
# - how to handle action transitions
# - simultaneous actions should be superposition of single ones


def get_action_params(action, num_values):
    seed = hash(tuple(action))
    random.seed(seed)
    params = [random.random() for _ in range(num_values)]
    return params


class FakeSimulator:
    def __init__(self, num_values=10, noise=0.1):
        self.num_values = num_values
        self.noise = noise

    def __call__(self, actions):
        features = []
        for action in actions:
            params = get_action_params(action, num_values=self.num_values)
            params = np.array(params)
            params += np.random.rand(*params.shape) * self.noise
            features.append(params)

        return np.stack(features)


class ParametrizedSin:
    def __call__(self, x, action):
        height, amplitude, phase_shift, period = get_action_params(action, 4)
        return height + amplitude * np.sin(phase_shift + period * x)


class EMGSimulator:
    def __init__(self, n_channels: int, window_size: int):
        self.channels = [ParametrizedSin() for _ in range(n_channels)]
        self.timestep = 0
        self.step_size = 0.1
        self.sampling_frequency = 20
        self.n_channels = n_channels
        self.window_size = window_size

    def __call__(self, actions):
        windows = []
        for action in actions:
            start_time = self.timestep
            end_time = start_time + self.step_size * self.window_size
            ts = np.arange(start_time, end_time, step=self.step_size)

            self.timestep = end_time
            window = np.stack([ch(ts, action+ch_id) for ch_id, ch in enumerate(self.channels)])
            windows.append(window)
        return np.stack(windows)


class NormalSimulator:
    '''things that change are:
        - ground noise
        - burst size
        - burst frequency
        - burst duration
        - induced noise'''

    def __init__(self, window_size=200) -> None:
        self.window_size = window_size

    def __call__(self, action):
        # TODO get parameters depending on action
        split_probabilities = np.array([0.4, 0.3, 0.3])
        num_splits = np.random.choice(np.arange(1,4), p=split_probabilities)
        bias_range = 0.1
        emg_range = 1
        noise = 0.1

        biases = np.random.uniform(-bias_range, bias_range, num_splits + 1)
        limits = np.random.uniform(-emg_range, emg_range, num_splits + 1)

        # TODO make sure limits are never really small
        cutoffs = random.choices(np.arange(0, self.window_size), k=num_splits)
        cutoffs.sort()
        cutoffs.append(self.window_size)
        num_samples_list = [cutoffs[i] - cutoffs[i-1] if i > 0 else cutoffs[i] for i in range(len(cutoffs))]

        window_parts = []
        for limit, bias, num_samples in zip(limits, biases, num_samples_list):
            window_part = np.random.uniform(-limit, limit, num_samples) + bias
            window_parts.append(window_part)

        window = np.concatenate(window_parts)

        # induce noise
        window += np.random.normal(0, noise, len(window))

        return window


class WindowSimulator:
    """Assume equal burst durations, only tune ranges and biases"""
    def __init__(self, num_actions=5, num_bursts=3, num_channels=8, window_size=200):
        self.num_actions = num_actions
        self.num_bursts = num_bursts
        self.num_channels = num_channels
        self.window_size = window_size
        self.burst_durations = [window_size // num_bursts] * (self.num_bursts - 1)
        self.burst_durations = self.burst_durations + [window_size - int(np.sum(self.burst_durations))]
        
        # init params
        self.bias_range = torch.ones(self.num_bursts, self.num_actions, self.num_channels)
        self.emg_range = torch.ones(self.num_bursts, self.num_actions, self.num_channels)
        self.noise = 0.1

        self.bias_range_shape = [self.num_bursts, self.num_actions, self.num_channels]
        self.emg_range_shape = [self.num_bursts, self.num_actions, self.num_channels]
    
    def set_params(self, bias_range, emg_range):
        self.bias_range = bias_range
        self.emg_range = emg_range
    
    def transform_params(self, bias_range, emg_range):
        # TODO limits should never be > 1
        # do we need random limits / biases when we have a Uniform distribution?
        bias_range = 0.1 * torch.relu(bias_range + torch.randn_like(bias_range) * self.noise) + 1e-5
        emg_range = F.relu(emg_range + torch.randn_like(emg_range) * self.noise) + 1e-5
        return bias_range, emg_range
    
    def __call__(self, actions):
        batch_size = len(actions)

        # biases = torch_dist.Uniform(-self.bias_range, self.bias_range).sample()
        # limits = torch_dist.Uniform(torch.zeros_like(self.emg_range) + 1e-6, self.emg_range).sample()
        biases, limits = self.transform_params(self.bias_range, self.emg_range)

        window_parts = [None] * self.num_bursts
        for i in range(self.num_bursts):
            window_parts[i] = torch_dist.Uniform(-limits[i], limits[i]).sample((batch_size, self.burst_durations[i],)) + biases[i]
        window = torch.cat(window_parts, dim=-3)

        window = torch.einsum("ntac, na -> ntc", window, actions).transpose(-1, -2)
        # window += torch.randn_like(window) * self.noise
        return window
    
    def compute_features(self, windows, feature_list=['MAV', 'SSC', 'ZC', 'WL']):
        features = FeatureExtractor().extract_features(
            feature_list, 
            windows.numpy()
        )
        features = np.stack(list(features.values()), axis=-1)
        features = torch.from_numpy(features).flatten(start_dim=1).to(torch.float32)
        return features


if __name__ == '__main__':
    simulator_type = 'normal' # 'sin'
    
    # test fixed burst window sim
    num_actions = 5
    num_channels = 8
    window_size = 200
    simulator = WindowSimulator(
        num_actions=num_actions,
        num_channels=num_channels,
        window_size=window_size,
    )

    batch_size = 32
    actions = F.one_hot(
        torch.randint(0, num_actions, size=(batch_size,)), 
        num_classes=simulator.num_actions
    ).float()
    windows = simulator(actions)

    assert list(windows.shape) == [batch_size, num_channels, window_size]

    # old version of simulator
    if simulator_type == 'sin':
        emg_sim = EMGSimulator(4, window_size=100)
        emgs = [emg_sim(np.array([1,0,1])) for _ in range(3)]
        emgs += [emg_sim(np.array([1,0,1,1])) for _ in range(3)]
        emgs += [emg_sim(np.array([5,2,0,1,1])) for _ in range(3)]
        emg_series = np.concatenate(emgs, axis=1)

        for emg_channel in emg_series:
            plt.plot(emg_channel)
        plt.show()

    # New version of simulator
    elif simulator_type == 'normal':
        emg_sim = NormalSimulator()
        sim_emg_window = emg_sim(np.array([1,0,0]))

        with open('../../datasets/emg_recording.pkl', 'rb') as f:
            data = pickle.load(f)
        real_emg  = data[0]["user_signals"]
        real_emg_window = real_emg[50,0,:]

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(sim_emg_window)
        axs[0].set_title('Simulated EMG')
        # axs[0].legend()

        axs[1].plot(real_emg_window)
        axs[1].set_title('Real EMG')

        plt.tight_layout()
        plt.show()