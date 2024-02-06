import random
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.distributions as torch_dist

from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows

from lift.datasets import get_mad_sample

# TODO
# - how to handle action transitions
# - simultaneous actions should be superposition of single ones


class FakeSimulator:
    def __init__(self, action_size=4, features_per_action=4, noise=0.1):
        self.noise = noise
        self.features_per_action = features_per_action
        self.weights = np.random.rand(1, (action_size*features_per_action))
        self.biases = np.random.rand(1, (action_size*features_per_action))

    def __call__(self, actions):
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        n_actions = actions.shape[0]
        rep_actions = actions.repeat(self.features_per_action, axis=1)
        rep_weights = self.weights.repeat(n_actions, axis=0)
        rep_biases = self.biases.repeat(n_actions, axis=0)

        assert rep_actions.shape == rep_weights.shape == rep_biases.shape, 'Shape mismatch'

        features = rep_actions * rep_weights + rep_biases
        return features


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
        self.recording_strength = 0.5 # scaling value that equals the recording

        self.bias_range_shape = [self.num_bursts, self.num_actions, self.num_channels]
        self.emg_range_shape = [self.num_bursts, self.num_actions, self.num_channels]
    
    def set_params(self, bias_range, emg_range):
        self.bias_range = bias_range
        self.emg_range = emg_range
    
    # what exactly does this transformation do? How would we use it when estimating limits and biases from data=
    # def transform_params(self, bias_range, emg_range):
    #     # do we need random limits / biases when we have a Uniform distribution?
    #     bias_range = 0.1 * torch.relu(bias_range + torch.randn_like(bias_range) * self.noise) + 1e-5
    #     emg_range = F.relu(emg_range + torch.randn_like(emg_range) * self.noise) + 1e-5
    #     return bias_range, emg_range
    
    def __call__(self, actions):
        # batch_size = len(actions)
        actions[actions.abs() < .1] = 0

        # find scaling for each action to account for movement strength
        scaling = torch.zeros((actions.shape[0], actions.shape[1]*2))
        for action_idx, action in enumerate(actions):
            for i, value in enumerate(action):
                idx = i * 2 + (value < 0).type(torch.int)
                scaling[action_idx, idx] = value.abs()
        scaling = scaling / self.recording_strength

        biases, limits = self.bias_range, self.emg_range

        biases += (torch.randn_like(biases) * 2 - 1) * self.noise
        limits += (torch.randn_like(limits) * 2 - 1) * self.noise
        limits.clip_(min=1e-5)

        # map for active actions
        action_map = (scaling.abs() > 0).type(torch.float32)

        # get biases and limits for each action
        # both are superpositions of the single action biases and limits
        action_biases = torch.einsum('ijk, nj -> ink', biases, action_map)
        action_limits = torch.einsum('ijk, nj -> ink', limits, scaling)

        window_parts = [None] * self.num_bursts
        for i in range(self.num_bursts):
            window_parts[i] = (torch_dist.Uniform(-action_limits, action_limits).sample((200,)) + action_biases).permute(1, 2, 3, 0)
        window = torch.cat(window_parts, dim=-3).squeeze(0)
        # is it fine that values can be > 1 or < -1?
        return window

    def fit_params_to_mad_sample(self, data_path, desired_labels = None):
        emg_list, label_list = get_mad_sample(data_path, desired_labels = desired_labels)
        sort_id = np.argsort(label_list)
        emg_list = [emg_list[i] for i in sort_id]

        min_len = min([len(emg) for emg in emg_list])
        short_emgs = [emg[:min_len,:] for emg in emg_list]
        windows_list = [get_windows(s_emg, 200, 200, as_tensor=True) for s_emg in short_emgs]
        windows = torch.stack(windows_list, dim=0)

        mean_windows = windows.mean(dim=1)
        emg_bias = mean_windows.mean(dim=-1)
        window_wo_mean = mean_windows - emg_bias[:,:,None]
        emg_limits = window_wo_mean.abs().max(dim=-1)[0]

        self.set_params(emg_bias.unsqueeze(0), emg_limits.unsqueeze(0))


if __name__ == '__main__':
    
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
    sim_emg_window = simulator(actions)

    assert list(sim_emg_window.shape) == [batch_size, num_channels, window_size]

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