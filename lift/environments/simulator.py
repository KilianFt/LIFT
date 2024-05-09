import logging

import numpy as np
from libemg.utils import get_windows

import torch
import torch.distributions as torch_dist

from tensordict import TensorDict

from lift.datasets import load_mad_person_trial, compute_features
from lift.datasets import MAD_LABELS_TO_DOF


logging.basicConfig(level=logging.INFO)


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
    def __init__(
            self, 
            config,
            return_features=False,
        ):
        self.num_features = 4
        self.return_features = return_features
        self.action_size = config.action_size
        self.num_bursts = config.simulator.n_bursts
        self.num_channels = config.n_channels
        self.window_size = config.window_size
        self.burst_durations = [self.window_size // self.num_bursts] * (self.num_bursts - 1)
        self.burst_durations = self.burst_durations + [self.window_size - int(np.sum(self.burst_durations))]
        
        # init params
        self.biases = TensorDict({
            'baseline': torch.zeros(self.num_channels),
            'positive': torch.zeros(self.action_size, self.num_channels),
            'negative': torch.zeros(self.action_size, self.num_channels),
        })
        self.limits = TensorDict({
            'baseline': torch.ones(self.num_channels),
            'positive': torch.ones(self.action_size, self.num_channels),
            'negative': torch.ones(self.action_size, self.num_channels),
        })

        self.noise = config.simulator.noise
        self.base_noise = config.simulator.base_noise
        # scaling value that equals the recording (normal strength)
        self.recording_strength = config.simulator.recording_strength 

        feature_size = self.num_features * self.num_channels
        self.feature_means = torch.zeros(feature_size)
        self.feature_stds = torch.ones(feature_size)


    def fit_normalization_params(self):
        actions = torch.rand(10000, 3) * 2 - 1
        features = self(actions)
        self.feature_means = features.mean(dim=0)
        self.feature_stds = features.std(dim=0)


    def get_norm_features(self, emg_window):
        features = compute_features(emg_window)
        return (features - self.feature_means) / self.feature_stds

    
    """Map fetch actions to emg windows"""
    def __call__(self, actions, no_clip=False):
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32).clone()

        assert len(actions.shape) > 1, "Actions should be a 2D tensor"

        act_dim = actions.shape[-1]

        if act_dim > self.action_size:
            actions = actions[..., :self.action_size]

        # shape: [n_actions, window_size, num_channels]
        base_limits = self.limits['baseline']
        base_bias = self.biases['baseline']
        base_limits += (torch.randn_like(base_limits) * 2 - 1) * self.base_noise
        base_bias += (torch.randn_like(base_bias) * 2 - 1) * self.base_noise

        sample_baseline = torch_dist.Uniform(-base_limits,
                                             base_limits
                                             ).sample((actions.shape[0], self.window_size))
        sample_baseline += base_bias
        
        sample_emg = torch.zeros((actions.shape[0], self.window_size, self.num_channels))

        positive_actions = actions.clone() # [n_actions, act_dim]
        positive_actions[positive_actions < 0] = 0
        negative_actions = actions.clone() # [n_actions, act_dim]
        negative_actions[negative_actions > 0] = 0

        # limits: [n_actions, act_dim] * [act_dim, num_channels] -> [n_actions, num_channels]
        pos_sample_limits = positive_actions.abs() @ self.limits['positive'] # [n_actions, num_channels]
        neg_sample_limits = negative_actions.abs() @ self.limits['negative'] # [n_actions, num_channels]
        sample_limits = pos_sample_limits + neg_sample_limits

        assert (sample_limits >= 0).all(), "Limits should be positive"

        sample_limits += (torch.randn_like(sample_limits) * 2 - 1) * self.noise
        sample_limits.clip_(min=1e-6)

        # sample_emg shape [n_actions, window_size, num_channels]
        sample_emg = torch_dist.Uniform(-sample_limits,
                                         sample_limits
                                         ).sample((self.window_size,)).permute(1,0,2)
        
        # biases
        # do we scale this by actions?
        pos_sample_biases = positive_actions.abs() @ self.biases['positive']
        neg_sample_biases = negative_actions.abs() @ self.biases['negative']
        sample_biases = pos_sample_biases + neg_sample_biases

        sample_biases += (torch.randn_like(sample_biases) * 2 - 1) * self.noise

        sample_emg += sample_biases[:, None, :]

        sample_emg = sample_emg / actions.abs().sum(dim=-1)[:, None, None] + sample_baseline
        # sample_emg = sample_emg / act_dim  + sample_baseline
        # sample_emg = sample_emg + sample_baseline

        # shape [n_actions, num_channels, window_size]
        sample_emg = sample_emg.permute(0, 2, 1)

        # noise
        # sample_emg += (torch.randn_like(sample_emg) * 2 - 1) * self.noise
        if not no_clip:
            sample_emg = torch.clip(sample_emg, -1, 1)

        if self.return_features:
            return self.get_norm_features(sample_emg)
        
        return sample_emg


    """Calculate the upper limit of the uniform distribution for a given emg sample"""
    def _get_action_emg_limit(self, action_emg):
        # remove bias for limits
        action_emg -= action_emg.mean(dim=0)
        # get 95% quantile of abs values
        idx = min(int(0.95 * len(action_emg)), len(action_emg) - 1)
        limit = action_emg.abs().sort(dim=0)[0][idx]
        # scale by recording strength
        limit = limit / self.recording_strength
        return limit

    def _get_biases_and_limits(self, emg, actions):
        biases = self.biases.copy()
        limits = self.limits.copy()

        for s_idx, action in enumerate(actions):
            action_emg = torch.tensor(emg[s_idx], dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.float32)

            if (action == 0).all():
                biases['baseline'] = action_emg.mean(dim=0)
                limits['baseline'] = self._get_action_emg_limit(action_emg)
            else:
                idx = torch.argmax(torch.abs(action))
                key = 'negative' if action[idx] < 0 else 'positive'
                biases[key][idx, :] = action_emg.mean(dim=0)
                action_limit = self._get_action_emg_limit(action_emg) - limits['baseline']
                limits[key][idx, :] = action_limit.clip(min=0.)

        return biases, limits

    """TODO: 
    make simulator fitting and sampling consistent with mad augmentation
    maybe aggregate all mad samples and feed as input to this function in case we want to fit to multiple participants
    """
    def fit_params_to_mad_sample(self, data_path, emg_range: list = [-128, 127], desired_labels: list = [0, 1, 2, 3, 4, 5, 6]):
        emg, labels = load_mad_person_trial(
            data_path, 
            num_channels=self.num_channels,
            emg_range=emg_range,
            desired_labels=desired_labels,
        )
        actions = [MAD_LABELS_TO_DOF[l] for l in labels]
        self.biases, self.limits = self._get_biases_and_limits(emg, actions)
        logging.info('Fitted simulator to MAD sample')


if __name__ == "__main__":
    from configs import BaseConfig
    import matplotlib.pyplot as plt

    config = BaseConfig()
    sim = WindowSimulator(
        action_size=config.action_size, 
        num_bursts=config.simulator.n_bursts, 
        num_channels=config.n_channels,
        window_size=config.window_size, 
        recording_strength=config.simulator.recording_strength,
        return_features=False,
        noise=0.0,
    )
    sim.fit_params_to_mad_sample(
        config.mad_data_path / "Female0" / "training0"
    )
    single_actions = torch.from_numpy(MAD_LABELS_TO_DOF).to(torch.float32) * config.simulator.recording_strength
    out = sim(single_actions)

    assert out.min() >= -1 and out.max() <= 1, "Values should be in range [-1, 1]"

    # verify that generated emg is similar to the MAD dataset
    emg, labels = load_mad_person_trial(
            config.mad_data_path / "Female0" / "training0", 
            num_channels=config.n_channels,
        )
    actions = [MAD_LABELS_TO_DOF[l] for l in labels]

    emg_means, emg_maxs = [], []
    for action_emg in emg:
        emg_means.append(torch.tensor(action_emg, dtype=torch.float32).mean(dim=0))
        emg_maxs.append(torch.tensor(action_emg, dtype=torch.float32).abs().max(dim=0)[0])

    emg_means = torch.stack(emg_means, dim=0)
    emg_maxs = torch.stack(emg_maxs, dim=0)

    # test biases
    pos_idxs = [1, 2, 5]
    neg_idxs = [3, 4, 6]

    sim_mean = out.mean(dim=-1)

    base_mse = torch.mean(torch.pow(sim_mean[0] - emg_means[0], 2))
    pos_mse = torch.mean(torch.pow(sim_mean[pos_idxs] - emg_means[pos_idxs], 2))
    neg_mse = torch.mean(torch.pow(sim_mean[neg_idxs] - emg_means[neg_idxs], 2))

    logging.info('--- Biases ---')
    logging.info(f'Baseline MSE: {base_mse}')
    logging.info(f'Positive MSE: {pos_mse}')
    logging.info(f'Negative MSE: {neg_mse}')

    assert base_mse < 5e-3, "Baseline mean should be close to MAD mean"
    assert pos_mse < 5e-3, "Positive mean should be close to MAD mean"
    assert neg_mse < 5e-3, "Negative mean should be close to MAD mean"

    # test limits
    sim_max = out.abs().max(dim=-1)[0]

    base_max_mse = torch.mean(torch.pow(sim_max[0] - emg_maxs[0], 2))
    pos_max_mse = torch.mean(torch.pow(sim_max[pos_idxs] - emg_maxs[pos_idxs], 2))
    neg_max_mse = torch.mean(torch.pow(sim_max[neg_idxs] - emg_maxs[neg_idxs], 2))

    logging.info('--- Limits ---')
    logging.info(f'Baseline MSE: {base_max_mse}')
    logging.info(f'Positive MSE: {pos_max_mse}')
    logging.info(f'Negative MSE: {neg_max_mse}')

    assert base_max_mse < 5e-3, "Baseline mean should be close to MAD mean"
    assert pos_max_mse < 5e-3, "Positive mean should be close to MAD mean"
    assert neg_max_mse < 5e-3, "Negative mean should be close to MAD mean"

    # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # ax.plot(out[:, 0].T)
    # plt.show()
    logging.info('Simulator test passed')
