import logging

import torch

from lift.datasets import (
    get_samples_per_group,
    load_all_mad_datasets,
    compute_features,
    mad_labels_to_actions,
    )
from lift.environments.interpolation import WeightedInterpolator

logging.basicConfig(level=logging.INFO)


class SimulatorFactory:
    @staticmethod
    def create_class(data_path, config, num_samples_per_group=None):
        return Simulator(data_path, config, num_samples_per_group=num_samples_per_group)


class Simulator:
    """Non-parametric simulator that interpolates EMG samples from the MAD dataset.
    The interpolation is based on the weighted average of the closest k samples from different actions."""
    def __init__(self, data_path, config, num_samples_per_group=None) -> None:
        self.num_channels = config.num_channels
        self.window_size = config.window_size
        self.num_features = 1
        self.return_features = True # flag to show that the simulator returns features

        # load emg data of one person
        p = data_path.split('/')[-2]
        people_list = [f"Female{i}" for i in range(10)] + [f"Male{i}" for i in range(16)]
        other_list = [o_p for o_p in people_list if not o_p == p]
        person_windows, person_labels, _ = load_all_mad_datasets(
            config.mad_base_path.as_posix(),
            num_channels=config.num_channels,
            emg_range=config.emg_range,
            window_size=config.window_size,
            window_overlap=config.window_overlap,
            desired_labels=config.desired_mad_labels,
            skip_person=other_list,
            return_tensors=True,
            verbose=False,
            cutoff_n_outer_samples=config.cutoff_n_outer_samples,
        )
        if num_samples_per_group is not None:
            person_windows, person_labels = get_samples_per_group(person_windows, person_labels, num_samples_per_group)

        person_features = compute_features(person_windows, feature_list = ['MAV'])
        person_actions = mad_labels_to_actions(
            person_labels, recording_strength=config.simulator.recording_strength,
        )
        self.interpolator = WeightedInterpolator(person_features, person_actions,
                                                 k=config.simulator.k, sample=config.simulator.sample)

    def __call__(self, actions):
        features = self.interpolator(actions)
        return features


if __name__ == "__main__":
    from configs import BaseConfig
    from lift.datasets import MAD_LABELS_TO_DOF

    config = BaseConfig()
    data_path = (config.mad_data_path / "Female0" / "training0").as_posix()
    sim = Simulator(
        data_path,
        config,
    )

    single_actions = torch.from_numpy(MAD_LABELS_TO_DOF).to(torch.float32) * config.simulator.recording_strength
    out = sim(single_actions)

    assert out.min() >= -1 and out.max() <= 1, "Values should be in range [-1, 1]"

    # verify that generated emg is similar to the MAD dataset
    people_list = [f"Female{i}" for i in range(10)] + [f"Male{i}" for i in range(16)]
    p = data_path.split('/')[-2]
    other_list = [o_p for o_p in people_list if not o_p == p]
    person_windows, person_labels, _ = load_all_mad_datasets(
        config.mad_base_path.as_posix(),
        num_channels=config.num_channels,
        emg_range=config.emg_range,
        window_size=config.window_size,
        window_overlap=config.window_overlap,
        desired_labels=config.desired_mad_labels,
        skip_person=other_list,
        return_tensors=True,
        verbose=False,
        cutoff_n_outer_samples=config.cutoff_n_outer_samples,
    )
    actions = mad_labels_to_actions(
        person_labels, recording_strength=config.simulator.recording_strength,
    )
    features = compute_features(person_windows, feature_list=['MAV'])

    mad_sim_feats = sim(actions)

    # verify that it stays in limits [-1, 1]
    base_mse = torch.mean(torch.pow(mad_sim_feats[0] - features[0], 2))
    assert base_mse < 5e-3, "Baseline features should be close to MAD features"

    rand_actions = torch.rand((1000, 3)) * 2 - 1
    rand_feats = sim(rand_actions)

    assert rand_feats.min() >= -1 and rand_feats.max() <= 1, "Values should be in range [-1, 1]"
    logging.info('Simulator test passed')
