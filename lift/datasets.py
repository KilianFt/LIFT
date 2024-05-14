import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import Dataset, DataLoader, random_split
from libemg.utils import get_windows
from libemg.feature_extractor import FeatureExtractor

class EMGSLDataset(Dataset):
    """Supervised learning dataset"""
    def __init__(self, data_dict):
        assert isinstance(data_dict, dict), "data must be a dictionary"
        self.data_keys = list(data_dict.keys())
        self.data = {
            k: v if isinstance(v, torch.Tensor) 
            else torch.from_numpy(v).to(torch.float32) 
            for k, v in data_dict.items()
        }

    def __len__(self):
        return len(self.data[self.data_keys[0]])
    
    def __getitem__(self, idx):
        out = {k: v[idx] for k, v in self.data.items()}
        return out


def get_dataloaders(data_dict, train_ratio=0.8, batch_size=32, num_workers=4):
    dataset = EMGSLDataset(data_dict)

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        persistent_workers=True, 
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        persistent_workers=True,
    )
    return train_dataloader, val_dataloader

def make_overlap_windows(emg: np.array, window_size: int = 200, window_overlap: int = 150):
    """Make overlapping windows from streaming signals

    Returns:
        windows (np.array): emg windows. size = [num_windows, num_channels, window_size]
    """
    append_len = window_size - window_overlap
    windows = [emg[:window_size]]
    num_windows = (len(emg) - window_size) // append_len
    for i in range(num_windows):
        start = (i + 1) * append_len
        windows.append(emg[start:start+window_size])
    windows = np.stack(windows)
    windows = np.moveaxis(windows, -2, -1)
    return windows

def compute_features(
    windows: np.ndarray | torch.Tensor, 
    feature_list: list = ['MAV', 'SSC', 'ZC', 'WL'],
):
    """Compute emg features using libemg and flatten
    
    Returns:
        features (torch.tensor): emg features. size = [num_samples, num_channels * num_features]
    """
    if not isinstance(windows, np.ndarray):
        windows = windows.numpy()

    features = FeatureExtractor().extract_features(feature_list, windows)
    features = np.stack(list(features.values()), axis=-1)
    features = torch.from_numpy(features).flatten(start_dim=1).to(torch.float32)
    return features

"""
mad dataset
"""
MAD_LABELS_TO_DOF = np.array([
    [0, 0, 0], # Neutral
    [1, 0, 0], # Radial Deviation
    [0, 1, 0], # Wrist Flexion
    [-1, 0, 0], # Ulnar Deviation
    [0, -1, 0], # Wrist Extension
    [0, 0, 1], # Hand Close
    [0, 0, -1], # Hand Open
])

def maybe_download_mad_dataset(mad_base_dir):
    if isinstance(mad_base_dir, Path):
        mad_base_dir = mad_base_dir.as_posix()

    if os.path.exists(mad_base_dir):
        return
    print("MyoArmbandDataset not found")

    if os.path.exists(mad_base_dir + '/.lock'):
        print("Waiting for download to finish")
        # wait for download to finish
        while os.path.exists(mad_base_dir + '/.lock'):
            print(".", end="")
            time.sleep(1)
        return

    # create a lock file to prevent multiple downloads
    os.system(f'touch {mad_base_dir}/.lock')

    print("Downloading MyoArmbandDataset")
    cmd = f'git clone https://github.com/UlysseCoteAllard/MyoArmbandDataset {mad_base_dir}'
    subprocess.call(cmd, shell=True)
    print("Download finished")

    # remove the lock file
    os.system(f'rm {mad_base_dir}/.lock')

def load_mad_person_trial(
    trial_path: str, 
    num_channels: int = 8, 
    emg_range: list = [-128., 127.], 
    desired_labels: list = None,
):
    """Load from mad dataset one person one trial, i.e., all classe_*.dat files in training0
    
    Returns:
        emg (list[np.array]): list of np arries of emg signals where each element corresponds to a class label. 
            The emg signals are normalized to range [-1, 1] based on argument emg_range.
            The size of each array is [seq_len, num_channels].
        labels (list): list of class labels. The length of the list is equal to the length of emg.
            The class labels and corresponding emg is sorted.
    """
    filenames = [n for n in os.listdir(trial_path) if n.endswith(".dat")]
    labels = [int(n.split("_")[1].replace(".dat", "")) for n in filenames]

    # labels cycle every 7 classes, thus class 0 is same as 7 which is same as 14
    labels = [l % 7 for l in labels]

    # filter labels
    if desired_labels is not None:
        assert min(desired_labels) >= 0 and max(desired_labels) <= 6, "desired_labels should be in range [0, 6]"
        is_desired_label = [l in desired_labels for l in labels]
        filenames = [n for i, n in enumerate(filenames) if is_desired_label[i] == True]
        labels = [l for i, l in enumerate(labels) if is_desired_label[i] == True]

    # sort by labels
    unique_labels = np.sort(np.unique(labels))
    emg = [None for _ in range(len(unique_labels))]

    for i, u_label in enumerate(unique_labels):
        filenames_label = [f for f, l in zip(filenames, labels) if l == u_label]

        for filename in filenames_label:
            data = np.fromfile(os.path.join(trial_path, filename), dtype=np.int16).astype(np.float32)
            data = data.reshape(-1, num_channels)
            data = np.interp(data, emg_range, (-1, 1))
            if emg[i] is None:
                emg[i] = data
            else:
                emg[i] = np.concatenate([emg[i], data], axis=0)
    
    return emg, unique_labels

def load_mad_dataset(
    data_path: str, 
    num_channels: int = 8,
    emg_range: list = [-128, 127],
    window_size: int = 200, 
    window_overlap: int = 50, 
    desired_labels: list = None,
    skip_person: list = None,
):
    """Load all person trials in mad dataset in either pretrain or eval path
    
    Returns:
        dataset (dict[dict[np.array]]): dictionary of pairs of emg and labels np arraies. 
            The first level keys are mad data trial names. The second level keys are ["emg", "labels"].
    """
    person_folders = [p for p in os.listdir(data_path) if p != ".DS_Store"]
    first_folder = person_folders[0]
    trial_names = next(os.walk((data_path + first_folder)))[1]

    dataset = {}
    for trial in trial_names:
        trial_dataset = {'emg': [], 'labels': []}
        for person_dir in person_folders:
            # skip loading data for a certain person
            if skip_person is not None and person_dir == skip_person:
                print(f'skipping {skip_person}')
                continue
            
            trial_path = data_path + person_dir + '/' + trial
            emg, labels = load_mad_person_trial(
                trial_path, 
                num_channels=num_channels, 
                emg_range=emg_range, 
                desired_labels=desired_labels,
            )
            windows = [
                make_overlap_windows(
                    e, 
                    window_size=window_size, 
                    window_overlap=window_overlap,
                ) for e in emg
            ]
            labels = [np.ones(len(w)) * labels[i] for i, w in enumerate(windows)]
            
            trial_dataset['emg'].append(np.concatenate(windows))
            trial_dataset['labels'].append(np.concatenate(labels))
        
        trial_dataset['emg'] = np.concatenate(trial_dataset['emg'])
        trial_dataset['labels'] = np.concatenate(trial_dataset['labels']).astype(int)
        dataset[trial] = trial_dataset
    return dataset

def load_all_mad_datasets(
    mad_base_dir: str, 
    num_channels: int = 8,
    emg_range: list = [-128, 127],
    window_size: int = 200, 
    window_overlap: int = 50, 
    desired_labels: list = None,
    skip_person: list = None,
    return_tensors: bool = False, 
):
    """Load all mad pretain and eval data
    
    Returns:
        mad_windows (np.array | torch.tensor): concatenated windows. size = [num_samples, num_channels, window_size]
        mad_labels (np.array | torch.tensor): concatenated labels. size = [num_samples]
    """
    maybe_download_mad_dataset(mad_base_dir)
    train_path = mad_base_dir + '/PreTrainingDataset/'
    eval_path = mad_base_dir + '/EvaluationDataset/'

    train_dataset = load_mad_dataset(
        train_path,        
        num_channels=num_channels,
        emg_range=emg_range,
        window_size=window_size,
        window_overlap=window_overlap,
        desired_labels=desired_labels,
        skip_person=skip_person,
    )
    eval_dataset = load_mad_dataset(
        eval_path, 
        num_channels=num_channels,
        emg_range=emg_range,
        window_size=window_size,
        window_overlap=window_overlap,
        desired_labels=desired_labels,
        skip_person=skip_person,
    )

    mad_windows = np.concatenate([
        train_dataset['training0']['emg'],
        eval_dataset['training0']['emg'],
        eval_dataset['Test0']['emg'],
        eval_dataset['Test1']['emg'],
    ])

    mad_labels = np.concatenate([
        train_dataset['training0']['labels'],
        eval_dataset['training0']['labels'],
        eval_dataset['Test0']['labels'],
        eval_dataset['Test1']['labels'],
    ])

    print("MAD dataset loaded")
    if return_tensors:
        mad_windows = torch.tensor(mad_windows, dtype=torch.float32)
        mad_labels = torch.tensor(mad_labels)
    return mad_windows, mad_labels

def mad_groupby_labels(emg: np.array, labels: np.array):
    """Group emg into list according to labels
    
    emg_list (list[np.array]): list of emg windows or features grouped by labels.
    label_list (list[int]): list of labels. 
    """
    emg_list = []
    label_list = []
    for u_label in np.sort(np.unique(labels)):
        label_idxs = np.where(labels == u_label)[0]
        label_emg = emg[label_idxs]
        emg_list.append(label_emg)
        label_list.append(u_label)
    return emg_list, label_list

def mad_labels_to_actions(labels: list, recording_strength: float = 1.0):
    """Convert labels (0 to 6) to actions"""
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    if labels.max() > 6 or labels.min() < 0:
        raise ValueError("Labels should be in range [0, 6]")
    
    assert labels.max() <= 6 and labels.min() >= 0, "Labels should be in range [0, 6]"

    actions = np.stack([MAD_LABELS_TO_DOF[label] for label in labels], axis=0)
    actions = torch.tensor(actions, dtype=torch.float32)

    actions *= recording_strength
    return actions


def interpolate_emg(base_emg, base_actions, actions, reduction='abs'):
    # interpolate emg as: (abs - baseline) * act
    # init emg samples with baseline
    num_augmentation, act_dim = actions.shape

    emg_baseline = base_emg['baseline']
    idx_sample_baseline = torch.randint(len(emg_baseline), (num_augmentation,))
    sample_baseline = emg_baseline[idx_sample_baseline]
    sample_emg = torch.zeros(*[num_augmentation] + list(emg_baseline.shape)[1:])
    for i in range(act_dim):
        idx_sample_pos = torch.randint(len(emg_baseline), (num_augmentation,))
        idx_sample_neg = torch.randint(len(emg_baseline), (num_augmentation,))
        pos_component = base_emg['pos'][i][idx_sample_pos] / base_actions['pos'][i][i].abs()
        neg_component = base_emg['neg'][i][idx_sample_neg] / base_actions['neg'][i][i].abs()
        
        abs_action = actions[:, i].abs().view(-1, 1, 1)
        is_pos = 1 * (actions[:, i] > 0).view(-1, 1, 1)
        sample_emg += (
            is_pos * abs_action * pos_component + \
            (1 - is_pos) * abs_action * neg_component
        )

    if reduction == 'mean':
        sample_emg = sample_emg / act_dim + sample_baseline
    elif reduction == 'abs':
        sample_emg = sample_emg / actions.abs().sum(dim=-1).clip(min=1.0)[:, None, None] + sample_baseline
    else:
        raise ValueError("reduction must be 'mean' or 'abs'")

    sample_emg = torch.clip(sample_emg, -1, 1)
    return sample_emg


def mad_augmentation(
    emg: list[torch.Tensor], 
    actions: list[torch.Tensor], 
    num_augmentation: int, 
    augmentation_distribution: str = 'uniform',
    reduction: str = 'act_dim',
):
    """Discrete emg data augmentation using random interpolation

    Args:
        emg (list[torch.Tensor]): list of emg windows for each dof activation
        actions (list[torch.Tensor]): list of dof activations for each discrete action
        num_augmentation (int): number of augmented samples to generate
        augmentation_distribution (str): distribution to sample from. choices = ["uniform", "normal"]

    Returns:
        sample_emg (torch.tensor): sampled emgs windows. size=[num_augmentation, num_channels, window_size]
        sample_actions (torch.tensor): sampled actions. size=[num_augmentation, act_dim]
    """
    assert torch.all(actions == 0, dim=1).any(), "Baseline action (0, 0, 0) required for augmentation"
    idx_baseline = [i for i in range(len(actions)) if torch.all(actions[i] == 0)][0]
    idx_pos = [i for i in range(len(actions)) if torch.any(actions[i] > 0)]
    idx_neg = [i for i in range(len(actions)) if torch.any(actions[i] < 0)]
    
    # truncate to the smallest sample size
    min_samples = min([el.shape[0] for el in emg])
    emg = [el[:min_samples] for el in emg]

    emg_baseline = emg[idx_baseline]
    base_emg = TensorDict({
        'baseline': emg_baseline,
        'pos': torch.stack([emg[i] - emg_baseline for i in idx_pos]),
        'neg': torch.stack([emg[i] - emg_baseline for i in idx_neg]),
    })

    base_actions = TensorDict({
        'baseline': actions[idx_baseline],
        'pos': torch.stack([actions[i] for i in idx_pos]),
        'neg': torch.stack([actions[i] for i in idx_neg]),
    })

    act_dim = actions.shape[-1]

    if augmentation_distribution == 'uniform':
        sample_actions = torch.rand(num_augmentation, act_dim) * 2 - 1
    elif augmentation_distribution == 'normal':
        sample_actions = torch.normal(0, .5, (num_augmentation, act_dim)).clip(-1, 1)
    else:
        raise ValueError("augmentation_distribution must be 'uniform' or 'normal'")
    
    sample_emg = interpolate_emg(base_emg, base_actions, sample_actions, reduction)
    
    return sample_emg, sample_actions

if __name__ == "__main__":
    from configs import BaseConfig
    config = BaseConfig()

    mad_base_dir = config.mad_base_path.as_posix()
    desired_labels = [0, 1, 2, 3, 4, 5, 6]

    # test load single person trial
    emg, labels = load_mad_person_trial(
        (config.mad_data_path / "Female0"/ "training0").as_posix(),
        num_channels=config.n_channels,
        emg_range=config.emg_range,
        desired_labels=desired_labels,
    )
    assert len(emg) == len(desired_labels)
    assert all([l in desired_labels for l in labels])
    assert all([a.shape[-1] == config.n_channels for a in emg])
    assert all([np.all(np.abs(a) <= 1.) for a in emg])

    # test make windows
    windows = make_overlap_windows(
        emg[0],
        window_size=config.window_size,
        window_overlap=config.window_overlap,
    )
    assert list(windows.shape[1:]) == [config.n_channels, config.window_size]

    # test compute features
    features = compute_features(
        windows,
        feature_list=["MAV", "SSC", "ZC", "WL"],
    )
    assert list(features.shape) == [len(windows), config.n_channels * 4]

    # test load all pretrain
    train_dataset = load_mad_dataset(
        mad_base_dir + '/PreTrainingDataset/',
        num_channels=config.n_channels,
        emg_range=config.emg_range,
        window_size=config.window_size,
        window_overlap=config.window_overlap,
        desired_labels=desired_labels,
        skip_person='Female0',
    )
    assert list(train_dataset["training0"]["emg"].shape[1:]) == [config.n_channels, config.window_size]
    assert train_dataset["training0"]["emg"].shape[0] == train_dataset["training0"]["labels"].shape[0]
    
    # test load all data
    mad_windows, mad_labels = load_all_mad_datasets(
        mad_base_dir,
        num_channels=config.n_channels,
        emg_range=config.emg_range,
        window_size=config.window_size,
        window_overlap=config.window_overlap,
        desired_labels=desired_labels,
        skip_person='Female0',
        return_tensors=True,
    )
    assert list(mad_windows.shape[1:]) == [config.n_channels, config.window_size]
    assert mad_windows.shape[0] == mad_labels.shape[0]
    assert torch.all(mad_windows.abs() <= 1.)

    # test augmentation
    window_list, label_list = mad_groupby_labels(mad_windows, mad_labels)
    actions_list = mad_labels_to_actions(
        label_list, recording_strength=config.simulator.recording_strength,
    )
    sample_windows, sample_actions = mad_augmentation(
        window_list, 
        actions_list, 
        config.pretrain.num_augmentation,
        augmentation_distribution=config.pretrain.augmentation_distribution
    )
    assert list(sample_windows.shape) == [config.pretrain.num_augmentation, config.n_channels, config.window_size]
    assert list(sample_actions.shape) == [config.pretrain.num_augmentation, 3]
    assert torch.all(sample_actions.abs() <= 1.)
    assert torch.all(sample_windows.abs() <= 1.)