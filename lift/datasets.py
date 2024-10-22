import os
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from libemg.feature_extractor import FeatureExtractor
from sklearn.preprocessing import LabelEncoder

from lift.environments.interpolation import WeightedInterpolator


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
        return len(self.data[self.data_keys[0]])
    
    def __getitem__(self, idx):
        out = {k: v[idx] for k, v in self.data.items()}
        return out


class EMGSeqDataset(Dataset):
    """Sequence learning dataset"""
    def __init__(self, data: list[dict]):
        assert isinstance(data[0], dict), "data elements must be dictionaries"
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # concat obs, emg_ob, emg_act
        data = self.data[idx]
        obs = np.concatenate(list(data["obs"].values()), axis=-1)
        act = data["act"]
        out = np.concatenate([obs, act], axis=-1)
        out = torch.from_numpy(out).to(torch.float32)
        return out


def get_samples_per_group(windows, labels, num_samples_per_group=1):
    mad_windows_group, mad_labels_group = mad_groupby_labels(windows, labels)
    sample_idx = [torch.randint(0, len(g), size=(num_samples_per_group,)) for g in mad_windows_group]
    mad_windows_group = [g[sample_idx[i]] for i, g in enumerate(mad_windows_group)]
    mad_labels_group = [l * torch.ones_like(sample_idx[l]) for l in mad_labels_group]

    new_windows = torch.cat(mad_windows_group, dim=0)
    new_labels = torch.cat(mad_labels_group, dim=0)
    return new_windows, new_labels


def format_seq_data(data: dict) -> list[dict]:
    """Format rollout data into sequences"""
    obs_keys = data["obs"].keys()
    eps_ids = data["done"].cumsum()
    eps_ids = np.insert(eps_ids, 0, 0)[:-1]
    unique_eps_ids = np.unique(eps_ids)
    
    seq_data = []
    for eps_id in unique_eps_ids:
        idx = eps_ids == eps_id
        obs = {k: data["obs"][k][idx] for k in obs_keys}
        act = data["act"][idx]
        rwd = data["rwd"][idx]
        next_obs = {k: data["next_obs"][k][idx] for k in obs_keys}
        done = data["done"][idx]
        seq_data.append({
            "obs": obs,
            "act": act,
            "rwd": rwd,
            "next_obs": next_obs,
            "done": done
        })
    return seq_data

def seq_collate_fn(batch):
    """Pad sequence return mask"""
    pad_batch = pad_sequence(batch)
    mask = pad_sequence([torch.ones(len(b)) for b in batch])
    return pad_batch, mask

"""TODO: figure out better way to handle bc and sequence data"""
def get_dataloaders(data_dict, is_seq=False, train_ratio=0.8, batch_size=32, num_workers=4):

    if "val" in data_dict.keys():
        train_dataset = EMGSLDataset(data_dict["train"])
        val_dataset = EMGSLDataset(data_dict["val"])
        collate_fn = None
    else:
        if not is_seq:
            dataset = EMGSLDataset(data_dict)
            collate_fn = None
        else:
            seq_data = format_seq_data(data_dict)
            dataset = EMGSeqDataset(seq_data)
            collate_fn = seq_collate_fn

        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        num_workers=num_workers, 
        persistent_workers=True, 
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
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
    verbose: bool = True,
    cutoff_n_outer_samples: int = 0
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
        trial_dataset = {'emg': [], 'labels': [], 'person_id': []}
        for person_dir in person_folders:
            # skip loading data for a certain person
            if skip_person is not None and person_dir in skip_person:
                if verbose:
                    logging.info(f'Skipping {skip_person}')
                continue
            
            trial_path = data_path + person_dir + '/' + trial
            emg, labels = load_mad_person_trial(
                trial_path, 
                num_channels=num_channels, 
                emg_range=emg_range, 
                desired_labels=desired_labels,
            )
            # take away n samples
            if cutoff_n_outer_samples > 0:
                emg = [e[cutoff_n_outer_samples:-cutoff_n_outer_samples,:] for e in emg]
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
            trial_dataset['person_id'].append([f"{trial}_{person_dir}"] * len(trial_dataset["labels"][-1]))
        
        if len(trial_dataset['emg']) > 0:
            trial_dataset['emg'] = np.concatenate(trial_dataset['emg'])
            trial_dataset['labels'] = np.concatenate(trial_dataset['labels']).astype(int)
            trial_dataset['person_id'] = sum(trial_dataset['person_id'], [])
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
    verbose: bool = True,
    cutoff_n_outer_samples: int = 0,
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
        verbose=verbose,
        cutoff_n_outer_samples=cutoff_n_outer_samples,
    )
    eval_dataset = load_mad_dataset(
        eval_path, 
        num_channels=num_channels,
        emg_range=emg_range,
        window_size=window_size,
        window_overlap=window_overlap,
        desired_labels=desired_labels,
        skip_person=skip_person,
        verbose=verbose,
        cutoff_n_outer_samples=cutoff_n_outer_samples,
    )

    if len(train_dataset['training0']['emg']) < 1:
        mad_windows = np.concatenate([
            eval_dataset['training0']['emg'],
            eval_dataset['Test0']['emg'],
            eval_dataset['Test1']['emg'],
        ])
        mad_labels = np.concatenate([
            eval_dataset['training0']['labels'],
            eval_dataset['Test0']['labels'],
            eval_dataset['Test1']['labels'],
        ])
        mad_person_id = sum(
            [
                eval_dataset['training0']['person_id'],
                eval_dataset['Test0']['person_id'],
                eval_dataset['Test0']['person_id'],
            ], []
        )
    elif len(eval_dataset['training0']['emg']) < 1:
        mad_windows = np.concatenate([
            train_dataset['training0']['emg'],
        ])
        mad_labels = np.concatenate([
            train_dataset['training0']['labels'],
        ])
        mad_person_id = train_dataset['training0']['person_id']
    else:
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
        mad_person_id = sum(
            [
                train_dataset['training0']['person_id'],
                eval_dataset['training0']['person_id'],
                eval_dataset['Test0']['person_id'],
                eval_dataset['Test0']['person_id'],
            ], []
        )
    mad_person_id = LabelEncoder().fit_transform(mad_person_id)

    if verbose:
        logging.info("MAD dataset loaded")
    if return_tensors:
        mad_windows = torch.tensor(mad_windows, dtype=torch.float32)
        mad_labels = torch.tensor(mad_labels)
        mad_person_id = torch.tensor(mad_person_id)
    return mad_windows, mad_labels, mad_person_id

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

def weighted_augmentation(mad_windows, mad_actions, config):
    mad_features = compute_features(mad_windows, feature_list = ['MAV'])
    interpolator = WeightedInterpolator(mad_features, mad_actions,
                                        k=config.simulator.k,
                                        sample=config.simulator.sample)

    if config.pretrain.augmentation_distribution == 'uniform':
        sample_actions = torch.rand(config.pretrain.num_augmentation,
                                    config.action_size) * 2 - 1
    elif config.pretrain.augmentation_distribution == 'normal':
        sample_actions = torch.normal(0, .5, (config.pretrain.num_augmentation,
                                                config.action_size)).clip(-1, 1)

    sample_features = interpolator(sample_actions)
    return sample_features, sample_actions

def weighted_per_person_augmentation(config):
    mad_features = None
    mad_actions = None
    sample_features = None
    sample_actions = None

    people_list = [f"Female{i}" for i in range(10)] + [f"Male{i}" for i in range(16)]
    people_list = [p for p in people_list if not p == config.target_person]

    aug_per_person = config.pretrain.num_augmentation // len(people_list)

    for p in people_list:
        other_list = [o_p for o_p in people_list if not o_p == p]
        person_windows, person_labels = load_all_mad_datasets(
            config.mad_base_path.as_posix(),
            num_channels=config.n_channels,
            emg_range=config.emg_range,
            window_size=config.window_size,
            window_overlap=config.window_overlap,
            desired_labels=config.desired_mad_labels,
            skip_person=other_list,
            return_tensors=True,
            verbose=False,
        )

        person_features = compute_features(person_windows, feature_list = ['MAV'])
        person_actions = mad_labels_to_actions(
                person_labels, recording_strength=config.simulator.recording_strength,
            )

        p_interpolator = WeightedInterpolator(person_features, person_actions,
                                              k=config.simulator.k)

        if config.pretrain.augmentation_distribution == 'uniform':
            p_sample_action = torch.rand(aug_per_person,
                                        config.action_size) * 2 - 1
        elif config.pretrain.augmentation_distribution == 'normal':
            p_sample_action = torch.normal(0, .5, (aug_per_person,
                                                  config.action_size)).clip(-1, 1)

        p_sample_features = p_interpolator(p_sample_action)

        if mad_features is None:
            mad_features = person_features
            mad_actions = person_actions
            sample_features = p_sample_features
            sample_actions = p_sample_action
        else:
            mad_features = torch.cat((mad_features, person_features), dim=0)
            mad_actions = torch.cat((mad_actions, person_actions), dim=0)
            sample_features = torch.cat((sample_features, p_sample_features), dim=0)
            sample_actions = torch.cat((sample_actions, p_sample_action), dim=0)
        return sample_features, sample_actions, mad_features, mad_actions

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
    # window_list, label_list = mad_groupby_labels(mad_windows, mad_labels)
    # actions_list = mad_labels_to_actions(
    #     label_list, recording_strength=config.simulator.recording_strength,
    # )
    # sample_windows, sample_actions = mad_augmentation(
    #     window_list, 
    #     actions_list, 
    #     config.pretrain.num_augmentation,
    #     augmentation_distribution=config.pretrain.augmentation_distribution
    # )
    # assert list(sample_windows.shape) == [config.pretrain.num_augmentation, config.n_channels, config.window_size]
    # assert list(sample_actions.shape) == [config.pretrain.num_augmentation, 3]
    # assert torch.all(sample_actions.abs() <= 1.)
    # assert torch.all(sample_windows.abs() <= 1.)