import os
import re
import subprocess
import time

import numpy as np
import torch
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
    [-1, 0, 0], # Ulnar Deviation
    [0, 1, 0], # Wrist Flexion
    [0, -1, 0], # Wrist Extension
    [0, 0, 1], # Hand Close
    [0, 0, -1], # Hand Open
])

def maybe_download_mad_dataset(mad_base_dir):
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

    # filter labels
    is_desired_label = [l in desired_labels for l in labels]
    filenames = [n for i, n in enumerate(filenames) if is_desired_label[i] == True]
    labels = [l for i, l in enumerate(labels) if is_desired_label[i] == True]

    # sort by labels
    idx_sort = np.argsort(labels)
    filenames = [filenames[i] for i in idx_sort]
    labels = [labels[i] for i in idx_sort]

    emg = [None for _ in range(len(filenames))]
    labels = [None for _ in range(len(filenames))]
    for i, filename in enumerate(filenames):
        labels[i] = int(re.findall(r'\d+', filename)[0])
        data = np.fromfile(os.path.join(trial_path, filename), dtype=np.int16).astype(np.float32)
        data = data.reshape(-1, num_channels)
        emg[i] = np.interp(data, emg_range, (-1, 1))
    
    return emg, labels

"""TODO: not sure what this is supposed to do"""
def get_mad_windows(data_path, window_size, window_increment, emg_min = -128, emg_max = 127, desired_labels = None, return_lists=False):
    emg_list, label_list = load_mad_person_trial(data_path, emg_min, emg_max, desired_labels)

    sort_id = np.argsort(label_list)
    label_list = [label_list[i] for i in sort_id]
    emg_list = [emg_list[i] for i in sort_id]
    # I used labels 0 - 4 (including 4), where label 0 is rest

    min_len = min([len(emg) for emg in emg_list])
    short_emgs = [emg[:min_len,:] for emg in emg_list]
    windows_list = [torch.from_numpy(get_windows(s_emg, window_size, window_increment)) for s_emg in short_emgs]
    windows = torch.stack(windows_list, dim=0)
    flat_windows = windows.flatten(start_dim=0, end_dim=1)

    n_repeats = windows_list[0].shape[0]
    short_labels = torch.tensor([np.repeat(label, repeats=n_repeats) for label in label_list])
    labels = short_labels.flatten(start_dim=0, end_dim=1)

    # actions = F.one_hot(short_labels, num_classes=5).float()
    # flat_actions = actions.flatten(start_dim=0, end_dim=1)
    if return_lists:
        return flat_windows, labels, windows_list, label_list

    return flat_windows, labels

# def get_raw_mad_dataset(eval_path, window_size, overlap, skip_person=None):
#     person_folders = [p for p in os.listdir(eval_path) if p != ".DS_Store"]
#     first_folder = person_folders[0]

#     keys = next(os.walk((eval_path + first_folder)))[1]

#     number_of_classes = 7
#     size_non_overlap = window_size - overlap

#     raw_dataset_dict = {}
#     for key in keys:

#         raw_dataset = {
#             'examples': [],
#             'labels': [],
#         }

#         for person_dir in person_folders:
#             # skip loading data for a certain person
#             if skip_person is not None and person_dir == skip_person:
#                 print(f'skipping {skip_person}')
#                 continue
#             examples = []
#             labels = []
#             data_path = eval_path + person_dir + '/' + key
#             for data_file in os.listdir(data_path):
#                 if data_file.endswith(".dat"):
#                     data_read_from_file = np.fromfile((data_path + '/' + data_file), dtype=np.int16)
#                     data_read_from_file = np.array(data_read_from_file, dtype=np.float32)

#                     dataset_example_formatted = []
#                     example = None
#                     emg_vector = []
#                     for value in data_read_from_file:
#                         emg_vector.append(value)
#                         if (len(emg_vector) >= 8):
#                             if example is None:
#                                 example = emg_vector
#                             else:
#                                 example = np.row_stack((example, emg_vector))
#                             emg_vector = []
#                             if (len(example) >= window_size):
#                                 example = example.transpose()
#                                 dataset_example_formatted.append(example)
#                                 example = example.transpose()
#                                 example = example[size_non_overlap:]

#                     dataset_example_formatted = np.array(dataset_example_formatted)
#                     examples.append(dataset_example_formatted)
#                     data_file_index = int(data_file.split('classe_')[1][:-4])
#                     label = data_file_index % number_of_classes + np.zeros(dataset_example_formatted.shape[0])
#                     labels.append(label)
            
#             raw_dataset['examples'].append(np.concatenate(examples))
#             raw_dataset['labels'].append(np.concatenate(labels))

#         raw_dataset_dict[key] = raw_dataset

#     return raw_dataset_dict

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

    actions = torch.zeros(len(labels), 3)
    # labels == 0 is Rest
    actions[labels == 1, 0] = 1
    actions[labels == 2, 0] = -1
    actions[labels == 3, 1] = 1
    actions[labels == 4, 1] = -1
    actions[labels == 5, 2] = 1
    actions[labels == 6, 2] = -1

    actions *= recording_strength
    return actions

def mad_augmentation(
    emg: list[torch.Tensor], 
    actions: list[torch.Tensor], 
    num_augmentation: int, 
    augmentation_distribution: str = 'uniform',
):
    """Discrete emg data augmentation using random interpolation

    Args:
        emg (list[torch.Tensor]): list of emg windows for each dof activation
        actions (list[torch.Tensor]): list of dof activations for each discrete action
        num_augmentation (int): number of augmented samples to generate
        augmentation_distribution (str): distribution to sample from. choices = ["uniform", "normal"]

    Returns:
        sample_emg (torch.tensor): sampled emgs. size=[num_augmentation, num_channels, window_size]
        sample_actions (torch.tensor): sampled actions. size=[num_augmentation, act_dim]
    """
    idx_baseline = [i for i in range(len(actions)) if torch.all(actions[i] == 0)][0]
    idx_pos = [i for i in range(len(actions)) if torch.any(actions[i] > 0)]
    idx_neg = [i for i in range(len(actions)) if torch.any(actions[i] < 0)]
    
    # truncate to the smallest sample size
    min_samples = min([el.shape[0] for el in emg])
    emg = [el[:min_samples] for el in emg]

    emg_baseline = emg[idx_baseline]
    action_baseline = actions[idx_baseline]
    emg_pos = torch.stack([emg[i] for i in idx_pos])
    action_pos = torch.stack([actions[i] for i in idx_pos])
    emg_neg = torch.stack([emg[i] for i in idx_neg])
    action_neg = torch.stack([actions[i] for i in idx_neg])
    
    act_dim = action_baseline.shape[-1]

    if augmentation_distribution == 'uniform':
        sample_actions = torch.rand(num_augmentation, act_dim) * 2 - 1
    elif augmentation_distribution == 'normal':
        sample_actions = torch.normal(0, .5, (num_augmentation, act_dim)).clip(-1, 1)
    else:
        raise ValueError("augmentation_distribution must be 'uniform' or 'normal'")
    
    # init emg samples with baseline
    idx_sample_baseline = torch.randint(len(emg_baseline), (num_augmentation,))
    sample_baseline = emg_baseline[idx_sample_baseline]
    
    # interpolate emg as: (abs - baseline) * act
    sample_emg = torch.zeros(*[num_augmentation] + list(emg_baseline.shape)[1:])
    for i in range(act_dim):
        idx_sample_pos = torch.randint(len(emg_baseline), (num_augmentation,))
        idx_sample_neg = torch.randint(len(emg_baseline), (num_augmentation,))
        pos_component = (emg_pos[i][idx_sample_pos] - sample_baseline) / action_pos[i][i]
        neg_component = emg_neg[i][idx_sample_neg] - sample_baseline / action_neg[i][i].abs()
        
        abs_action = sample_actions[:, i].abs().view(-1, 1, 1)
        is_pos = 1 * (sample_actions[:, i] > 0).view(-1, 1, 1)
        sample_emg += (
            is_pos * abs_action * pos_component + \
            (1 - is_pos) * abs_action * neg_component # + \
            # sample_baseline
        )

    sample_emg = sample_emg + sample_baseline
    
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
    assert labels == desired_labels
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

    # test augmentation
    window_list, label_list = mad_groupby_labels(mad_windows, mad_labels)
    actions_list = mad_labels_to_actions(
        label_list, recording_strength=config.simulator.recording_strength,
    )
    sample_emg, sample_actions = mad_augmentation(
        window_list, 
        actions_list, 
        config.pretrain.num_augmentation,
        augmentation_distribution=config.pretrain.augmentation_distribution
    )
    assert list(sample_emg.shape) == [config.pretrain.num_augmentation, config.n_channels, config.window_size]
    assert list(sample_actions.shape) == [config.pretrain.num_augmentation, 3]
    assert torch.all(sample_actions.abs() <= 1.)