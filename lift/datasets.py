import os
import re

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

def get_mad_sample(data_path, emg_min = -128, emg_max = 127, desired_labels = None):
    emg = []
    labels = []
    for data_file in os.listdir(data_path):
        if data_file.endswith(".dat"):
            label = int(re.findall(r'\d+', data_file)[0])
            data_read_from_file = np.fromfile((data_path + '/' + data_file), dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float32)

            example = None
            emg_vector = []
            for value in data_read_from_file:
                emg_vector.append(value)
                if (len(emg_vector) >= 8):
                    if example is None:
                        example = emg_vector
                    else:
                        example = np.row_stack((example, emg_vector))
                    emg_vector = []

            norm_emg = np.interp(example, (emg_min, emg_max), (-1, 1))

            emg.append(norm_emg)
            labels.append(label)

    if desired_labels is not None:
        # desired_labels = [0, 1, 2, 3, 4]

        filtered_labels = []
        filtered_emg = []
        for i, label in enumerate(labels):
            if label in desired_labels:
                filtered_emg.append(emg[i])
                filtered_labels.append(label)

        emg = filtered_emg
        labels = filtered_labels

    return emg, labels


def get_mad_windows(data_path, window_size, window_increment, emg_min = -128, emg_max = 127, desired_labels = None, return_lists=False):
    emg_list, label_list = get_mad_sample(data_path, emg_min, emg_max, desired_labels)

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

def mad_augmentation(emg, actions, num_augmentation):
    """Discrete emg data augmentation using random interpolation

    Args:
        emg (list): list of emg signals for each dof activation
        actions (list): list of dof activations for each discrete action

    Returns:
        sample_emg (torch.tensor): sampled emgs. size=[num_augmentation, num_channels, window_size]
        sample_actions (torch.tensor): sampled actions. size=[num_augmentation, act_dim]
    """
    idx_baseline = [i for i in range(len(actions)) if torch.all(actions[i] == 0)][0]
    idx_pos = [i for i in range(len(actions)) if torch.any(actions[i] > 0)]
    idx_neg = [i for i in range(len(actions)) if torch.any(actions[i] < 0)]
    
    emg_baseline = emg[idx_baseline]
    action_baseline = actions[idx_baseline]
    emg_pos = torch.stack([emg[i] for i in idx_pos])
    action_pos = torch.stack([actions[i] for i in idx_pos])
    emg_neg = torch.stack([emg[i] for i in idx_neg])
    action_neg = torch.stack([actions[i] for i in idx_neg])
    
    act_dim = action_baseline.shape[-1]
    sample_actions = torch.rand(num_augmentation, act_dim) * 2 - 1
    
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
            (1 - is_pos) * abs_action * neg_component + \
            sample_baseline
        )

    sample_emg = sample_emg + sample_baseline
    
    return sample_emg, sample_actions

def compute_features(windows, feature_list=['MAV', 'SSC', 'ZC', 'WL']):
    features = FeatureExtractor().extract_features(
        feature_list, 
        windows.numpy()
    )
    features = np.stack(list(features.values()), axis=-1)
    features = torch.from_numpy(features).flatten(start_dim=1).to(torch.float32)
    return features