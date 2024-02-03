import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from libemg.utils import get_windows

class EMGSLDataset(Dataset):
    def __init__(self, obs, action):
        self.obs = torch.tensor(np.stack(obs), dtype=torch.float32)
        self.action = torch.tensor(np.stack(action), dtype=torch.float32)

    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        return self.obs[idx], self.action[idx]


def get_mad_sample(data_path, emg_min = -128, emg_max = 127, filter_labels = False):
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

    if filter_labels:
        desired_labels = [0, 1, 2, 3, 4]

        filtered_labels = []
        filtered_emg = []
        for i, label in enumerate(labels):
            if label in desired_labels:
                filtered_emg.append(emg[i])
                filtered_labels.append(label)

        emg = filtered_emg
        labels = filtered_labels

    return emg, labels


def get_mad_windows(data_path, window_size, emg_min = -128, emg_max = 127, filter_labels = False):
    emg_list, label_list = get_mad_sample(data_path, emg_min, emg_max, filter_labels)

    sort_id = np.argsort(label_list)
    label_list = [label_list[i] for i in sort_id]
    emg_list = [emg_list[i] for i in sort_id]
    # I used labels 0 - 4 (including 4), where label 0 is rest

    min_len = min([len(emg) for emg in emg_list])
    short_emgs = [emg[:min_len,:] for emg in emg_list]
    windows_list = [get_windows(s_emg, window_size, window_size, as_tensor=True) for s_emg in short_emgs]
    windows = torch.stack(windows_list, dim=0)
    flat_windows = windows.flatten(start_dim=0, end_dim=1)

    n_repeats = windows_list[0].shape[0]
    short_labels = torch.tensor([np.repeat(label, repeats=n_repeats) for label in label_list])
    labels = short_labels.flatten(start_dim=0, end_dim=1)

    # actions = F.one_hot(short_labels, num_classes=5).float()
    # flat_actions = actions.flatten(start_dim=0, end_dim=1)
    return flat_windows, labels

