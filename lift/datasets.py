import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset


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
