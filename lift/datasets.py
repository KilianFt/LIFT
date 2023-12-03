import torch
from torch.utils.data import Dataset


class EMGSLDataset(Dataset):
    def __init__(self, obs, action):
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.action = torch.tensor(action, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        return self.obs[idx], self.action[idx]
