import pickle
import hashlib

import torch
import numpy as np
from libemg.feature_extractor import FeatureExtractor


def compute_features(windows, feature_list=['MAV', 'SSC', 'ZC', 'WL']):
    features = FeatureExtractor().extract_features(
        feature_list, 
        windows.numpy()
    )
    features = np.stack(list(features.values()), axis=-1)
    features = torch.from_numpy(features).flatten(start_dim=1).to(torch.float32)
    return features

def cross_entropy(p, q, eps=1e-6):
    logq = torch.log(q + eps)
    ce = -torch.sum(p * logq, dim=-1)
    return ce

def obs_wrapper(obs):
    if isinstance(obs, tuple):
        return obs[0]
    else:
        return obs
    

def hash_config(config):
    print("Hashing hyperparameters")
    values = sorted(config, key=lambda x: x[0])
    config_hash = hashlib.sha256(pickle.dumps(values)).hexdigest()
    return config_hash
