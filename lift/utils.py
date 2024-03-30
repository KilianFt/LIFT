import pickle
import hashlib
import torch

def cross_entropy(p, q, eps=1e-6):
    logq = torch.log(q + eps)
    ce = -torch.sum(p * logq, dim=-1)
    return ce

def hash_config(config):
    print("Hashing hyperparameters")
    values = sorted(config, key=lambda x: x[0])
    config_hash = hashlib.sha256(pickle.dumps(values)).hexdigest()
    return config_hash
