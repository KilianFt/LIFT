import pickle
import hashlib
import torch

def cross_entropy(p, q, eps=1e-6):
    logq = torch.log(q + eps)
    ce = -torch.sum(p * logq, dim=-1)
    return ce

def hash_config(config):
    print("Hashing hyperparameters")
    conf_dict = config.model_dump()
    config_hash = hashlib.sha256(pickle.dumps(conf_dict)).hexdigest()
    return config_hash
