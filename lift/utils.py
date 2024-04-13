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

""" convert labels (0 to 6) to actions """
def mad_labels_to_actions(labels, recording_strength=1.0):
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
