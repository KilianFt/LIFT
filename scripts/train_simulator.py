import os
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows
from lift.datasets import get_mad_sample
from lift.simulator.trainer import Trainer

def extract_features_labels(emg_list, label_list, window_size=200, window_increment=50, feature_list=['MAV', 'SSC', 'ZC', 'WL']):
    """apply windows to every action"""
    fe = FeatureExtractor()

    features = [None] * len(emg_list)
    for i, emg in enumerate(emg_list):
        windows = get_windows(emg, window_size, window_increment)

        features[i] = fe.extract_features(feature_list, windows)
        features[i] = np.stack(list(features[i].values()), axis=-1)

    short_labels = [np.repeat(label, repeats=len(f)) for label, f in zip(label_list, features)]

    features = np.concatenate(features, axis=0)
    short_labels = np.concatenate(short_labels, axis=0)
    return features, short_labels

def main():
    np.random.seed(0)
    torch.manual_seed(0)
    
    mad_base_dir = '../datasets/MyoArmbandDataset'
    if not os.path.exists(mad_base_dir):
        os.system(f'git clone https://github.com/UlysseCoteAllard/MyoArmbandDataset {mad_base_dir}')

    data_paths = [p for p in glob.glob('../datasets/MyoArmbandDataset/PreTrainingDataset/*') if "Female" in p]
    
    feature_list = ['MAV', 'SSC', 'ZC', 'WL']
    window_size = 200

    # aggregrate all female data
    features = [None] * len(data_paths)
    labels = [None] * len(data_paths)
    for i, data_path in enumerate(data_paths):
        emg_list, label_list = get_mad_sample(os.path.join(data_path, 'training0'), filter_labels = True)
        features[i], labels[i] = extract_features_labels(
            emg_list, 
            label_list,
            window_size=window_size,
            window_increment=50,
            feature_list=feature_list,
        )
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    num_channels = features.shape[1]
    num_actions = len(np.unique(labels))

    features = torch.from_numpy(features).flatten(start_dim=1).to(torch.float32)
    labels = F.one_hot(torch.from_numpy(labels), num_classes=num_actions).to(torch.float32)
    data = torch.cat([features, labels], dim=-1)
    print(f"feature size: {features.shape}, label size: {labels.shape}")
    
    num_bursts = 1
    batch_size = 64
    grad_target = 1.
    d_iters = 50
    g_iters = 300
    trainer = Trainer(
        feature_list,
        num_actions, 
        num_channels,
        window_size,
        num_bursts,
        hidden_sizes=[128, 128, 128, 128],
        batch_size=batch_size,
        grad_target=grad_target,
        d_iters=d_iters, 
        g_iters=g_iters,
    )

    epochs = 100
    history = trainer.train(data, epochs)
    df_history = pd.DataFrame(history)

    plt.plot(df_history["d_loss"], label="d_loss")
    plt.plot(df_history["g_loss"], label="g_loss")
    plt.legend()
    plt.show()

    return 

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument()
    arglist = vars(parser.parse_args())
    return arglist

if __name__ == "__main__":
    # args = parse_args()
    main()