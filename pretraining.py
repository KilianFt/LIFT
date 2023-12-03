from collections import defaultdict

import gymnasium as gym
from torch.utils.data import DataLoader
import lightning as L

from lift.simulator import EMGSimulator
from lift.datasets import EMGSLDataset
from lift.controllers import MLP, EMGPolicy


def rollout(emg_env, config):
    observation = emg_env.reset()
    history = defaultdict(list)

    for _ in range(config.n_steps_rollout):
        action, _ = emg_env.get_ideal_action(observation)
        observation, reward, terminated, info = emg_env.step(action)

        history['emg_observation'].append(observation["emg_observation"][-1])
        history['action'].append(action[-1])

        if terminated:
            observation = emg_env.reset()

    return history


def train_policy(emg_env, config):
    history = rollout(emg_env, config)

    dataset = EMGSLDataset(obs=history['emg_observation'], action=history['action'])
    train_dataloader = DataLoader(dataset, batch_size=config.batch_size)

    input_size = config.n_channels * config.window_size
    action_size = emg_env.action_space.shape[0]

    mlp = MLP(input_size=input_size, output_size=action_size, hidden_sizes=[128,128])
    pl_model = EMGPolicy(lr=config.lr, model=mlp)

    trainer = L.Trainer(max_epochs=config.epochs, log_every_n_steps=1, check_val_every_n_epoch=1, enable_checkpointing=False) 
    trainer.fit(model=pl_model, train_dataloaders=train_dataloader)

    return pl_model
