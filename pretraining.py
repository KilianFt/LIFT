from collections import defaultdict

import gymnasium as gym
from torch.utils.data import DataLoader
import lightning as L

from lift.simulator import EMGSimulator
from lift.datasets import EMGSLDataset
from lift.controllers import MLP, EMGPolicy


def rollout(teacher, config):
    env = gym.make('FetchReachDense-v2', max_episode_steps=100)
    emg_simulator = EMGSimulator(n_channels=config.n_channels)

    # FIXME might not always be index 0
    action_size = env.action_space.shape[0]

    observation, info = env.reset()
    history = defaultdict(list)

    for _ in range(config.n_steps_rollout):
        action, _ = teacher.predict(observation)
        emg_observation = emg_simulator.sample(action, n_timesteps=100)
        
        observation, reward, terminated, truncated, info = env.step(action)

        history['observation'].append(observation)
        history['emg_observation'].append(emg_observation)
        history['action'].append(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

    return history, action_size


def train_policy(teacher, config):
    history, action_size = rollout(teacher, config)

    dataset = EMGSLDataset(obs=history['emg_observation'], action=history['action'])
    train_dataloader = DataLoader(dataset, batch_size=config.batch_size)

    input_size = config.n_channels * config.window_size
    mlp = MLP(input_size=input_size, output_size=action_size, hidden_sizes=[128,128])
    pl_model = EMGPolicy(lr=1e-3, model=mlp)

    trainer = L.Trainer(max_epochs=50, log_every_n_steps=1, gradient_clip_val=0.5, check_val_every_n_epoch=1, enable_checkpointing=False) 
    trainer.fit(model=pl_model, train_dataloaders=train_dataloader)

    return pl_model
