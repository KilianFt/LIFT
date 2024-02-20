from collections import defaultdict
from pathlib import Path
import pickle

import torch
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lift.datasets import EMGSLDataset
from lift.controllers import MLP, EMGPolicy, EMGAgent, EMGEncoder
from lift.evaluation import evaluate_emg_policy


def rollout(emg_env, config):
    observation = emg_env.reset()
    history = defaultdict(list)

    for _ in tqdm(range(config.n_steps_rollout), desc="Rollout", unit="item"):
        action, _ = emg_env.get_ideal_action(observation)
        observation, reward, terminated, info = emg_env.step(action)

        history['emg_observation'].append(observation["emg_observation"][-1])
        history['action'].append(action[-1])

        if terminated:
            observation = emg_env.reset()

    return history


def get_pretrain_dataloaders(history, config, train_percentage: float = 0.8):
    dataset = EMGSLDataset(obs=history['emg_observation'], action=history['action'])

    train_size = int(train_percentage * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)
    return train_dataloader, val_dataloader


def train_policy(emg_env, config):
    """Supervised pretraining"""
    hist_name = Path('datasets') / f'rollout_history_features_{(config.n_channels*config.window_size)}_v2.pkl'
    if hist_name.exists():
        with open(hist_name, 'rb') as f:
            history = pickle.load(f)
    else:
        history = rollout(emg_env, config)
        with open(hist_name, 'wb') as f:
            pickle.dump(history, f)

        # is this still needed when we fit params to MAD sample?
        # with open(Path('datasets') / f'rollout_params_{(config.n_channels*config.window_size)}.pkl', "wb") as f:
        #     pickle.dump({
        #         'weights': emg_env.emg_simulator.weights,
        #         'biases': emg_env.emg_simulator.biases,
        #     }, f)
    
    logger = WandbLogger(log_model="all")

    train_dataloader, val_dataloader = get_pretrain_dataloaders(history, config)

    # input_size = 32#config.n_channels * config.window_size FIXME
    # action_size = emg_env.action_space.shape[0]
    # hidden_sizes = [config.hidden_size for _ in range(config.n_layers)]
    # model = MLP(input_size=input_size, output_size=action_size, hidden_sizes=hidden_sizes, dropout=config.dropout)
    # pl_model = EMGPolicy(lr=config.lr, model=model)
    pl_model = EMGEncoder(config)
    agent = EMGAgent(policy=pl_model.encoder)

    data = evaluate_emg_policy(emg_env, agent)
    before_mean_reward = data['rwd'].mean()
    print(f"Pretrain reward before training {before_mean_reward}")
    wandb.log({'pretrain_reward': before_mean_reward})
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path('models'), 
        filename='policy_{epoch}_{val_loss:.2f}',
        save_top_k=config.save_top_k, 
        every_n_epochs=config.checkpoint_frequency,
        verbose=True,
    )
    trainer = L.Trainer(
        max_epochs=config.epochs, 
        log_every_n_steps=1, 
        check_val_every_n_epoch=1,
        enable_checkpointing=True, 
        logger=logger, 
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=pl_model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader
    )

    data = evaluate_emg_policy(emg_env, agent)
    mean_reward = data['rwd'].mean()
    print(f"Pretrain reward {mean_reward}")
    wandb.log({'pretrain_reward': mean_reward})

    torch.save(pl_model.encoder, Path('models') / 'encoder.pt')

    return agent
