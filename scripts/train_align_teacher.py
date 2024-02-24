import os
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import TD3
from torch.utils.data import DataLoader, random_split

import lightning as L
from pytorch_lightning.loggers import WandbLogger

from configs import BaseConfig
from lift.simulator.simulator import WindowSimulator
from lift.evaluation import evaluate_policy
from lift.datasets import EMGSLDataset
from lift.controllers import EMGEncoder, EMGAgent
from lift.environment import EMGWrapper

def load_teacher(config):
    teacher_filename = config.model_path / 'teacher.zip'

    env = gym.make('FetchReachDense-v2')

    print('Loading trained teacher')
    teacher = TD3.load(teacher_filename, env=env)    
    return teacher

def get_dataloaders(observations, actions, train_percentage=0.8, batch_size=32):
    dataset = EMGSLDataset(obs=observations, action=actions)

    train_size = int(train_percentage * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader

def train(sim, actions, model, logger, config):
    features = sim(actions)
    train_dataloader, val_dataloader = get_dataloaders(features, actions, batch_size=config.batch_size)

    trainer = L.Trainer(
        max_epochs=config.epochs, 
        log_every_n_steps=1, 
        check_val_every_n_epoch=1,
        enable_checkpointing=False, 
        gradient_clip_val=config.gradient_clip_val,
        logger=logger,
    )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

def main():
    config = BaseConfig()
    L.seed_everything(config.seed)

    logger = WandbLogger(project='lift', tags='align_teacher')
    
    teacher = load_teacher(config)
    encoder = EMGEncoder(config)
    sim = WindowSimulator(
        action_size=config.action_size, 
        num_bursts=config.simulator.n_bursts, 
        num_channels=config.n_channels,
        window_size=config.window_size, 
        return_features=True,
    )
    sim.fit_params_to_mad_sample(
        str(config.mad_data_path / "Female0/training0/")
    )

    # collect teacher data
    num_samples = 5000
    data = evaluate_policy(
        teacher.get_env(), 
        teacher, 
        eval_steps=num_samples,
        use_terminate=False,
        is_sb3=True
    )
    mean_rwd = data["rwd"].mean()
    print("teacher reward", mean_rwd)
    if logger is not None:
        logger.log_metrics({"teacher reward": mean_rwd})

    # test once before train
    test_env = EMGWrapper(teacher, sim)
    agent = EMGAgent(encoder.encoder)
    eval_data = evaluate_policy(test_env, agent, eval_steps=1000, use_terminate=False)
    mean_rwd = eval_data["rwd"].mean()
    print("encoder reward", mean_rwd)
    if logger is not None:
        logger.log_metrics({"encoder reward": mean_rwd})

    train(sim, data["act"], encoder, logger, config)
    
    # test once after train
    test_env = EMGWrapper(teacher, sim)
    agent = EMGAgent(encoder.encoder)
    eval_data = evaluate_policy(test_env, agent, eval_steps=1000, use_terminate=False)
    mean_rwd = eval_data["rwd"].mean()
    print("encoder reward", mean_rwd)
    if logger is not None:
        logger.log_metrics({"encoder reward": mean_rwd})

if __name__ == "__main__":
    main()