import pickle
import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import wandb
import lightning as L
from pytorch_lightning.loggers import WandbLogger

from configs import BaseConfig
from lift.simulator.simulator import WindowSimulator
from lift.evaluation import evaluate_policy
from lift.datasets import EMGSLDataset
from lift.controllers import EMGEncoder, EMGAgent
from lift.environment import EMGWrapper, rollout
from lift.teacher import load_teacher
from lift.utils import hash_config

def get_dataloaders(observations, actions, train_percentage=0.8, batch_size=32, num_workers=4):
    dataset = EMGSLDataset(obs=observations, action=actions)

    train_size = int(train_percentage * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=True, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)
    return train_dataloader, val_dataloader


def train(sim, actions, model, logger, config):
    features = sim(actions)
    train_dataloader, val_dataloader = get_dataloaders(features,
                                                       actions,
                                                       batch_size=config.batch_size,
                                                       num_workers=config.num_workers)

    trainer = L.Trainer(
        max_epochs=config.epochs, 
        log_every_n_steps=1, 
        check_val_every_n_epoch=1,
        enable_checkpointing=False, 
        gradient_clip_val=config.gradient_clip_val,
        logger=logger,
    )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


def validate(teacher, env, obs_agg_fun, sim, encoder, logger):
    test_env = EMGWrapper(teacher, env, obs_agg_fun, sim)
    agent = EMGAgent(encoder.encoder)
    rewards = evaluate_policy(test_env, agent, eval_steps=1000, use_terminate=False)
    mean_rwd = rewards.mean()
    print("encoder reward", mean_rwd)
    if logger is not None:
        logger.log_metrics({"encoder reward": mean_rwd})
    return rewards


def maybe_rollout(teacher, env, obs_agg_fun, config):
    config_hash = hash_config(config)
    rollout_file = config.rollout_data_path / f"data_{config_hash}.pkl"
    if rollout_file.exists():
        with open(rollout_file, "rb") as f:
            data = pickle.load(f)
    else:
        data = rollout(
            env, 
            teacher,
            obs_agg_fun,
            n_steps=config.n_steps_rollout,
            random_pertube_prob=config.random_pertube_prob,
            action_noise=config.action_noise,
        )
        rollout_file.parent.mkdir(exist_ok=True, parents=True)
        with open(rollout_file, "wb") as f:
            pickle.dump(data, f)
    return data


def main():
    config = BaseConfig()
    L.seed_everything(config.seed)
    if config.use_wandb:
        _ = wandb.init(project='lift', tags='align_teacher')
        logger = WandbLogger()
    else:
        logger = None
    
    teacher = load_teacher(config)
    sim = WindowSimulator(
        action_size=config.action_size, 
        num_bursts=config.simulator.n_bursts, 
        num_channels=config.n_channels,
        window_size=config.window_size, 
        return_features=True,
    )
    sim.fit_params_to_mad_sample(
        (config.mad_data_path / "Female0"/ "training0").as_posix()
    )

    env = gym.make('FetchReachDense-v2')
    def obs_agg_fun(obs):
        return torch.from_numpy(
            np.concatenate([obs["observation"], obs["desired_goal"], obs["achieved_goal"]])
        ).to(torch.float32)
    
    # collect teacher data
    data = maybe_rollout(teacher, env, obs_agg_fun, config)
    mean_rwd = data["rwd"].mean()
    print("teacher reward", mean_rwd)
    if logger is not None:
        logger.log_metrics({"teacher reward": mean_rwd})

    # test once before train
    encoder = EMGEncoder(config)
    validate(teacher, env, obs_agg_fun, sim, encoder, logger)

    train(sim, data["act"], encoder, logger, config)
    
    validate(teacher, env, obs_agg_fun, sim, encoder, logger)

    torch.save(encoder.encoder, config.models_path / 'encoder.pt')


if __name__ == "__main__":
    main()
