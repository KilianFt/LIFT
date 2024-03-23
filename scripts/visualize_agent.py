from pathlib import Path
import argparse

import torch
import gymnasium as gym
from stable_baselines3 import TD3

from configs import BaseConfig
from lift.environments.emg_envs import EMGEnv
from lift.controllers import EMGAgent
from lift.environments.simulator import WindowSimulator
from lift.environments.rollout import rollout
from lift.environments.gym_envs import NpGymEnv
from lift.teacher import load_teacher

def visualize_teacher(config: BaseConfig):
    teacher = load_teacher(config)
    
    env = NpGymEnv(
        "FetchReachDense-v2", 
        cat_obs=True, 
        cat_keys=config.teacher.env_cat_keys,
        render_mode="human",
    )
    data = rollout(
        env,
        teacher,
        n_steps=1000,
        terminate_on_done=False,
        reset_on_done=True
    )
    print(f"mean reward: {data['rwd'].mean():.4f}")

def visualize_encoder(config: BaseConfig):
    teacher = load_teacher(config)

    env = NpGymEnv(
        "FetchReachDense-v2", 
        cat_obs=True, 
        cat_keys=config.teacher.env_cat_keys,
        render_mode="human",
    )
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
    env = EMGEnv(env, teacher, sim)
    
    # load encoder
    encoder = torch.load(config.models_path / "encoder.pt")
    agent = EMGAgent(policy=encoder)
    
    data = rollout(
        env,
        agent,
        n_steps=1000,
        terminate_on_done=False,
        reset_on_done=True
    )
    print(f"mean reward: {data['rwd'].mean():.4f}")

def main(args):
    print(f"\nvisualizing {args['agent']}")
    config = BaseConfig()

    if args["agent"] == "teacher":
        visualize_teacher(config)
    elif args["agent"] == "encoder":
        visualize_encoder(config)
    else:
        raise ValueError("agent to visualize not recognized")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["teacher", "encoder"])
    args = vars(parser.parse_args())

    main(args)