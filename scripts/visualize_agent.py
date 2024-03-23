from pathlib import Path
import argparse

import torch
import gymnasium as gym
from stable_baselines3 import TD3

from configs import BaseConfig
from lift.environments.emg_envs import EMGWrapper
from lift.controllers import EMGAgent
from lift.environments.simulator import WindowSimulator
from lift.environments.rollout import rollout
from lift.environments.gym_envs import NpGymEnv
from lift.teacher import load_teacher

def visualize_teacher():
    config = BaseConfig()
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

def visualize_encoder():
    base_path = Path(__file__).resolve().parents[1]
    teacher_filename = base_path / 'models' / 'teacher.zip'
    encoder_filename = base_path / 'models' / 'encoder.pt'

    raw_env = gym.make('FetchReachDense-v2', render_mode="human")
    teacher = TD3.load(teacher_filename, env=raw_env)

    config = BaseConfig()
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
    # sim.fit_normalization_params()

    emg_env = EMGWrapper(teacher, sim)
    encoder = torch.load(encoder_filename)
    agent = EMGAgent(policy=encoder)
    rewards = evaluate_policy(
        emg_env, agent, eval_steps=1000, use_terminate=False
    )
    print("reward", rewards.mean())

    emg_env.close()
    return 

def main(args):
    print(f"\nvisualizing {args['agent']}")

    if args["agent"] == "teacher":
        visualize_teacher()
    elif args["agent"] == "encoder":
        visualize_encoder()
    else:
        raise ValueError("agent to visualize not recognized")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["teacher", "encoder"])
    args = vars(parser.parse_args())

    main(args)