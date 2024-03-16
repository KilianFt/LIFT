from pathlib import Path

import torch
import gymnasium as gym
from stable_baselines3 import TD3

from configs import BaseConfig
from lift.environment import EMGWrapper
from lift.controllers import EMGAgent
from lift.simulator.simulator import WindowSimulator
from lift.evaluation import evaluate_policy

def main():
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

if __name__ == "__main__":
    main()