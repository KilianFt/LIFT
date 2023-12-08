import os
from pathlib import Path

import wandb
import gymnasium as gym
from stable_baselines3 import TD3

from lift.evaluation import evaluate_emg_policy


def maybe_train_teacher(config):    
    base_dir = Path(__file__).resolve().parents[1]
    teacher_filename = base_dir / 'models' / 'teacher.zip'

    env = gym.make('FetchReachDense-v2', max_episode_steps=100)

    if not os.path.exists(teacher_filename):
        print('Training teacher')
        os.makedirs(teacher_filename.parent, exist_ok=True)

        teacher = train_teacher(env, config)
        teacher.save(teacher_filename)
    else:
        print('Loading trained teacher')
        teacher = TD3.load(teacher_filename, env=env)

    mean_reward = evaluate_emg_policy(teacher.get_env(), teacher, is_teacher=True)
    wandb.log({'teacher_reward': mean_reward})
    print(f"Teacher reward {mean_reward}")

    return teacher


def train_teacher(env, config):
    teacher = TD3("MultiInputPolicy", env, verbose=1)
    teacher.learn(total_timesteps=config.teacher_train_timesteps)

    return teacher
