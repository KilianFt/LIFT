import os
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import TD3


def maybe_train_teacher(config):
    base_dir = Path(__file__).resolve().parents[1]
    teacher_filename = base_dir / 'models' / 'teacher.zip'

    if not os.path.exists(teacher_filename):
        print('Training teacher')
        os.makedirs(teacher_filename.parent, exist_ok=True)

        teacher = train_teacher(config)
        teacher.save(teacher_filename)
    else:
        print('Loading trained teacher')
        env = gym.make('FetchReachDense-v2', max_episode_steps=100)
        teacher = TD3.load(teacher_filename, env=env)

    return teacher


def train_teacher(config):
    env = gym.make('FetchReachDense-v2', max_episode_steps=100)

    teacher = TD3("MultiInputPolicy", env, verbose=1)
    teacher.learn(total_timesteps=config.teacher_train_timesteps)

    return teacher
