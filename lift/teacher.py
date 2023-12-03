import os
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy


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
        mean_reward, _ = evaluate_policy(teacher, teacher.get_env(), n_eval_episodes=10)
        print(f"Mean reward {mean_reward}")

    return teacher


def train_teacher(env, config):
    teacher = TD3("MultiInputPolicy", env, verbose=1)
    teacher.learn(total_timesteps=config.teacher_train_timesteps)

    return teacher
