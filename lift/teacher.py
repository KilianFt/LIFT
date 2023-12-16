import os
from pathlib import Path

# import wandb
# import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3

from lift.evaluation import evaluate_emg_policy
from lift.evaluation import evaluate_teacher_policy


def maybe_train_teacher(train_env, test_env, config, filename, test_eps=5):    
    base_dir = Path(__file__).resolve().parents[1]
    teacher_filename = base_dir / 'models' / filename

    # env = gym.make('FetchReachDense-v2', max_episode_steps=100)

    print("filename", teacher_filename)

    if not os.path.exists(teacher_filename):
        print('Training teacher')
        os.makedirs(teacher_filename.parent, exist_ok=True)

        teacher = train_teacher(train_env, config)
        teacher.save(teacher_filename)
    else:
        print('Loading trained teacher')
        teacher = TD3.load(teacher_filename, env=test_env)
    
    train_reward = evaluate_teacher_policy(train_env, teacher, is_teacher=True)
    test_reward = np.mean([
        evaluate_teacher_policy(test_env, teacher, is_teacher=True) for _ in range(test_eps)
    ])
    # wandb.log({'teacher_reward': train_reward})
    print(f"Teacher train_reward {train_reward}")
    print(f"Teacher test_reward {test_reward}")

    stats = {
        "train_reward": train_reward,
        "test_reward": test_reward,
    }

    return teacher, stats


def train_teacher(env, config):
    teacher = TD3("MultiInputPolicy", env, verbose=1)
    teacher.learn(total_timesteps=config.teacher_train_timesteps)

    return teacher
