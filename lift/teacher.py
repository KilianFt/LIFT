import os
import gymnasium as gym
from stable_baselines3 import TD3


def load_teacher(config):
    teacher_filename = config.models_path / 'teacher.zip'

    env = gym.make('FetchReachDense-v2')

    print('Loading trained teacher')
    teacher = TD3.load(teacher_filename, env=env)    
    return teacher


def maybe_train_teacher(config):    
    teacher_filename = config.models_path / 'teacher.zip'

    env = gym.make('FetchReachDense-v2')

    if not os.path.exists(teacher_filename):
        print('Training teacher')
        os.makedirs(teacher_filename.parent, exist_ok=True)

        teacher = train_teacher(env, config)
        teacher.save(teacher_filename)
    else:
        print('Loading trained teacher')
        teacher = TD3.load(teacher_filename, env=env)

    return teacher


def train_teacher(env, config):
    teacher = TD3("MultiInputPolicy", env, verbose=1)
    teacher.learn(total_timesteps=config.teacher_train_timesteps)

    return teacher
