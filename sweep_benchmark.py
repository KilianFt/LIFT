import wandb
import pandas as pd
import gymnasium as gym
from configs import BaseConfig
from lift.environment import ActionNoiseWrapper
from lift.teacher import maybe_train_teacher

def sweep(noise_levels, config):
    for i, noise_level in enumerate(noise_levels):
        train_env = ActionNoiseWrapper(
            gym.make("FetchReachDense-v2", max_episode_steps=100),
            noise_level,
            add_task_id=False
        )        
        test_env = ActionNoiseWrapper(
            gym.make("FetchReachDense-v2", max_episode_steps=100),
            noise_level=0.,
            add_task_id=False
        )

        teacher, stats = maybe_train_teacher(
            train_env, 
            test_env, 
            config, 
            f"teacher_{noise_level}.zip",
            test_eps=5,
        )

        stats["noise_level"] = noise_level
        # wandb.log(stats)
        pd.DataFrame([stats]).to_csv(
            "./models/sweep.csv",
            mode="w" if i == 0 else "a",
            header=True if i == 0 else False
        )

        train_env.close()
        test_env.close()

def main():
    config = BaseConfig()
    # run = wandb.init(project="lift", config=config)
    # debug
    # config.teacher_train_timesteps = 1

    noise_levels = [0., 0.1, 0.3, 0.5, 0.7, 1.]
    sweep(noise_levels, config)

if __name__ == "__main__":
    main()