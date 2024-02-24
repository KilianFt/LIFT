import argparse
import os
import glob
from pathlib import Path
import pickle
import random
import torch
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import TD3
from lift.environment import UserSimulator
from lift.controllers import MLP, EMGPolicy, EMGAgent
from lift.evaluation import evaluate_policy
from configs import BaseConfig

def load_decoder(decoder_path, env, config):
    checkpoint = torch.load(decoder_path)

    input_size = config.n_channels * config.window_size
    action_size = env.action_space.shape[0]
    hidden_sizes = [config.hidden_size for _ in range(config.n_layers)]

    model = MLP(
        input_size=input_size, 
        output_size=action_size, 
        hidden_sizes=hidden_sizes, 
        dropout=config.dropout,
    )
    pl_model = EMGPolicy(lr=config.lr, model=model)
    pl_model.load_state_dict(checkpoint["state_dict"])
    agent = EMGAgent(policy=pl_model.model)
    return agent

def train_user(train_decoder_path, test_decoder_path, save_path, config, eval_eps=5):
    env = gym.make('FetchReachDense-v2', max_episode_steps=100)

    train_decoder = load_decoder(train_decoder_path, env, config)
    test_decoder = load_decoder(test_decoder_path, env, config)
    with open(Path('datasets') / 'rollout_params_16.pkl', "rb") as f:
        emg_sim_params = pickle.load(f)
    
    train_env = UserSimulator(env, train_decoder, config)
    train_env.emg_simulator.weights = emg_sim_params["weights"]
    train_env.emg_simulator.biases = emg_sim_params["biases"]

    test_env = UserSimulator(env, test_decoder, config)
    test_env.emg_simulator.weights = emg_sim_params["weights"]
    test_env.emg_simulator.biases = emg_sim_params["biases"]
    
    if not os.path.exists(save_path):
        user = TD3("MultiInputPolicy", train_env, verbose=1)
        user.learn(total_timesteps=config.teacher_train_timesteps)
        user.save(save_path)
    else:
        print(f"load from pretrained: {save_path}")
        user = TD3.load(save_path, env=train_env)
    
    train_mean_rewards = [evaluate_policy(user.get_env(), user, is_sb3=True) for _ in range(eval_eps)]
    test_mean_rewards = [evaluate_policy(test_env, user, is_sb3=True) for _ in range(eval_eps)]
    
    stats = {
        "train_r_mean": np.mean(train_mean_rewards),
        "train_r_std": np.std(train_mean_rewards),
        "test_r_mean": np.mean(test_mean_rewards),
        "test_r_std": np.std(test_mean_rewards),
    }
    return stats

def main(args):
    """train teacher policy for different checkpoints, test on best checkpoint"""
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    
    config = BaseConfig()

    # load decoder
    for i, e in enumerate([0, 1, 2, 3, 4, 29]):
        train_decoder_path = glob.glob(f"models/policy_epoch={e}_*.ckpt")[0]
        test_decoder_path = glob.glob(f"models/policy_epoch=29_*.ckpt")[0]
        save_path = Path('models') / f"user_policy_{e}.zip"
        stats = train_user(train_decoder_path, test_decoder_path, save_path, config, eval_eps=10)
        print("eval performance", stats)
        
        pd.DataFrame([stats]).to_csv(
            "./models/user_sweep_.csv",
            mode="w" if i == 0 else "a",
            header=True if i == 0 else False
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = vars(parser.parse_args())

    main(args)