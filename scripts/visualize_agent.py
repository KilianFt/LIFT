import argparse
import torch

from configs import BaseConfig
from lift.environments.emg_envs import EMGEnv
from lift.environments.user_envs import UserEnv
from lift.environments.teacher_envs import TeacherEnv
from lift.environments.simulator import WindowSimulator
from lift.environments.gym_envs import NpGymEnv
from lift.environments.rollout import rollout

from lift.controllers import EMGAgent
from lift.teacher import load_teacher

def visualize_teacher(config: BaseConfig, meta=False, sample_mean=False):
    teacher = load_teacher(config, meta=meta)
    
    env = NpGymEnv(
        "FetchReachDense-v2", 
        cat_obs=True, 
        cat_keys=config.teacher.env_cat_keys,
        render_mode="human",
    )
    if meta:
        env = TeacherEnv(env, config.noise_range, config.alpha_range)
    
    data = rollout(
        env,
        teacher,
        n_steps=1000,
        sample_mean=sample_mean,
        terminate_on_done=False,
        reset_on_done=True,
    )
    print(f"mean reward: {data['rwd'].mean():.4f}")

def visualize_encoder(config: BaseConfig, encoder_type, sample_mean=False):
    teacher = load_teacher(config)

    env = NpGymEnv(
        "FetchReachDense-v2", 
        cat_obs=True, 
        cat_keys=config.teacher.env_cat_keys,
        render_mode="human",
        max_episode_steps=100,
    )
    sim = WindowSimulator(
        config,
        return_features=True,
    ) 
    sim.fit_params_to_mad_sample(
        str(config.mad_data_path / "Female0/training0/")
    )
    emg_env = EMGEnv(env, teacher, sim)
    
    # load encoder
    if encoder_type == "bc":
        trainer = torch.load(config.models_path / "bc.pt")
    elif encoder_type == "mi":
        trainer = torch.load(config.models_path / "mi.pt")
    
    agent = EMGAgent(policy=trainer.encoder)
    
    data = rollout(
        emg_env,
        agent,
        n_steps=1000,
        sample_mean=sample_mean,
        terminate_on_done=False,
        reset_on_done=True
    )
    print(f"mean reward: {data['rwd'].mean():.4f}") 

def visualize_user(config: BaseConfig, encoder_type, sample_mean=False):
    teacher = load_teacher(config)

    env = NpGymEnv(
        "FetchReachDense-v2", 
        cat_obs=True, 
        cat_keys=config.teacher.env_cat_keys,
        render_mode="human",
        max_episode_steps=100,
    )
    sim = WindowSimulator(
        config,
        return_features=True,
    ) 
    sim.fit_params_to_mad_sample(
        str(config.mad_data_path / "Female0/training0/")
    )
    
    # load encoder
    trainer = torch.load(config.models_path / "mi.pt")
    
    emg_agent = EMGAgent(policy=trainer.encoder)
    emg_env = UserEnv(env, emg_agent, sim)
    
    data = rollout(
        emg_env,
        teacher,
        n_steps=1000,
        sample_mean=sample_mean,
        terminate_on_done=False,
        reset_on_done=True
    )
    print(f"mean reward: {data['rwd'].mean():.4f}") 

"""TODO: differentiate env view and emg policy in argparse"""
def main(args):
    print(f"\nvisualizing {args['agent']}")
    config = BaseConfig()

    if args["agent"] in ["teacher", "meta_teacher"]:
        meta = True if args["agent"] == "meta_teacher" else False
        visualize_teacher(config, meta, args["sample_mean"])
    elif args["agent"] in ["bc", "mi"]:
        visualize_encoder(config, args["agent"], args["sample_mean"])
    elif args["agent"] in ["user"]:
        visualize_user(config, args["agent"], args["sample_mean"])
    else:
        raise ValueError("agent to visualize not recognized")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    bool_ = lambda x: x.lower() == "true"
    parser.add_argument("--agent", type=str, choices=["teacher", "meta_teacher", "bc", "mi", "user"])
    parser.add_argument("--sample_mean", type=bool_, default=False)
    args = vars(parser.parse_args())

    main(args)