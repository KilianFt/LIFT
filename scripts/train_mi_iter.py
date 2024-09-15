"""
Iterative MI training
"""
import os
import logging
import copy
import pickle
import wandb
import torch
import numpy as np
import lightning as L
from pytorch_lightning.loggers import WandbLogger

from configs import BaseConfig
from lift.environments.gym_envs import NpGymEnv
from lift.environments.emg_envs import EMGEnv
from lift.environments.simulator import SimulatorFactory
from lift.environments.rollout import rollout

from lift.teacher import load_teacher
from lift.environments.teacher_envs import apply_gaussian_drift, ConditionedTeacher
from lift.datasets import get_dataloaders
from lift.controllers import MITrainer, EMGAgent

# FIXME move this to dataset or somewhere else
from train_mi import get_ft_data, combine_pt_ft_data


logging.basicConfig(level=logging.INFO)


def validate_data(data, logger):
    assert data["info"]["intended_action"].shape == data["act"].shape
    mae = np.abs(data["info"]["intended_action"] - data["act"]).mean()
    sum_rwd = data["rwd"].sum()
    mean_rwd = data["rwd"].mean()
    std_rwd = data["rwd"].std()
    n_episodes = data["done"].sum()
    logging.info(f"encoder reward mean: {mean_rwd:.4f}, std: {std_rwd:.4f}, mae: {mae:.4f}")
    if logger is not None:
        logger.log_metrics({"encoder_reward": mean_rwd,
                            "encoder_reward_sum": sum_rwd,
                            "n_episodes": n_episodes,
                            "encoder_mae": mae})

    return mean_rwd, std_rwd, mae

# def validate(env, teacher, sim, encoder, mu, sd, logger):
#     emg_env = EMGEnv(env, teacher, sim)
#     agent = EMGAgent(encoder, mu, sd)
#     data = rollout(
#         emg_env, 
#         agent, 
#         n_steps=3000, 
#         sample_mean=True,
#         terminate_on_done=False,
#         reset_on_done=True,
#     )
#     mean_rwd, std_rwd, mae = validate_data(data, logger)
#     return mean_rwd, std_rwd, mae


def train(data, model, logger, config: BaseConfig):
    train_dataloader, val_dataloader = get_dataloaders(
        data_dict=data,
        train_ratio=config.mi.train_ratio,
        batch_size=config.mi.batch_size,
        num_workers=config.num_workers,
    )

    trainer = L.Trainer(
        max_epochs=config.mi.epochs, 
        max_steps=config.mi.max_steps,
        log_every_n_steps=1, 
        check_val_every_n_epoch=1,
        enable_checkpointing=False, 
        gradient_clip_val=config.gradient_clip_val,
        logger=logger,
    )
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader,
    )

def append_dataset(dataset, data):
    for k, v in copy.deepcopy(data).items():
        if k not in dataset:
            dataset[k] = v
        else:
            if isinstance(v, torch.Tensor):
                dataset[k] = torch.cat([dataset[k], v], dim=0)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    dataset[k][kk] = np.concatenate([dataset[k][kk], vv], axis=0)
            else:
                dataset[k] = np.concatenate([dataset[k], v], axis=0)

def main():
    config = BaseConfig()
    if config.use_wandb:
        tags = ['align_teacher']
        _ = wandb.init(project='lift', tags=tags)
        config = BaseConfig(**wandb.config)
        logger = WandbLogger()
        wandb.config.update(config.model_dump())
    else:
        logger = None

    L.seed_everything(config.seed)

    env = NpGymEnv(
        "FetchReachDense-v2", 
        cat_obs=True, 
        cat_keys=config.teacher.env_cat_keys,
    )
    sim = SimulatorFactory.create_class(
        (config.mad_data_path / config.target_person / "training0").as_posix(),
        config,
        return_features=True,
    )
    
    # when noise, slope and alpha is set, overwrite ranges here
    if config.noise is not None:
        config.noise_range = [config.noise, config.noise]
    if config.noise_slope is not None:
        config.noise_slope_range = [config.noise_slope, config.noise_slope]
    if config.alpha is not None:
        config.alpha_range = [config.alpha, config.alpha]
    wandb.config.update(config.model_dump(), allow_val_change=True)

    # init user with known noise and alpha for controlled experiment
    user = load_teacher(config, meta=True, filename="teacher_meta_3_layer_relu.pt")
    user = ConditionedTeacher(
        user,
        noise_range=config.noise_range,
        noise_slope_range=config.noise_slope_range,
        alpha_range=config.alpha_range,
        alpha_apply_range=config.alpha_apply_range,
        user_bias=config.user_bias,
    )
    user.reset()

    # load models
    teacher = load_teacher(config, meta=True, filename="teacher_meta_3_layer_relu.pt")
    teacher = ConditionedTeacher(
        teacher, 
        noise_range=[0., 0.], 
        noise_slope_range=[0., 0.], 
        alpha_range=[1., 1.],
    )
    teacher.reset()
    bc_encoder_state_dict = torch.load(config.models_path / "pretrain_mi_encoder.pt")
    bc_critic_state_dict = torch.load(config.models_path / "pretrain_mi_critic.pt")
    
    trainer = MITrainer(config, env, teacher, supervise=True)
    trainer.encoder.load_state_dict(bc_encoder_state_dict)
    trainer.critic.load_state_dict(bc_critic_state_dict)
    
    with open(os.path.join(config.data_path, "pt_dataset.pkl"), 'rb') as f:
        pt_data = pickle.load(f)
    emg_mu = pt_data["mu"]
    emg_sd = pt_data["sd"]
    pt_data.pop("mu", None)
    pt_data.pop("sd", None)

    # interactive training loop
    num_sessions = config.mi.num_sessions

    dataset = {}

    for i in range(num_sessions):
        # collect user data
        print("user meta vars", user.get_meta_vars())
        emg_env = EMGEnv(env, user, sim)
        emg_policy = EMGAgent(trainer.encoder, emg_mu, emg_sd)

        data = rollout(
            emg_env,
            emg_policy,
            n_steps=config.mi.n_steps_rollout,
            sample_mean=True,
            terminate_on_done=False,
            reset_on_done=True,
            random_pertube_prob=config.mi.random_pertube_prob,
            action_noise=config.mi.action_noise,
        )
        validate_data(data, logger)
        ft_data = get_ft_data(data, emg_mu, emg_sd)
        append_dataset(dataset, ft_data)

        # we might want to aggregate data from multiple sessions
        dataset = dataset if config.mi.aggregate_data else data
        print("dataset size", len(dataset["obs"]))
        combined_data = combine_pt_ft_data(pt_data, dataset)

        train(combined_data, trainer, logger, config)
        # if i == 1:
        #     import pdb
        #     pdb.set_trace()

        user_meta_vars = user.get_meta_vars()

        # update user meta vars
        if config.noise_drift is not None:
            user_meta_vars['noise'] = apply_gaussian_drift(
                user_meta_vars['noise'],
                config.noise_drift[0],
                config.noise_drift[1],
                config.noise_range,
            )

        if config.alpha_drift is not None:
            user_meta_vars['alpha'] = apply_gaussian_drift(
                user_meta_vars['alpha'],
                config.alpha_drift[0],
                config.alpha_drift[1],
                config.alpha_range,
            )
            logging.info(f"new alpha: {user_meta_vars['alpha']}")

        user.set_meta_vars(user_meta_vars)

    # test once at the end
    # validate(env, teacher, sim, trainer.encoder, emg_mu, emg_sd, logger)
    emg_env = EMGEnv(env, user, sim)
    emg_policy = EMGAgent(trainer.encoder, emg_mu, emg_sd)

    data = rollout(
        emg_env,
        emg_policy,
        n_steps=config.mi.n_steps_rollout,
        sample_mean=True,
        terminate_on_done=False,
        reset_on_done=True,
        random_pertube_prob=config.mi.random_pertube_prob,
        action_noise=config.mi.action_noise,
    )
    validate_data(data, logger)

    # torch.save(trainer, config.models_path / 'mi_iter.pt')

    if config.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
