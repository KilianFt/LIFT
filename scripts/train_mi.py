import os
import pickle
import wandb
import torch
import logging
import numpy as np
import lightning as L
from pytorch_lightning.loggers import WandbLogger

from configs import BaseConfig
from lift.environments.gym_envs import NpGymEnv
from lift.environments.emg_envs import EMGEnv
from lift.environments.simulator import SimulatorFactory
from lift.environments.rollout import rollout

from lift.teacher import load_teacher
from lift.datasets import get_dataloaders
from lift.controllers import MITrainer, EMGAgent
from lift.environments.teacher_envs import ConditionedTeacher
from lift.utils import normalize


def maybe_rollout(env: EMGEnv, policy: EMGAgent, config: BaseConfig, use_saved=True):
    rollout_file = config.rollout_data_path / f"mi_rollout_data.pkl"
    if use_saved and rollout_file.exists():
        print(f"\nload rollout data from file: {rollout_file}")
        with open(rollout_file, "rb") as f:
            data = pickle.load(f)
    else:
        print("collecting training data from teacher")
        data = rollout(
            env,
            policy,
            n_steps=config.mi.n_steps_rollout,
            terminate_on_done=False,
            reset_on_done=True,
            random_pertube_prob=config.mi.random_pertube_prob,
            action_noise=config.mi.action_noise,
        )
        rollout_file.parent.mkdir(exist_ok=True, parents=True)
        with open(rollout_file, "wb") as f:
            pickle.dump(data, f)
    return data

def get_ft_data(data, emg_mu, emg_sd):
    ft_data = {
        "obs": data["obs"]["observation"],
        "emg_obs": data["obs"]["emg_observation"],
        "intended_action": data["info"]["intended_action"],
    }
    ft_data = {k: torch.from_numpy(v).to(torch.float32) for k, v in ft_data.items()}
    ft_data["emg_obs"] = normalize(ft_data["emg_obs"], emg_mu, emg_sd)
    return ft_data

def combine_pt_ft_data(pt_data: dict, ft_data: dict):
    """Upsample and combine pretrain and fine tuning data"""
    if "val" in pt_data.keys():
        pt_train_len = len(pt_data["train"]["pt_emg_obs"])
        pt_val_len = len(pt_data["val"]["pt_emg_obs"])
        pt_len = pt_train_len + pt_val_len
        ft_len = len(ft_data["emg_obs"])

        if pt_len >= ft_len:
            idxs = torch.randint(0, ft_len, (pt_len,))
            idx_train = idxs[:pt_train_len]
            idx_val = idxs[pt_train_len:]
            data = pt_data
            data["train"]["obs"] = ft_data["obs"][idx_train]
            data["train"]["emg_obs"] = ft_data["emg_obs"][idx_train]
            data["train"]["intended_action"] = ft_data["intended_action"][idx_train]
            data["val"]["obs"] = ft_data["obs"][idx_val]
            data["val"]["emg_obs"] = ft_data["emg_obs"][idx_val]
            data["val"]["intended_action"] = ft_data["intended_action"][idx_val]
        elif pt_len < ft_len:
            train_val_ratio = pt_train_len / pt_val_len
            n_samples_val = int(ft_len / (1 + train_val_ratio))
            n_samples_train = ft_len - n_samples_val

            pt_idxs_train = torch.randint(0, pt_train_len, (n_samples_train,))
            pt_idxs_val = torch.randint(0, pt_val_len, (n_samples_val,))
            ft_idxs = torch.randint(0, ft_len, (ft_len,))
            data = {
                "train": {
                    "pt_emg_obs": pt_data["train"]["pt_emg_obs"][pt_idxs_train],
                    "pt_act": pt_data["train"]["pt_act"][pt_idxs_train],
                    "obs": ft_data["obs"][ft_idxs[:n_samples_train]],
                    "emg_obs": ft_data["emg_obs"][ft_idxs[:n_samples_train]],
                    "intended_action": ft_data["intended_action"][ft_idxs[:n_samples_train]],
                },
                "val": {
                    "pt_emg_obs": pt_data["val"]["pt_emg_obs"][pt_idxs_val],
                    "pt_act": pt_data["val"]["pt_act"][pt_idxs_val],
                    "obs": ft_data["obs"][ft_idxs[n_samples_train:]],
                    "emg_obs": ft_data["emg_obs"][ft_idxs[n_samples_train:]],
                    "intended_action": ft_data["intended_action"][ft_idxs[n_samples_train:]],
                },
            }

    else:
        pt_len = len(pt_data["pt_emg_obs"])
        ft_len = len(ft_data["emg_obs"])

        if pt_len >= ft_len:
            idx = torch.randint(0, ft_len, (pt_len,))
            data = pt_data
            data["obs"] = ft_data["obs"][idx]
            data["emg_obs"] = ft_data["emg_obs"][idx]
            data["intended_action"] = ft_data["intended_action"][idx]
        elif pt_len < ft_len:
            idx = torch.randint(0, pt_len, (ft_len,))
            data = ft_data
            data["pt_emg_obs"] = pt_data["pt_emg_obs"][idx]
            data["pt_act"] = pt_data["pt_act"][idx]
    return data

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

def validate(env, teacher, sim, encoder, mu, sd, logger):
    emg_env = EMGEnv(env, teacher, sim)
    agent = EMGAgent(encoder, mu, sd)
    data = rollout(
        emg_env, 
        agent, 
        n_steps=3000, 
        sample_mean=True,
        terminate_on_done=False,
        reset_on_done=True,
    )
    mean_rwd, std_rwd, mae = validate_data(data, logger)
    return mean_rwd, std_rwd, mae

def train(data_dict, model, logger, config: BaseConfig):
    train_dataloader, val_dataloader = get_dataloaders(
        data_dict=data_dict,
        train_ratio=config.mi.train_ratio,
        batch_size=config.mi.batch_size,
        num_workers=config.num_workers,
    )

    trainer = L.Trainer(
        max_epochs=config.mi.epochs, 
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


def main():
    config = BaseConfig()

    if config.use_wandb:
        _ = wandb.init(project='lift', tags=['train_mi'])
        config = BaseConfig(**wandb.config)
        logger = WandbLogger()
        wandb.config.update(config.model_dump())
    else:
        logger = None
    
    L.seed_everything(config.seed)

    # teacher = load_teacher(config)
    teacher = load_teacher(config, meta=True)
    teacher = ConditionedTeacher(
        teacher, 
        noise_range=[0., 0.], 
        noise_slope_range=[0., 0.], 
        alpha_range=[1., 1.],
    )
    teacher.reset()

    data_path = (config.mad_data_path / config.target_person / "training0").as_posix()
    sim = SimulatorFactory.create_class(
        data_path,
        config,
    )

    env = NpGymEnv(
        "FetchReachDense-v2", 
        cat_obs=True, 
        cat_keys=config.teacher.env_cat_keys,
    )

    # init trainer
    trainer = MITrainer(config, env, teacher, pretrain=False, supervise=True)

    pt_encoder_state_dict = torch.load(config.models_path / "pretrain_mi_encoder.pt")
    pt_critic_state_dict = torch.load(config.models_path / "pretrain_mi_critic.pt")
    trainer.encoder.load_state_dict(pt_encoder_state_dict)
    trainer.critic.load_state_dict(pt_critic_state_dict)

    with open(os.path.join(config.data_path, "pt_dataset.pkl"), 'rb') as f:
        pt_data = pickle.load(f)
    emg_mu = pt_data["mu"]
    emg_sd = pt_data["sd"]
    pt_data.pop("mu", None)
    pt_data.pop("sd", None)

    # collect user data
    emg_env = EMGEnv(env, teacher, sim)
    emg_policy = EMGAgent(trainer.encoder, emg_mu, emg_sd)
    data = maybe_rollout(emg_env, emg_policy, config, use_saved=False)
    validate_data(data, logger)

    # append data with supervised data
    ft_data = get_ft_data(data, emg_mu, emg_sd)
    data = combine_pt_ft_data(pt_data, ft_data)

    print("data shapes", {k: v.shape for k, v in data["train"].items()})
    print("pt_emg_obs mean", data["train"]["pt_emg_obs"].mean())
    print("emg_obs mean", data["train"]["emg_obs"].mean())

    # test once before train
    validate(env, teacher, sim, trainer.encoder, emg_mu, emg_sd, logger)

    train(data, trainer, logger, config)
    
    # test once after train
    validate(env, teacher, sim, trainer.encoder, emg_mu, emg_sd, logger)

    # torch.save(trainer, config.models_path / 'mi.pt')
    wandb.finish()


if __name__ == "__main__":
    main()
