import os
import pickle
import wandb
import torch
import torch.nn as nn
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

# def append_sl_data(data, sl_data, config):
#     # sample indexes
#     data_len = data['act'].shape[0]
#     sl_data_len = sl_data['sl_act'].shape[0]
#     sl_idxs = torch.randint(0, sl_data_len, (data_len,))
#     data['sl_act'] = sl_data['sl_act'][sl_idxs]
#     data['sl_emg_obs'] = sl_data['sl_emg_obs'][sl_idxs]

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
            idx_train = torch.randint(0, ft_len, (pt_train_len,))
            idx_val = torch.randint(0, ft_len, (pt_val_len,))
            data = pt_data
            data["train"]["obs"] = ft_data["obs"][idx_train]
            data["train"]["emg_obs"] = ft_data["emg_obs"][idx_train]
            data["train"]["intended_action"] = ft_data["intended_action"][idx_train]
            data["val"]["obs"] = ft_data["obs"][idx_val]
            data["val"]["emg_obs"] = ft_data["emg_obs"][idx_val]
            data["val"]["intended_action"] = ft_data["intended_action"][idx_val]
        elif pt_len < ft_len:
            idx = torch.randint(0, pt_len, (ft_len,))
            raise NotImplementedError
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
    # sl_data_dict = {
    #     "obs": data["obs"]["observation"],
    #     "emg_obs": data["obs"]["emg_observation"],
    #     "intended_action": data["info"]["intended_action"],
    #     "sl_emg_obs": data["sl_emg_obs"],
    #     "sl_act": data["sl_act"],
    # }
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


def main(kwargs=None):
    if kwargs is not None:
        config = BaseConfig(**kwargs)
    else:
        config = BaseConfig()
    L.seed_everything(config.seed)

    if config.use_wandb:
        tags = ['align_teacher']
        if kwargs is not None:
            tags.append('beta_sweep_2')
        _ = wandb.init(project='lift', tags=tags)

        if kwargs is None:
            # make sure kwargs are not overwritten by wandb
            config = BaseConfig(**wandb.config)
        logger = WandbLogger()
        wandb.config.update(config.model_dump())
    else:
        logger = None
    
    teacher = load_teacher(config)
    data_path = (config.mad_data_path / config.target_person / "training0").as_posix()
    sim = SimulatorFactory.create_class(
        data_path,
        config,
        return_features=True,
        # num_samples_per_group=1,
    )

    env = NpGymEnv(
        "FetchReachDense-v2", 
        cat_obs=True, 
        cat_keys=config.teacher.env_cat_keys,
    )

    # init trainer
    pt_encoder_state_dict = torch.load(config.models_path / "pretrain_mi_encoder.pt")
    pt_critic_state_dict = torch.load(config.models_path / "pretrain_mi_critic.pt")
    
    trainer = MITrainer(config, env, teacher, pretrain=False, supervise=True, activation=nn.ReLU)
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

    torch.save(trainer, config.models_path / 'mi.pt')
    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta_sweep", action="store_true")
    args = parser.parse_args()
    
    if args.beta_sweep:
        import itertools

        beta_2_values = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        seeds = [42, 123, 456, 789]

        combinations = itertools.product(beta_2_values, seeds)

        for combination in combinations:
            beta_2, seed = combination
            kwargs = {"encoder": {"beta_2": beta_2},
                      "seed": seed,}
            main(kwargs)

    else:
        main()
