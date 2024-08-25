import os
import pickle
import wandb
import torch
import torch.nn as nn
import numpy as np
import lightning as L
from pytorch_lightning.loggers import WandbLogger

from configs import BaseConfig
from lift.environments.gym_envs import NpGymEnv
from lift.environments.emg_envs import EMGEnv
from lift.environments.simulator import SimulatorFactory
from lift.environments.rollout import rollout
from lift.teacher import load_teacher
from lift.utils import normalize

from lift.datasets import (
    get_samples_per_group,
    weighted_augmentation,
    load_all_mad_datasets, 
    mad_groupby_labels,
    mad_labels_to_actions,
    mad_augmentation, 
    compute_features, 
    get_dataloaders, 
)
from lift.controllers import MITrainer, EMGAgent


def validate(env, teacher, sim, encoder, mu, sd, logger):
    encoder.eval()
    emg_env = EMGEnv(env, teacher, sim)
    agent = EMGAgent(encoder, mu, sd)
    data = rollout(
        emg_env, 
        agent, 
        n_steps=1000,
        terminate_on_done=False,
        reset_on_done=True,
    )
    assert data["info"]["intended_action"].shape == data["act"].shape
    mae = np.abs(data["info"]["intended_action"] - data["act"]).mean()
    sum_rwd = data["rwd"].sum()
    mean_rwd = data["rwd"].mean()
    std_rwd = data["rwd"].std()
    n_episodes = data["done"].sum()
    print(f"encoder reward mean: {mean_rwd:.4f}, std: {std_rwd:.4f}, mae: {mae:.4f}")
    if logger is not None:
        logger.log_metrics({"encoder_reward": mean_rwd,
                            "encoder_reward_sum": sum_rwd,
                            "n_episodes": n_episodes,
                            "encoder_mae": mae})
    return data

def train(sl_data_dict, model, logger, config: BaseConfig):
    train_dataloader, val_dataloader = get_dataloaders(
        data_dict=sl_data_dict,
        train_ratio=config.pretrain.train_ratio,
        batch_size=config.pretrain.batch_size,
        num_workers=config.num_workers,
    )

    trainer = L.Trainer(
        max_epochs=config.pretrain.epochs, 
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

def load_fake_data(config):
    # load data
    data = torch.load(os.path.join(config.data_path, "fake_mad_data.pt"))
    num_samples = len(data["emg"][0])
    emg_dof = torch.cat(data["emg"], dim=0)
    actions_dof = torch.cat([a.view(1, -1).repeat_interleave(num_samples, 0) for a in data["action"]], dim=0)
    
    # data augmentation
    emg_aug, actions_aug = mad_augmentation(data["emg"], data["action"], config.pretrain.num_augmentation)
    emg = torch.cat([emg_dof, emg_aug], dim=0)
    actions = torch.cat([actions_dof, actions_aug], dim=0)

    emg_features = compute_features(emg)

    return emg_features, actions

def load_augmentation(mad_windows, mad_labels, mad_actions, config):
    if config.simulator.interpolation == "weighted":        
        sample_features, sample_actions = weighted_augmentation(mad_windows, mad_actions, config)
    elif config.simulator.interpolation == "random":
        window_list, label_list = mad_groupby_labels(mad_windows, mad_labels)
        actions_list = mad_labels_to_actions(
            label_list, recording_strength=config.simulator.recording_strength,
        )
        sample_windows, sample_actions = mad_augmentation(
            window_list, 
            actions_list, 
            config.pretrain.num_augmentation,
            augmentation_distribution=config.pretrain.augmentation_distribution,
            reduction=config.simulator.reduction,
        )
        sample_features = compute_features(sample_windows)
    else:
        raise NotImplementedError(f"Interpolation {config.simulator.interpolation} not found")
    
    return sample_features, sample_actions

def load_data(config: BaseConfig, load_fake=False):
    if load_fake:
        return load_fake_data(config)

    all_people_list = [f"Female{i}" for i in range(10)] + [f"Male{i}" for i in range(16)]
    # people_list = [p for p in all_people_list if not p == config.target_person]

    train_features = None
    train_actions = None
    val_features = None
    val_actions = None

    for p in all_people_list:
        other_list = [o_p for o_p in all_people_list if not o_p == p]

        p_windows, p_labels, _ = load_all_mad_datasets(
            config.mad_base_path.as_posix(),
            num_channels=config.num_channels,
            emg_range=config.emg_range,
            window_size=config.window_size,
            window_overlap=config.window_overlap,
            desired_labels=config.desired_mad_labels,
            skip_person=other_list,
            return_tensors=True,
            verbose=False,
            cutoff_n_outer_samples=config.cutoff_n_outer_samples,
        )

        p_actions = mad_labels_to_actions(
            p_labels, recording_strength=config.simulator.recording_strength,
        )

        if config.simulator.interpolation == "weighted":
            p_features = compute_features(p_windows, feature_list=['MAV'])
        else:
            p_features = compute_features(p_windows)

        if config.pretrain.num_augmentation > 0:
            # sample only one window for each action in augementation
            # this prevents augmentation to pick samples from the same action
            # p_windows_aug, p_labels_aug = get_samples_per_group(p_windows, p_labels, 1)
            # p_actions_aug = mad_labels_to_actions(
            #     p_labels_aug, recording_strength=config.simulator.recording_strength,
            # )
            # sample_features, sample_actions = load_augmentation(p_windows_aug, p_labels_aug,
            #                                                     p_actions_aug, config)
            sample_features, sample_actions = load_augmentation(p_windows, p_labels, p_actions, config)

            if config.pretrain.train_subset == "interpolation":
                features = sample_features
                actions = sample_actions
            else:
                features = torch.cat([p_features, sample_features], dim=0)
                actions = torch.cat([p_actions, sample_actions], dim=0)
        else:
            features, actions = p_features, p_actions

        # hold out n people for validation, including target_person
        if p in [config.target_person] + config.val_people:
            if val_features is None:
                val_features = features
                val_actions = actions
            else:
                val_features = torch.cat([val_features, features], dim=0)
                val_actions = torch.cat([val_actions, actions], dim=0)
        else:
            if train_features is None:
                train_features = features
                train_actions = actions
            else:
                train_features = torch.cat([train_features, features], dim=0)
                train_actions = torch.cat([train_actions, actions], dim=0)

    print(f"train size: {train_features.shape}, val size: {val_features.shape}")

    # normalize data
    emg_mu = train_features.mean(0)
    emg_sd = train_features.std(0)
    train_features_norm = normalize(train_features, emg_mu, emg_sd)
    val_features_norm = normalize(val_features, emg_mu, emg_sd)

    pt_data_dict = {
        "train": {
            "pt_emg_obs": train_features_norm,
            "pt_act": train_actions,
            },
        "val": {
            "pt_emg_obs": val_features_norm,
            "pt_act": val_actions,
        },
        "mu": emg_mu,
        "sd": emg_sd,
    }
    return pt_data_dict


def main():
    config = BaseConfig()
    if config.use_wandb:
        _ = wandb.init(project='lift',
                       tags=['pretrain'])
        config = BaseConfig(**wandb.config)
        logger = WandbLogger()
        wandb.config.update(config.model_dump())
    else:
        logger = None

    L.seed_everything(config.seed)

    pt_data_dict = load_data(config)
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

    trainer = MITrainer(config, env, pretrain=True, supervise=True)
    
    # test once before train
    validate(env, teacher, sim, trainer.encoder, pt_data_dict["mu"], pt_data_dict["sd"], logger)

    train(pt_data_dict, trainer, logger, config)

    # test once after train
    validate(env, teacher, sim, trainer.encoder, pt_data_dict["mu"], pt_data_dict["sd"], logger)

    # save dataset and pretrained model
    with open(os.path.join(config.data_path, "pt_dataset.pkl"), 'wb') as f:
        pickle.dump(pt_data_dict, f)

    torch.save(trainer.encoder.state_dict(), config.models_path / "pretrain_mi_encoder.pt")
    torch.save(trainer.critic.state_dict(), config.models_path / "pretrain_mi_critic.pt")
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()