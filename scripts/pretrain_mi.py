import os
import wandb
import torch
import lightning as L
from pytorch_lightning.loggers import WandbLogger

from configs import BaseConfig
from lift.environments.gym_envs import NpGymEnv
from lift.environments.emg_envs import EMGEnv
from lift.environments.simulator import SimulatorFactory, NonParametricWeightedSimulator
from lift.environments.rollout import rollout
from lift.teacher import load_teacher
from lift.utils import normalize

from lift.datasets import (
    get_samples_per_group,
    weighted_per_person_augmentation,
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
    mean_rwd = data["rwd"].mean()
    std_rwd = data["rwd"].std()
    print(f"encoder reward mean: {mean_rwd:.4f}, std: {std_rwd:.4f}")
    if logger is not None:
        logger.log_metrics({"encoder_reward": mean_rwd})
    return data

def train(emg_features, actions, model, logger, config: BaseConfig):
    sl_data_dict = {
        "emg_obs": emg_features,
        "act": actions,
    }
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
        # sample_features, sample_actions, _, _ = weighted_per_person_augmentation(config)
        
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

    # people_list = [f"Female{i}" for i in range(10)] + [f"Male{i}" for i in range(16)]
    # people_list = [p for p in people_list if not p == config.target_person]

    # mad_windows = None
    # mad_labels = None

    # for p in people_list:
    #     other_list = [o_p for o_p in people_list if not o_p == p]

    mad_windows, mad_labels = load_all_mad_datasets(
        config.mad_base_path.as_posix(),
        num_channels=config.num_channels,
        emg_range=config.emg_range,
        window_size=config.window_size,
        window_overlap=config.window_overlap,
        desired_labels=config.desired_mad_labels,
        skip_person=config.target_person,
        # skip_person=other_list,
        return_tensors=True,
    )
        # """DEBUG: sample only one window for each action"""
        # p_windows, p_labels = get_samples_per_group(p_windows, p_labels, config, num_samples_per_group=1)

        # if mad_windows is None:
        #     mad_windows = p_windows
        #     mad_labels = p_labels
        # else:
        #     mad_windows = torch.cat([mad_windows, p_windows], dim=0)
        #     mad_labels = torch.cat([mad_labels, p_labels], dim=0)


    mad_actions = mad_labels_to_actions(
        mad_labels, recording_strength=config.simulator.recording_strength,
    )

    if config.simulator.interpolation == "weighted":
        mad_features = compute_features(mad_windows, feature_list=['MAV'])
    else:
        mad_features = compute_features(mad_windows)


    if config.pretrain.num_augmentation > 0:
        sample_features, sample_actions = load_augmentation(mad_windows, mad_labels,
                                                            mad_actions, config)

        if config.pretrain.train_subset == "interpolation":
            features = sample_features
            actions = sample_actions
        else:
            features = torch.cat([mad_features, sample_features], dim=0)
            actions = torch.cat([mad_actions, sample_actions], dim=0)
    else:
        features, actions = mad_features, mad_actions


    return features, actions

# def main(beta_1, beta_2, beta_3):
def main():
    config = BaseConfig()
    # config.mi.beta_1 = beta_1
    # config.mi.beta_2 = beta_2
    # config.mi.beta_3 = beta_3
    L.seed_everything(config.seed)
    if config.use_wandb:
        _ = wandb.init(project='lift', tags=['align_teacher', 'beta_sweep_generalization'])
        # config = BaseConfig(**wandb.config)
        logger = WandbLogger()
        wandb.config.update(config.model_dump())
    else:
        logger = None

    emg_features, actions, = load_data(config)
    emg_mu = emg_features.mean(0)
    emg_sd = emg_features.std(0)
    emg_features_norm = normalize(emg_features, emg_mu, emg_sd)
    
    # validation setup
    teacher = load_teacher(config)
    data_path = (config.mad_data_path / config.target_person / "training0").as_posix()
    # sim = SimulatorFactory.create_class(
    #     data_path,
    #     config,
    #     return_features=True,
    # )
    """DEBUG: only one sample per action from one person"""
    sim = NonParametricWeightedSimulator(    
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

    trainer = MITrainer(config, env, supervise=True)
    
    # test once before train
    validate(env, teacher, sim, trainer.encoder, emg_mu, emg_sd, logger)

    train(emg_features_norm, actions, trainer, logger, config)

    # test once after train
    validate(env, teacher, sim, trainer.encoder, emg_mu, emg_sd, logger)

    torch.save(trainer.encoder.state_dict(), config.models_path / "pretrain_mi_encoder.pt")
    torch.save(trainer.critic.state_dict(), config.models_path / "pretrain_mi_critic.pt")
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # import numpy as np
    # from itertools import product

    # betas_space = np.linspace(0.1,1,3)

    # for b1, b2, b3 in product(betas_space, betas_space, betas_space):
    #     main(b1, b2, b3)

    main()