import os
import wandb
import torch
import lightning as L
from pytorch_lightning.loggers import WandbLogger

from configs import BaseConfig
from lift.environments.gym_envs import NpGymEnv
from lift.environments.emg_envs import EMGEnv
from lift.environments.simulator import WindowSimulator
from lift.environments.rollout import rollout
from lift.teacher import load_teacher

from lift.datasets import mad_augmentation, compute_features, get_dataloaders, get_mad_windows_dataset, get_mad_lists
from lift.controllers import BCTrainer, EMGAgent
from lift.utils import mad_labels_to_actions


def validate(env, teacher, sim, encoder, logger):
    emg_env = EMGEnv(env, teacher, sim)
    agent = EMGAgent(encoder)
    data = rollout(
        emg_env, 
        agent, 
        n_steps=5000,
        terminate_on_done=False,
        reset_on_done=True,
        sample_mean=True,
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

def load_data(config, load_fake=False):
    if load_fake:
        return load_fake_data(config)

    mad_emg, mad_labels = get_mad_windows_dataset(config.mad_base_path.as_posix(),
                                                  config.window_size,
                                                  config.pretrain.window_increment,
                                                  desired_labels=[0, 1, 2, 3, 4, 5, 6],
                                                  return_tensors=True,
                                                  skip_person='Female0')
    
    windows_list, label_list = get_mad_lists(mad_emg, mad_labels)

    mad_features = compute_features(mad_emg)
    mad_actions = mad_labels_to_actions(mad_labels, recording_strength=config.simulator.recording_strength)

    if config.pretrain.num_augmentation > 0:
        actions_list = mad_labels_to_actions(label_list, recording_strength=config.simulator.recording_strength)
        sample_emg, sample_actions = mad_augmentation(windows_list, actions_list, config.pretrain.num_augmentation,
                                                      augmentation_distribution=config.pretrain.augmentation_distribution)
        sample_features = compute_features(sample_emg)

        features = torch.cat([mad_features, sample_features], dim=0)
        actions = torch.cat([mad_actions, sample_actions], dim=0)
    else:
        features, actions = mad_features, mad_actions

    return features, actions

def main():
    config = BaseConfig()
    L.seed_everything(config.seed)
    if config.use_wandb:
        _ = wandb.init(project='lift', tags='align_teacher')
        config = BaseConfig(**wandb.config)
        logger = WandbLogger()
        wandb.config.update(config.model_dump())
    else:
        logger = None

    emg_features, actions = load_data(config)
    
    # validation setup
    teacher = load_teacher(config)
    sim = WindowSimulator(
        action_size=config.action_size, 
        num_bursts=config.simulator.n_bursts, 
        num_channels=config.n_channels,
        window_size=config.window_size,
        recording_strength=config.simulator.recording_strength,
        return_features=True,
    )
    sim.fit_params_to_mad_sample(
        (config.mad_data_path / "Female0"/ "training0").as_posix()
    )
    env = NpGymEnv(
        "FetchReachDense-v2", 
        cat_obs=True, 
        cat_keys=config.teacher.env_cat_keys,
    )

    trainer = BCTrainer(config, env)
    
    # test once before train
    validate(env, teacher, sim, trainer.encoder, logger)

    train(emg_features, actions, trainer, logger, config)

    # test once after train
    validate(env, teacher, sim, trainer.encoder, logger)

    torch.save(trainer, config.models_path / "bc.pt")

if __name__ == "__main__":
    main()