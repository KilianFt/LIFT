"""
Iterative MI training
"""
import argparse
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

def validate_data(data, logger):
    assert data["info"]["teacher_action"].shape == data["act"].shape
    mae = np.abs(data["info"]["teacher_action"] - data["act"]).mean()
    mean_rwd = data["rwd"].mean()
    std_rwd = data["rwd"].std()
    print(f"encoder reward mean: {mean_rwd:.4f}, std: {std_rwd:.4f}, mae: {mae:.4f}")
    if logger is not None:
        logger.log_metrics({"encoder_reward": mean_rwd, "encoder_mae": mae})

    return mean_rwd, std_rwd, mae

def validate(env, teacher, sim, encoder, logger):
    emg_env = EMGEnv(env, teacher, sim)
    agent = EMGAgent(encoder)
    data = rollout(
        emg_env, 
        agent, 
        n_steps=1000, 
        terminate_on_done=False,
        reset_on_done=True,
    )
    mean_rwd, std_rwd, mae = validate_data(data, logger)
    return mean_rwd, std_rwd, mae


def train(data, model, logger, config: BaseConfig):
    sl_data_dict = {
        "obs": data["obs"]["observation"],
        "emg_obs": data["obs"]["emg_observation"],
        "act": data["info"]["teacher_action"], # privileged information for validation
    }
    train_dataloader, val_dataloader = get_dataloaders(
        data_dict=sl_data_dict,
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

def main(kwargs=None):
    if kwargs is not None:
        config = BaseConfig(**kwargs)
    else:
        config = BaseConfig()

    L.seed_everything(config.seed)

    if config.use_wandb:
        tags = ['align_teacher']
        if kwargs is not None:
            tags.append(kwargs['tag'])
        _ = wandb.init(project='lift', tags=tags)
        # config = BaseConfig(**wandb.config) this will overwrite the noise_range
        logger = WandbLogger()
        wandb.config.update(config.model_dump())
    else:
        logger = None

    env = NpGymEnv(
        "FetchReachDense-v2", 
        cat_obs=True, 
        cat_keys=config.teacher.env_cat_keys,
    )
    sim = SimulatorFactory.create_class(
        (config.mad_data_path / "Female0"/ "training0").as_posix(),
        config,
        return_features=True,
    )
    
    # init user with known noise and alpha for controlled experiment
    user = load_teacher(config, meta=True)
    user = ConditionedTeacher(
        user, 
        noise_range=config.noise_range,
        noise_slope_range=config.noise_slope_range,
        alpha_range=config.alpha_range,
        alpha_apply_range=config.alpha_apply_range,
    )
    user.reset()

    # load models
    teacher = load_teacher(config)
    bc_trainer = torch.load(config.models_path / "bc.pt")
    
    trainer = MITrainer(config, env, teacher)
    trainer.encoder.load_state_dict(bc_trainer.encoder.state_dict())
    
    # interactive training loop
    num_sessions = 10

    dataset = {}
    results = {
        'noise': config.noise_range,
        'noise_drift': config.noise_drift,
        'alpha': config.alpha_range,
        'alpha_drift': config.alpha_drift,
        'wandb_id': wandb.run.id if config.use_wandb else 'local',
        'rwd': [],
        'rwd_std': [],
        'mae': [],
    }

    for i in range(num_sessions):
        # collect user data
        emg_env = EMGEnv(env, user, sim)
        emg_policy = EMGAgent(trainer.encoder)

        data = maybe_rollout(emg_env, emg_policy, config, use_saved=False)
        rwd, std_rwd, mae = validate_data(data, logger)
        results['rwd'].append(rwd)
        results['rwd_std'].append(std_rwd)
        results['mae'].append(mae)

        append_dataset(dataset, data)

        # we might want to aggregate data from multiple sessions
        dataset = dataset if config.mi.aggregate_data else data
        train(dataset, trainer, logger, config)

        # update user meta vars
        if config.noise_drift is not None:
            user_meta_vars = user.get_meta_vars()
            user_meta_vars['noise'] = apply_gaussian_drift(
                user_meta_vars['noise'],
                config.noise_drift[0],
                config.noise_drift[1],
                config.noise_range,
            )
            user.set_meta_vars(user_meta_vars)

    # test once at the end
    rwd, std_rwd, mae = validate(env, teacher, sim, trainer.encoder, logger)
    results['rwd'].append(rwd)
    results['rwd_std'].append(std_rwd)
    results['mae'].append(mae)

    torch.save(trainer, config.models_path / 'mi_iter.pt')

    if config.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    
    if args.sweep:
        # seeds = [42, 123, 456, 789]
        for constant_alpha in np.linspace(1.0, 3.0, 5):
            for constant_noise in np.linspace(0.001, 1.0, 5):
                kwargs = {"alpha_range": [constant_alpha]*2,
                          "noise_range": [constant_noise]*2,
                          "noise_slope_range": [constant_noise]*2,
                          "tag": "mi_sweep",}
                main(kwargs)
    elif args.baseline:
        kwargs = {
            "encoder": {"beta_1": 0.,
                        "kl_approx_method": "mse"},
            "tag": "baseline",
        }
        main(kwargs)
    else:
        main()
