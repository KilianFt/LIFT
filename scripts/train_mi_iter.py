"""
Iterative MI training
"""
import pickle
import wandb
import torch
import lightning as L
from pytorch_lightning.loggers import WandbLogger

from configs import BaseConfig
from lift.environments.gym_envs import NpGymEnv
from lift.environments.emg_envs import EMGEnv
from lift.environments.simulator import SimulatorFactory
from lift.environments.rollout import rollout

from lift.teacher import ConditionedTeacher, load_teacher, apply_gaussian_drift
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
    mean_rwd = data["rwd"].mean()
    std_rwd = data["rwd"].std()
    print(f"encoder reward mean: {mean_rwd:.4f}, std: {std_rwd:.4f}")
    if logger is not None:
        logger.log_metrics({"encoder_reward": mean_rwd})
    return data

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
    
    # init user 
    user = load_teacher(config, meta=True)
    """TODO: init user with known noise and alpha for controlled experiment"""
    user = ConditionedTeacher(
        user, 
        noise_range=config.noise_range, 
        alpha_range=config.alpha_range,
    )
    user.reset()

    # load models
    teacher = load_teacher(config)
    bc_trainer = torch.load(config.models_path / "bc.pt")
    
    trainer = MITrainer(config, env, teacher)
    trainer.encoder.load_state_dict(bc_trainer.encoder.state_dict())
    
    """TODO: record mae and reward stats in between sessions"""
    # interactive training loop
    num_sessions = 10
    for i in range(num_sessions):
        # collect user data
        emg_env = EMGEnv(env, user, sim)
        emg_policy = EMGAgent(trainer.encoder)

        """TODO: reduce rollout steps here"""
        data = maybe_rollout(emg_env, emg_policy, config, use_saved=False)
        mean_rwd = data["rwd"].mean()
        print("encoder_reward", mean_rwd)
        if logger is not None:
            logger.log_metrics({"encoder_reward": mean_rwd})

        train(data, trainer, logger, config)

        # update user meta vars
        user.meta_vars[0] = apply_gaussian_drift(
            user.meta_vars[0], 
            config.noise_drift[0], 
            config.noise_drift[1], 
            config.noise_range,
        )
    
    # test once at the end
    validate(env, teacher, sim, trainer.encoder, logger)

    torch.save(trainer, config.models_path / 'mi_iter.pt')

if __name__ == "__main__":
    main()
