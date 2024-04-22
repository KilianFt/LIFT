import pickle
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
from lift.datasets import get_dataloaders
from lift.controllers import MITrainer, EMGAgent


def maybe_rollout(env: NpGymEnv, teacher, config: BaseConfig, use_saved=True):
    rollout_file = config.rollout_data_path / f"data.pkl"
    if use_saved and rollout_file.exists():
        print(f"\nload rollout data from file: {rollout_file}")
        with open(rollout_file, "rb") as f:
            data = pickle.load(f)
    else:
        print("collecting training data from teacher")
        data = rollout(
            env,
            teacher,
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
        logger.log_metrics({"encoder reward": mean_rwd})
    return data

def train(data, sim: WindowSimulator, model, logger, config: BaseConfig):
    emg_features = sim(data["act"])

    sl_data_dict = {
        "obs": data["obs"]["observation"],
        "emg_obs": emg_features,
        "act": data["act"],
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
    
    teacher = load_teacher(config)
    sim = WindowSimulator(
        action_size=config.action_size, 
        num_bursts=config.simulator.n_bursts, 
        num_channels=config.n_channels,
        window_size=config.window_size, 
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
    
    # collect teacher data
    # TODO use teacher env here
    data = maybe_rollout(env, teacher, config, use_saved=False)
    mean_rwd = data["rwd"].mean()
    print("teacher reward", mean_rwd)
    if logger is not None:
        logger.log_metrics({"teacher reward": mean_rwd})
    
    # load bc encoder
    bc_trainer = torch.load(config.models_path / "bc.pt")

    # init trainer
    trainer = MITrainer(config, env, teacher)
    trainer.encoder.load_state_dict(bc_trainer.encoder.state_dict())

    # test once before train
    validate(env, teacher, sim, trainer.encoder, logger)

    train(data, sim, trainer, logger, config)
    
    # test once after train
    validate(env, teacher, sim, trainer.encoder, logger)

    torch.save(trainer, config.models_path / 'encoder.pt')


if __name__ == "__main__":
    main()
