import pickle
import wandb
import torch
import numpy as np
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from configs import BaseConfig
from lift.environments.gym_envs import NpGymEnv
from lift.environments.emg_envs import EMGEnv
from lift.environments.user_envs import UserEnv
from lift.environments.simulator import WindowSimulator
from lift.environments.rollout import rollout

from lift.teacher import load_teacher, ConditionedTeacher
from lift.datasets import get_dataloaders
from lift.controllers import MITrainer, EMGAgent


def maybe_rollout(env: UserEnv, teacher, config: BaseConfig, use_saved=True):
    rollout_file = config.rollout_data_path / f"seq_mi_rollout_data.pkl"
    if use_saved and rollout_file.exists():
        print(f"\nload rollout data from file: {rollout_file}")
        with open(rollout_file, "rb") as f:
            data = pickle.load(f)
    else:
        print("collecting training data from teacher")
        data = rollout(
            env,
            teacher,
            # n_steps=config.mi.n_steps_rollout,
            n_steps=1000,
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

def format_data(data: dict) -> list[dict]:
    """Format rollout data into sequences"""
    obs_keys = data["obs"].keys()
    eps_ids = data["done"].cumsum()
    eps_ids = np.insert(eps_ids, 0, 0)[:-1]
    unique_eps_ids = np.unique(eps_ids)
    
    seq_data = []
    for eps_id in unique_eps_ids:
        idx = eps_ids == eps_id
        obs = {k: data["obs"][k][idx] for k in obs_keys}
        act = data["act"][idx]
        rwd = data["rwd"][idx]
        next_obs = {k: data["next_obs"][k][idx] for k in obs_keys}
        done = data["done"][idx]
        seq_data.append({
            "obs": obs,
            "act": act,
            "rwd": rwd,
            "next_obs": next_obs,
            "done": done
        })
    return seq_data


class EMGSeqDataset(Dataset):
    """Sequence learning dataset"""
    def __init__(self, data: list[dict]):
        assert isinstance(data[0], dict), "data elements must be dictionaries"
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # concat obs, emg_ob, emg_act
        data = self.data[idx]
        obs = np.concatenate(list(data["obs"].values()), axis=-1)
        act = data["act"]
        out = np.concatenate([obs, act], axis=-1)
        out = torch.from_numpy(out).to(torch.float32)
        return out


def collate_fn(batch):
    pad_batch = pad_sequence(batch)
    mask = pad_sequence([torch.ones(len(b)) for b in batch])
    return pad_batch, mask

def train(data, sim: WindowSimulator, model, logger, config: BaseConfig):
    # TODO this should be emg from user env
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
    
    teacher = load_teacher(config, meta=False)
    conditioned_teacher = ConditionedTeacher(
        teacher,
        config.noise_range,
        config.alpha_range,
    )
    sim = WindowSimulator(
        config,
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

    # load bc encoder
    bc_trainer = torch.load(config.models_path / "bc.pt")

    # init trainer
    trainer = MITrainer(config, env, teacher)
    trainer.encoder.load_state_dict(bc_trainer.encoder.state_dict())

    emg_policy = EMGAgent(trainer.encoder)
    emg_env = EMGEnv(env, conditioned_teacher, sim)

    # collect user data
    data = maybe_rollout(emg_env, emg_policy, config, use_saved=False)
    mean_rwd = data["rwd"].mean()
    print("encoder_reward", mean_rwd)
    seq_data = format_data(data)
    dataset = EMGSeqDataset(seq_data)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.mi.batch_size, 
        collate_fn=collate_fn,
        # num_workers=num_workers, 
        # persistent_workers=True, 
        # shuffle=True,
    )
    batch, mask = next(iter(dataloader))

    # if logger is not None:
    #     logger.log_metrics({"encoder_reward": mean_rwd})

    # # test once before train
    # validate(env, teacher, sim, trainer.encoder, logger)

    # train(data, sim, trainer, logger, config)
    
    # # test once after train
    # validate(env, teacher, sim, trainer.encoder, logger)

    # torch.save(trainer, config.models_path / 'mi.pt')


if __name__ == "__main__":
    main()
