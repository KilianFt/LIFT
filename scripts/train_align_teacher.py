import pickle
import wandb
import torch
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from torchrl.envs.utils import check_env_specs
from torchrl.data import ReplayBuffer, LazyTensorStorage, SliceSampler
from tensordict import TensorDict

from configs import BaseConfig
from lift.environments.gym_envs import NpGymEnv
from lift.environments.emg_envs import EMGEnv, EMGTransform
from lift.environments.simulator import WindowSimulator
from lift.environments.rollout import rollout

from lift.teacher import load_teacher
from lift.datasets import get_dataloaders
from lift.controllers import MITrainer, EMGAgent

from lift.rl.utils import parallel_env_maker, apply_env_transforms, gym_env_maker
from lift.rl.cql import CQL


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
            n_steps=config.n_steps_rollout,
            terminate_on_done=False,
            reset_on_done=True,
            random_pertube_prob=config.random_pertube_prob,
            action_noise=config.action_noise,
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
        train_ratio=config.train_ratio,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    trainer = L.Trainer(
        max_epochs=config.epochs, 
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


def data_to_replay_buffer(data, config):
    batch_size = config.offline_rl.batch_size
    n_samples = data['act'].shape[0]

    rb = ReplayBuffer(
        storage=LazyTensorStorage(config.offline_rl.replay_buffer_size),
        sampler=SliceSampler(num_slices=config.offline_rl.num_slices, traj_key=("collector", "traj_ids"),
                            truncated_key=None, strict_length=False),
                            batch_size=batch_size)
    
    # TODO use EMG features
    obs = torch.tensor(data['obs']['observation'], dtype=torch.float32)
    next_obs = torch.tensor(data['next_obs']['observation'], dtype=torch.float32)
    # emg_obs = sim()

    for i in range(n_samples // batch_size):
        s_idx = i * batch_size
        e_idx = (i + 1) * batch_size
        td_data = TensorDict(
            {
                "observation": obs[s_idx:e_idx],
                "emg": data['obs']['emg'][s_idx:e_idx],
                "action": torch.tensor(data['act'][s_idx:e_idx], dtype=torch.float32),
                # "episode_reward": torch.zeros(batch_size), 
                "is_init": torch.zeros(batch_size, 1).type(torch.bool),
                # "loc": ,
                ("next", "done"): torch.tensor(data['done'][s_idx:e_idx], dtype=torch.float32).unsqueeze(1),
                # ("next", "episode_reward"): torch.zeros(batch_size, 1),
                ("next", "is_init"): torch.zeros(batch_size, 1).type(torch.bool),
                ("next", "observation"): next_obs[s_idx:e_idx],
                ("next", "emg"): data['next_obs']['emg'][s_idx:e_idx],
                ("next", "reward"): torch.tensor(data['align_reward'][s_idx:e_idx], dtype=torch.float32).unsqueeze(1),
                ("next", "terminated"): torch.zeros(batch_size, 1).type(torch.bool),
                ("next", "truncated"): torch.zeros(batch_size, 1).type(torch.bool),
            },
            batch_size=[batch_size],
        )
        rb.extend(td_data)

    return rb


def get_align_reward(data, teacher_action, encoder):

    encoder_action = encoder.sample(data['obs']['emg'])

    # TODO add KL divergence
    align_reward = (teacher_action[:,:3] - encoder_action).pow(2).mean(dim=1)
    return align_reward


def train_offline_rl(data, teacher, sim, encoder, config: BaseConfig):
    # get emg obs
    obs = torch.tensor(data['obs']['observation'], dtype=torch.float32)
    next_obs = torch.tensor(data['next_obs']['observation'], dtype=torch.float32)
    with torch.no_grad():
        _, _, teacher_action = teacher.model.policy(obs)
        _, _, teacher_action_next = teacher.model.policy(next_obs)
    data['obs']['emg'] = sim(teacher_action)
    data['next_obs']['emg'] = sim(teacher_action_next)

    # reward and replay buffer
    data['align_reward'] = get_align_reward(data, teacher_action, encoder)
    replay_buffer = data_to_replay_buffer(data, config)

    # eval env
    # eval_env = parallel_env_maker(
    #     config.teacher.env_name,
    #     cat_obs=config.teacher.env_cat_obs,
    #     cat_keys=config.teacher.env_cat_keys,
    #     max_eps_steps=config.teacher.max_eps_steps,
    #     device="cpu",
    # )
    eval_env = apply_env_transforms(gym_env_maker(config.teacher.env_name))
    t_emg = EMGTransform(teacher, sim, in_keys=["observation"], out_keys=["emg"])
    eval_env.append_transform(t_emg)
    check_env_specs(eval_env)

    cql = CQL(config.offline_rl, replay_buffer, eval_env, encoder=encoder)
    cql.train(use_wandb=config.use_wandb)

    return cql.model.policy


def main():
    config = BaseConfig()
    L.seed_everything(config.seed)
    if config.use_wandb:
        _ = wandb.init(project='lift', tags='align_teacher')
        logger = WandbLogger()
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

    train_offline_rl(data, teacher, sim, trainer.encoder, config)

    validate(env, teacher, sim, trainer.encoder, logger)

    torch.save(trainer, config.models_path / 'encoder_offline_rl.pt')



if __name__ == "__main__":
    main()
