import pickle
import wandb
import torch
import lightning as L

from torchrl.envs.utils import check_env_specs
from torchrl.data import ReplayBuffer, LazyTensorStorage, SliceSampler
from tensordict import TensorDict

from lift.environments.simulator import SimulatorFactory
from lift.environments.emg_envs import EMGTransform

from lift.teacher import load_teacher
from lift.rl.utils import parallel_env_maker
from lift.rl.cql import CQL

from configs import BaseConfig


def data_to_replay_buffer(data, config):
    batch_size = config.offline_rl.batch_size
    n_samples = data['act'].shape[0]

    rb = ReplayBuffer(
        storage=LazyTensorStorage(config.offline_rl.replay_buffer_size, device=config.offline_rl.device),
        sampler=SliceSampler(num_slices=config.offline_rl.num_slices, traj_key=("collector", "traj_ids"),
                            truncated_key=None, strict_length=False),
                            batch_size=batch_size)
    
    obs = torch.tensor(data['obs']['observation'], dtype=torch.float32)
    next_obs = torch.tensor(data['next_obs']['observation'], dtype=torch.float32)

    emg_obs = torch.tensor(data['obs']['emg'], dtype=torch.float32).squeeze(1)
    next_emg_obs = torch.cat([emg_obs[1:,:], torch.zeros(1, emg_obs.shape[1])], dim=0)

    assert obs.shape[0] == next_obs.shape[0] == emg_obs.shape[0] == next_emg_obs.shape[0]

    for i in range(n_samples // batch_size):
        s_idx = i * batch_size
        e_idx = (i + 1) * batch_size
        td_data = TensorDict(
            {
                "observation": obs[s_idx:e_idx],
                "emg": emg_obs[s_idx:e_idx],
                "action": torch.tensor(data['act'][s_idx:e_idx], dtype=torch.float32),
                # "episode_reward": torch.zeros(batch_size), 
                "is_init": torch.zeros(batch_size, 1).type(torch.bool),
                # "loc": ,
                ("next", "done"): torch.tensor(data['done'][s_idx:e_idx], dtype=torch.float32).unsqueeze(1),
                # ("next", "episode_reward"): torch.zeros(batch_size, 1),
                ("next", "is_init"): torch.zeros(batch_size, 1).type(torch.bool),
                ("next", "observation"): next_obs[s_idx:e_idx],
                ("next", "emg"): next_emg_obs[s_idx:e_idx],
                ("next", "reward"): torch.tensor(data['align_reward'][s_idx:e_idx], dtype=torch.float32).unsqueeze(1),
                ("next", "terminated"): torch.zeros(batch_size, 1).type(torch.bool),
                ("next", "truncated"): torch.zeros(batch_size, 1).type(torch.bool),
            },
            batch_size=[batch_size],
        )
        rb.extend(td_data)

    return rb


def get_align_reward(data, teacher, encoder, kl_approx_method="logp"):
    emg_obs = torch.tensor(data['obs']['emg'], dtype=torch.float32).squeeze(1)
    obs = torch.tensor(data['obs']['observation'], dtype=torch.float32)
    teacher_policy = teacher.model.policy

    z_dist = encoder.get_dist(emg_obs)
    z = z_dist.sample()

    teacher_inputs = TensorDict({
            "observation": obs,
            "action": z,
        })

    if kl_approx_method == "logp":
        log_prior = teacher_policy.log_prob(teacher_inputs)
        ent = -z_dist.log_prob(z)
        kl = log_prior + ent
    # elif kl_approx_method == "abs":
    #     # john schulman kl approximation
    #     log_prior = teacher.log_prob(teacher_inputs)
    #     log_post = z_dist.log_prob(z)
    #     kl = 0.5 * nn.SmoothL1Loss()(log_post, log_prior)
    elif kl_approx_method == "mse":
        with torch.no_grad():
            teacher_dist = teacher_policy.get_dist(teacher_inputs)
            a_teacher = teacher_dist.mode
        kl = -torch.pow(z - a_teacher, 2).mean(dim=1)
    else:
        raise ValueError("kl approximation method much be one of [logp, abs, mse]")

    return kl


def train_offline_rl(data, teacher, sim, encoder, config: BaseConfig):
    # reward and replay buffer
    data['align_reward'] = get_align_reward(data, teacher, encoder, kl_approx_method=config.offline_rl.kl_approx_method)
    replay_buffer = data_to_replay_buffer(data, config)

    # eval env
    eval_env = parallel_env_maker(
        config.teacher.env_name,
        cat_obs=config.teacher.env_cat_obs,
        cat_keys=config.teacher.env_cat_keys,
        max_eps_steps=config.teacher.max_eps_steps,
        device="cpu",
    )
    t_emg = EMGTransform(teacher, sim, in_keys=["observation"], out_keys=["emg"])
    eval_env.append_transform(t_emg)
    # check_env_specs(eval_env)

    cql = CQL(config.offline_rl, replay_buffer, eval_env, encoder=encoder)
    cql.train(use_wandb=config.use_wandb)

    return cql.model.policy


def main():
    config = BaseConfig()
    L.seed_everything(config.seed)
    if config.use_wandb:
        _ = wandb.init(project='lift', tags='offline_rl')
        config = BaseConfig(**wandb.config)
        wandb.config.update(config.model_dump())

    teacher = load_teacher(config)
    data_path = (config.mad_data_path / "Female0"/ "training0").as_posix()
    sim = SimulatorFactory.create_class(
        data_path,
        config,
        return_features=True,
    )

    trainer = torch.load(config.models_path / 'mi.pt')
    encoder = trainer.encoder

    rollout_file = config.rollout_data_path / f"data.pkl"
    with open(rollout_file, "rb") as f:
        data = pickle.load(f)

    policy = train_offline_rl(data, teacher, sim, encoder, config)

    # TODO make sure this is visualizable or transform back to TanhGaussian
    torch.save(policy, config.models_path / 'offline_rl.pt')


if __name__ == "__main__":
    main()
