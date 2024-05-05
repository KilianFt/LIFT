from pathlib import Path, PosixPath
from pydantic import BaseModel
from typing import List

ROOT_PATH = Path(__file__).resolve().parents[0]

class TeacherConfig(BaseModel):
    # env
    env_name: str = "FetchReachDense-v2"
    env_cat_obs: bool = True # whether to concat observations
    env_cat_keys: list | None = None # auto sort obs keys
    max_eps_steps: int = 100
    seed: int = 0

    # collector
    total_frames: int = 150_000
    init_random_frames: int = 5000
    frames_per_batch: int = 1000
    init_env_steps: int = 1000
    env_per_collector: int = 1
    reset_at_each_iter: bool = False
    
    # replay
    replay_buffer_size: int = 1000000
    prioritize: int = 0
    scratch_dir: None = None

    # optim 
    utd_ratio: float = 1.0
    gamma: float = 0.99
    loss_function: str = "l2"
    lr: float = 3.0e-4
    weight_decay: float = 0.0
    batch_size: int = 256
    target_update_polyak: float = 0.995
    alpha_init: float = 1.0
    adam_eps: float = 1.0e-8
    
    # nets
    hidden_sizes: list = [256, 256]
    activation: str = "relu"
    default_policy_scale: float = 1.0
    scale_lb: float = 0.1
    device: str = "cpu"

    # eval
    eval_iter: int = 5000
    

class OfflineRLConfig(BaseModel):
    # replay
    replay_buffer_size: int = 1000000
    batch_size: int = 64
    num_slices: int = 8

    # optim 
    utd_ratio: float = 1.0
    gamma: float = 0.99
    loss_function: str = "l2"
    lr: float = 3.0e-4
    weight_decay: float = 0.0
    target_update_polyak: float = 0.995
    alpha_init: float = 1.0
    adam_eps: float = 1.0e-8
    
    # nets
    hidden_sizes: list = [256, 256]
    activation: str = "relu"
    default_policy_scale: float = 1.0
    scale_lb: float = 0.1
    device: str = "cpu"

    # train
    num_updates: int = 10_000

    # eval
    eval_iter: int = 1000
    eval_rollout_steps: int = 100


class EncoderConfig(BaseModel):
    h_dim: int = 128
    tau: float = 0.5
    beta_1: float = 1. # mi weight, use 0.5 for mse
    beta_2: float = 1. # kl weight
    kl_approx_method: str = "logp" # choices=[logp, abs, mse]
    hidden_size: int = 256
    n_layers: int = 4
    dropout: float = 0.3


class SimulatorConfig(BaseModel):
    n_bursts: int = 1
    recording_strength: float = 0.8


class PretrainConfig(BaseModel):
    epochs: int = 90
    num_augmentation: int = 10_000
    augmentation_distribution: str = "uniform" # choices=["uniform", "normal"]
    train_ratio: float = 0.8
    batch_size: int = 128
    lr: float = 1.5e-5


class MIConfig(BaseModel):
    train_ratio: float = 0.8
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-4
    n_steps_rollout: int = 10_000
    random_pertube_prob: float = 0.0
    action_noise: float = 0.0


class BaseConfig(BaseModel):
    # path config
    root_path: PosixPath = ROOT_PATH
    data_path: PosixPath = ROOT_PATH / "datasets"
    mad_base_path: PosixPath = ROOT_PATH / "datasets" / "MyoArmbandDataset"
    mad_data_path: PosixPath = mad_base_path / "PreTrainingDataset"
    models_path: PosixPath = ROOT_PATH / "models"
    rollout_data_path: PosixPath = ROOT_PATH / "datasets" / "rollouts"
    
    # wandb
    use_wandb: bool = True
    project_name: str = "lift"
    wandb_mode: str = "online"

    # data
    n_channels: int = 8
    window_size: int = 200
    window_overlap: int = 50
    emg_range: list = [-128., 127.]
    desired_mad_labels: list = [0, 1, 2, 3, 4, 5, 6]

    seed: int = 100#42
    num_workers: int = 7
    teacher_train_timesteps: int = 150_000
    action_size: int = 3 # could be read from env
    feature_size: int = 32 # could be read from env

    gradient_clip_val: float = 2.
    checkpoint_frequency: int = 1
    save_top_k: int = -1 # set to -1 to save all checkpoints
    
    teacher: TeacherConfig = TeacherConfig()
    encoder: EncoderConfig = EncoderConfig()
    simulator: SimulatorConfig = SimulatorConfig()
    pretrain: PretrainConfig = PretrainConfig()
    mi: MIConfig = MIConfig()
    offline_rl: OfflineRLConfig = OfflineRLConfig()
