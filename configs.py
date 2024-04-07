from pathlib import Path
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
    

class EncoderConfig(BaseModel):
    h_dim: int = 128
    tau: float = 0.5
    beta_1: float = 1. # mi weight, use 0.5 for mse
    beta_2: float = 1. # kl weight
    kl_approx_method: str = "logp" # choices=[logp, abs, mse]
    hidden_size: int = 256
    n_layers: int = 8
    dropout: float = 0.1


class SimulatorConfig(BaseModel):
    n_bursts: int = 1
    recording_strength: float = 0.8


"""TODO: make different configs for bc and mi training"""
class BaseConfig(BaseModel):
    # path config
    root_path: str = ROOT_PATH
    data_path: str = ROOT_PATH / "datasets"
    mad_base_path: str = ROOT_PATH / "datasets" / "MyoArmbandDataset"
    mad_data_path: str = mad_base_path / "PreTrainingDataset"
    models_path: str = ROOT_PATH / "models"
    rollout_data_path: str = ROOT_PATH / "datasets" / "rollouts"
    
    # wandb
    use_wandb: bool = True
    project_name: str = "lift"
    wandb_mode: str = "online"

    seed: int = 42
    num_workers: int = 7
    teacher_train_timesteps: int = 150_000
    action_size: int = 3 # could be read from env
    feature_size: int = 32 # could be read from env
    n_channels: int = 8
    window_size: int = 200
    n_steps_rollout: int = 10_000
    random_pertube_prob: int = 0.5
    action_noise: float = 0.3
    num_augmentation: int = 10000 # for pretraining
    window_increment: int = 100 # for pretraining
    
    # supervised learning config
    # dropout: float = .1
    train_ratio: float = 0.8
    batch_size: int = 128
    num_workers: int = 7
    epochs: int = 50
    lr: float = 1e-4
    gradient_clip_val: float = 2.
    noise: float = 0.0
    checkpoint_frequency: int = 1
    save_top_k: int = -1 # set to -1 to save all checkpoints
    
    teacher: TeacherConfig = TeacherConfig()
    encoder: EncoderConfig = EncoderConfig()
    simulator: SimulatorConfig = SimulatorConfig()
