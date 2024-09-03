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
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    weight_decay: float = 0.0
    batch_size: int = 256
    target_update_polyak: float = 0.995
    alpha_init: float = 1.0
    adam_eps: float = 1.0e-8
    grad_clip: float | None = 1.0
    
    # nets
    hidden_sizes: list = [256, 256, 256]
    activation: str = "relu"
    default_policy_scale: float = 1.0
    scale_lb: float = 0.1
    device: str = "cpu"

    # eval
    eval_iter: int = 5000

    teacher_filename: str = "teacher.pt"
    meta_teacher_filename: str = "teacher_meta_3_layer_relu.pt"
    

class OfflineRLConfig(BaseModel):
    # replay
    replay_buffer_size: int = 1000000
    batch_size: int = 256
    num_slices: int = 8

    # optim 
    utd_ratio: float = 1.0
    gamma: float = 0.99
    loss_function: str = "l2"
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    weight_decay: float = 0.0
    target_update_polyak: float = 0.995
    alpha_init: float = 0.2
    adam_eps: float = 1.0e-8
    
    # nets
    hidden_sizes: list = [256, 256, 256]
    activation: str = "relu" # choices=["relu", "tanh", "leaky_relu"]
    default_policy_scale: float = 1.0
    scale_lb: float = 0.1
    device: str = "cpu"

    # train
    num_updates: int = 40_000
    kl_approx_method: str = "logp" # choices=["logp", "mse"] ("abs" not impletemeted atm)
    bellman_scaling: float = 0.5 # they use 0.5 in original CQL paper
    bc_regularization: float = 0.0

    # eval
    eval_iter: int = 1000
    eval_rollout_steps: int = 100


class EncoderConfig(BaseModel):    
    hidden_size: int = 256
    n_layers: int = 3
    dropout: float = 0.1
    activation: str = "silu"  # "relu", "tanh", "silu", "leaky_relu"


class SimulatorConfig(BaseModel):
    parametric: bool = False
    interpolation: str = "weighted" # choices=["weighted", "random"]
    reduction: str = "abs" # choices=["mean", "abs"] mean: /act_dict; abs: / actions.abs().sum(dim=-1)
    bias_noise: float = 0.005
    limits_noise: float = 0.01
    base_noise: float = 0.005
    n_bursts: int = 1
    recording_strength: float = 0.8
    k: int | None = 3 # number of samples considered in weighted simulator
    sample: bool = False


class PretrainConfig(BaseModel):
    target_std: float = 0.5 # only used in BCTrainer
    epochs: int = 50
    num_augmentation: int = 1_000
    augmentation_distribution: str = "uniform" # choices=["uniform", "normal"]
    train_subset: str = "combined" # choices=["interpolation", "combined"] MAD only is when num_aug is 0
    train_ratio: float = 0.8
    batch_size: int = 512
    lr: float = 1.0e-3
    beta_1: float = 1.0 # mi weight, use 0.5 for mse
    beta_2: float = 0.1 # kl weight
    beta_3: float = 1.0 # sl weight


class MIConfig(BaseModel):
    beta_1: float = 1.0 # mi weight, use 0.5 for mse
    beta_2: float = 0.1 # kl weight
    # beta_3: float = 1.0 # sl weight
    entropy_beta: float = 0.01
    ft_weight: float = 1. # finetune loss weight
    pt_weight: float = 0.5 # pretrain loss weight
    kl_approx_method: str = "logp" # choices=[logp, abs, mse]
    num_neg_samples: int = 50
    sl_sd: float = 0.2 # fixed sl std
    train_ratio: float = 0.8
    batch_size: int = 256
    epochs: int = 20
    max_steps: int = 2000
    lr: float = 1.0e-3
    n_steps_rollout: int = 10_000
    random_pertube_prob: float = 0.0
    action_noise: float = 0.0
    activation: str = "silu"  # "relu", "tanh", "silu", "leaky_relu"

    # iter mi
    aggregate_data: bool = True

    # for comparison
    only_copy_teacher: bool = False

    num_sessions: int = 1


class BaseConfig(BaseModel):
    # path config
    root_path: PosixPath = ROOT_PATH
    data_path: PosixPath = ROOT_PATH / "datasets"
    mad_base_path: PosixPath = ROOT_PATH / "datasets" / "MyoArmbandDataset"
    mad_data_path: PosixPath = mad_base_path / "PreTrainingDataset"
    models_path: PosixPath = ROOT_PATH / "models"
    rollout_data_path: PosixPath = ROOT_PATH / "datasets" / "rollouts"
    results_path: PosixPath = ROOT_PATH / "results"
    target_person: str = "Female0"
    val_people: List[str] = ["Male8", "Female4"]
    
    # wandb
    use_wandb: bool = True
    project_name: str = "lift"
    run_name: str | None = None
    wandb_mode: str = "online"

    # data
    num_channels: int = 8
    window_size: int = 200
    window_overlap: int = 150
    emg_range: list = [-128., 127.]
    desired_mad_labels: list = [0, 1, 2, 3, 4, 5, 6]
    cutoff_n_outer_samples: int = 0 # num of samples to discard in beginning and end of each recording

    # user model
    noise_range: list | None = [-0.2, 1.2] # noise added to teacher env
    noise_slope_range: list | None = [-0.2, 1.2] # action dependent noise
    alpha_range: list | None = [1., 3.] # ratio multiplied to teacher std
    alpha_apply_range: list | None = [0., 3.,] # goal dist range to apply alpha scaling

    # use this to fix the values
    noise: float | None = 0.
    noise_slope: float | None = 0.
    alpha: float | None = 1.

    noise_drift: list | None = None#[-0.1, 0.0] # [offset, std]
    alpha_drift: list | None = None#[-0.1, 0.0] # [-0.1, 0.2] # [offset, std]
    # what if user is biased?
    user_bias: float | None = None

    seed: int = 1001
    num_workers: int = 7
    teacher_train_timesteps: int = 150_000
    action_size: int = 3  # could be read from env
    feature_size: int = 8  # could be read from env

    gradient_clip_val: float = 2.
    checkpoint_frequency: int = 1
    save_top_k: int = -1  # set to -1 to save all checkpoints
    
    teacher: TeacherConfig = TeacherConfig()
    encoder: EncoderConfig = EncoderConfig()
    simulator: SimulatorConfig = SimulatorConfig()
    pretrain: PretrainConfig = PretrainConfig()
    mi: MIConfig = MIConfig()
    offline_rl: OfflineRLConfig = OfflineRLConfig()
