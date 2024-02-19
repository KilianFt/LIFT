from pydantic import BaseModel
from typing import List

class EncoderConfig(BaseModel):
    h_dim: int = 128
    tau: float = 0.5
    beta: float = 0.1
    hidden_size: int = 256
    n_layers: int = 4

class SimulatorConfig(BaseModel):
    n_bursts: int = 1
    recording_strength: float = 0.5

class BaseConfig(BaseModel):
    seed: int = 100
    teacher_train_timesteps: int = 150_000
    action_size: int = 3 # could be read from env
    feature_size: int = 32 # could be read from env
    n_channels: int = 8
    window_size: int = 200 # set to same as action_size to match the dimesions with FakeSim
    n_steps_rollout: int = 20_000

    dropout: float = .1
    batch_size: int = 128
    epochs: int = 200
    lr: float = 1e-4
    gradient_clip_val: float = 0.5
    noise: float = 1e-6
    use_batch_norm: bool = False
    checkpoint_frequency: int = 1
    save_top_k: int = -1 # set to -1 to save all checkpoints

    encoder: EncoderConfig = EncoderConfig()
    simulator: SimulatorConfig = SimulatorConfig()
