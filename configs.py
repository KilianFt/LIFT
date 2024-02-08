from pydantic import BaseModel


class BaseConfig(BaseModel):
    seed: int = 100
    teacher_train_timesteps: int = 150_000
    action_size: int = 3 # could be read from env
    n_channels: int = 8
    window_size: int = 200 # set to same as action_size to match the dimesions with FakeSim
    n_bursts: int = 1
    n_steps_rollout: int = 20_000
    hidden_size: int = 128
    n_layers: int = 5
    dropout: float = .25
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    gradient_clip_val: float = 0.1
    noise: float = 0.001
    use_batch_norm: bool = False
    checkpoint_frequency: int = 1
    save_top_k: int = -1 # set to -1 to save all checkpoints
