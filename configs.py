from pydantic import BaseModel


class BaseConfig(BaseModel):
    seed: int = 100
    teacher_train_timesteps: int = 150_000
    action_size: int = 4 # could be read from env
    n_channels: int = 4
    window_size: int = 4 # set to same as action_size to match the dimesions with FakeSim
    n_steps_rollout: int = 20_000
    hidden_size: int = 1024
    n_layers: int = 5
    dropout: float = .35
    batch_size: int = 256
    epochs: int = 70
    lr: float = 1e-4
    gradient_clip_val: float = 0.1
    noise: float = 0.1
    use_batch_norm: bool = False
