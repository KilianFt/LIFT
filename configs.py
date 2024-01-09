import dataclasses

@dataclasses.dataclass
class BaseConfig:
    teacher_train_timesteps: int = 150_000
    n_channels: int = 1
    window_size: int = 32
    n_steps_rollout: int = 10_000
    hidden_size: int = 128
    n_layers: int = 2
    batch_size: int = 128 * 2
    epochs: int = 10
    lr: float = 1e-4
    gradient_clip_val: float = 1.0
    noise: float = 0.1
