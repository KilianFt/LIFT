import dataclasses

@dataclasses.dataclass
class BaseConfig:
    teacher_train_timesteps: int = 150_000
    n_channels: int = 4
    window_size: int = 100
    n_steps_rollout: int = 10000
    batch_size: int = 64
    epochs: int = 2
    lr: float = 1e-3
    gradient_clip_val: float = 1.0
