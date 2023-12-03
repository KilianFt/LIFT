import dataclasses

@dataclasses.dataclass
class BaseConfig:
    teacher_train_timesteps: int = 150_000
    n_channels: int = 4
    window_size: int = 100
    n_steps_rollout: int = 1000
    batch_size: int = 32
    epochs: int = 10
