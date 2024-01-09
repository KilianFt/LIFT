import dataclasses

@dataclasses.dataclass
class BaseConfig:
    seed: int = 100
    teacher_train_timesteps: int = 150_000
    n_channels: int = 1
    window_size: int = 10
    n_steps_rollout: int = 20_000
    hidden_size: int = 256
    n_layers: int = 7
    dropout: float = .45
    batch_size: int = 128
    epochs: int = 20
    lr: float = 3e-4
    gradient_clip_val: float = 0.4
    noise: float = 0.1
    use_batch_norm: bool = True
