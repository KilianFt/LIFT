import dataclasses

@dataclasses.dataclass
class BaseConfig:
    teacher_train_timesteps: int = 120_000
