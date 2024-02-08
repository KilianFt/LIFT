import wandb
import lightning as L

from configs import BaseConfig
from lift.teacher import maybe_train_teacher
from lift.environment import EMGWrapper
import pretraining


def main():
    run = wandb.init(project='lift')
    config = BaseConfig(**wandb.config)
    L.seed_everything(config.seed)

    teacher = maybe_train_teacher(config)

    emg_env = EMGWrapper(teacher, config)
    policy = pretraining.train_policy(emg_env, config)

    emg_env.close()


if __name__ == '__main__':
    main()
