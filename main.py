import wandb
import lightning as L

from configs import BaseConfig
from lift.teacher import maybe_train_teacher
from lift.environment import EMGWrapper
from lift.simulator.simulator import WindowSimulator
import pretraining


def main():
    run = wandb.init(project='lift')
    config = BaseConfig(**wandb.config)
    L.seed_everything(config.seed)

    teacher = maybe_train_teacher(config)

    sim = WindowSimulator(
        action_size=config.action_size, 
        num_bursts=config.simulator.n_bursts, 
        num_channels=config.n_channels,
        window_size=config.window_size, 
        return_features=True,
    )

    emg_env = EMGWrapper(teacher, sim)
    policy = pretraining.train_policy(emg_env, config)

    emg_env.close()


if __name__ == '__main__':
    main()
