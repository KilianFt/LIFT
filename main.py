from configs import BaseConfig
from lift.teacher import maybe_train_teacher
from lift.environment import EMGWrapper
import pretraining


def main():
    config = BaseConfig()

    teacher = maybe_train_teacher(config)

    emg_env = EMGWrapper(teacher, config)
    policy = pretraining.train_policy(emg_env, config)

    emg_env.close()


if __name__ == '__main__':
    main()
