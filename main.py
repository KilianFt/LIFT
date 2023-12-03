from configs import BaseConfig
from lift.teacher import maybe_train_teacher
import pretraining


def main():
    config = BaseConfig()
    teacher = maybe_train_teacher(config)

    policy = pretraining.train_policy(teacher, config)


if __name__ == '__main__':
    main()
