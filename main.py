from configs import BaseConfig
from lift.teacher import maybe_train_teacher


def main():
    config = BaseConfig()
    teacher = maybe_train_teacher(config)


if __name__ == '__main__':
    main()
