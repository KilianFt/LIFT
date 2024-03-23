from lift.rl.sac import SAC
from lift.rl.utils import parallel_env_maker


def load_teacher(config):
    train_env = parallel_env_maker(
        config.teacher.env_name,
        cat_obs=config.teacher.env_cat_obs,
        cat_keys=config.teacher.env_cat_keys,
        max_eps_steps=config.teacher.max_eps_steps,
        device="cpu",
    )
    eval_env = parallel_env_maker(
        config.teacher.env_name,
        cat_obs=config.teacher.env_cat_obs,
        cat_keys=config.teacher.env_cat_keys,
        max_eps_steps=config.teacher.max_eps_steps,
        device="cpu",
    )
    sac = SAC(
        config.teacher, 
        train_env,
        eval_env,
    )
    sac.load(config.models_path / "teacher.pt")
    return sac