from lift.rl.sac import SAC
from lift.rl.utils import parallel_env_maker


def load_teacher(config):
    train_env = parallel_env_maker(
        config.teacher.env_name,
        cat_obs_keys=["observation", "desired_goal", "achieved_goal"],
        max_eps_steps=config.teacher.max_eps_steps,
        device="cpu",
    )
    eval_env = parallel_env_maker(
        config.teacher.env_name,
        cat_obs_keys=["observation", "desired_goal", "achieved_goal"],
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