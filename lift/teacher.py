from lift.rl.sac import SAC
from lift.rl.sac_meta import MetaSAC
from lift.rl.utils import parallel_env_maker


def load_teacher(config, load_frozen=True, meta=False):
    train_env = parallel_env_maker(
        config.teacher.env_name,
        config,
        meta=meta,
        cat_obs=config.teacher.env_cat_obs,
        cat_keys=config.teacher.env_cat_keys,
        max_eps_steps=config.teacher.max_eps_steps,
        device="cpu",
    )
    eval_env = parallel_env_maker(
        config.teacher.env_name,
        config,
        meta=meta,
        cat_obs=config.teacher.env_cat_obs,
        cat_keys=config.teacher.env_cat_keys,
        max_eps_steps=config.teacher.max_eps_steps,
        device="cpu",
    )
    if not meta:
        sac = SAC(config.teacher, train_env, eval_env)
        sac.load(config.models_path / "teacher.pt")
        print("\nSAC teacher loaded")
    else:
        sac = MetaSAC(config.teacher, train_env, eval_env)
        sac.load(config.models_path / "teacher_meta.pt")
        print("\nMetaSAC teacher loaded")

    if load_frozen:
        for model in sac.model.values():
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
    return sac