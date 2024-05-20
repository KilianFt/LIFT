from configs import BaseConfig
from lift.rl.sac import SAC
from lift.rl.env_utils import parallel_env_maker
from torchrl.record.loggers import generate_exp_name, get_logger

def main():
    config = BaseConfig()
    
    exp_name = generate_exp_name("MetaSAC", f"{config.teacher.env_name}")
    logger = None
    if config.use_wandb:
        logger = get_logger(
            logger_type="wandb",
            logger_name="sac_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": config.wandb_mode,
                "config": dict(config),
                "project": config.project_name,
                "group": "teacher",
            },
        )

    train_env = parallel_env_maker(
        config.teacher.env_name,
        config,
        meta=True,
        cat_obs=config.teacher.env_cat_obs,
        cat_keys=config.teacher.env_cat_keys,
        max_eps_steps=config.teacher.max_eps_steps,
        device="cpu",
    )
    eval_env = parallel_env_maker(
        config.teacher.env_name,
        config,
        meta=True,
        cat_obs=config.teacher.env_cat_obs,
        cat_keys=config.teacher.env_cat_keys,
        max_eps_steps=config.teacher.max_eps_steps,
        device="cpu",
    )

    train_env.set_seed(config.seed)
    eval_env.set_seed(config.seed)
    
    sac = SAC(
        config.teacher, 
        train_env,
        eval_env,
    )
    sac.train(logger)
    sac.save(config.models_path / "teacher_meta.pt")

if __name__ == "__main__":
    main()