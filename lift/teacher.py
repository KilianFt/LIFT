import copy
import numpy as np
from lift.rl.sac import SAC
from lift.rl.sac_meta import MetaSAC
from lift.rl.utils import parallel_env_maker

class ConditionedTeacher:
    """Wrapper to reset teacher with random meta variables for simulated trajectories"""
    def __init__(self, teacher, noise_range=0., alpha_range=0.):
        self.noise_range = noise_range
        self.alpha_range = alpha_range
        self.teacher = teacher

    def reset(self):
        noise = np.random.uniform(self.noise_range[0], self.noise_range[1])
        # alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
        # self.meta_vars = np.array([noise, alpha], dtype=self.observation_space["observation"].dtype)
        self.meta_vars = np.array([noise])
    
    def sample_action(self, obs, sample_mean=False):
        obs_ = copy.deepcopy(obs)
        # obs_["observation"][..., -2] = self.noise
        # obs_["observation"][..., -1] = self.alpha
        obs_["observation"][..., -1] = self.meta_vars
        return self.teacher.sample_action(obs_, sample_mean)


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