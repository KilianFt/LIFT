import copy
import numpy as np
from lift.rl.sac import SAC
from lift.rl.sac_meta import MetaSAC
from lift.rl.utils import parallel_env_maker

class ConditionedTeacher:
    """Wrapper to reset teacher with random meta variables for simulated trajectories"""
    def __init__(
        self, 
        teacher: SAC | MetaSAC, 
        noise_range: list[float] | None = [0.001, 1.], 
        alpha_range: list[float] | None = [0.001, 1.], 
    ):
        self.is_meta = isinstance(teacher, MetaSAC)
        self.noise_range = noise_range
        self.alpha_range = alpha_range
        self.teacher = teacher

    def reset(self):
        self.meta_vars = None
        meta_vars = []
        if self.noise_range is not None:
            noise = np.random.uniform(self.noise_range[0], self.noise_range[1])
            meta_vars.append(noise)
        if self.alpha_range is not None:
            alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
            meta_vars.append(alpha)
        
        if meta_vars != []:
            self.meta_vars = np.hstack(meta_vars)
    
    def sample_action(self, obs, sample_mean=False):
        obs_ = copy.deepcopy(obs)

        if self.is_meta and self.meta_vars is not None:
            obs_["observation"] = np.concatenate([obs_["observation"], self.meta_vars], axis=-1)
        return self.teacher.sample_action(obs_, sample_mean)


def apply_gaussian_drift(z, offset, std, range=[-np.inf, np.inf]):
    """Apply gaussian drift to variable z and clip to bound"""
    drift = np.random.normal(offset, std)
    z_new = np.clip(z + drift, range[0], range[1])
    return z_new

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