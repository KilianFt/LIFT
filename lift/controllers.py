import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as torch_dist
from tensordict import TensorDict
from torchrl.modules.distributions import TanhNormal
from torchrl.modules import ProbabilisticActor
from torchrl.envs.utils import ExplorationType, set_exploration_type

from configs import BaseConfig
from lift.neural_nets import MLP
from lift.environments.gym_envs import NpGymEnv
from lift.rl.sac import SAC
from lift.utils import cross_entropy, normalize

class CategoricalEncoder(nn.Module):
    """Output a categorical distribution"""
    def __init__(self, input_dim, output_dim, hidden_dims, tau=0.5):
        super().__init__()
        self.mlp = MLP(
            input_dim, 
            output_dim, 
            hidden_dims, 
            dropout=0., 
            activation=nn.SiLU, 
            output_activation=None,
        )
        self.tau = tau
    
    def forward(self, x):
        p = torch.softmax(self.mlp(x), dim=-1)
        return p
    
    def sample(self, x):
        p = self.forward(x)
        z = F.gumbel_softmax(torch.log(p + 1e-6), tau=self.tau, hard=True)
        return z


class GaussianEncoder(nn.Module):
    """Output a gaussian distribution"""
    def __init__(self, input_dim, output_dim, hidden_dims, dropout=0.):
        super().__init__()
        self.mlp = MLP(
            input_dim, 
            output_dim * 2, 
            hidden_dims, 
            dropout=dropout,
            activation=nn.SiLU, 
            output_activation=None,
        )
        self._min_logstd = -20.0
        self._max_logstd = 2.0
    
    def forward(self, x):
        mu, ls = torch.chunk(self.mlp(x), 2, dim=-1)
        sd = torch.exp(ls.clip(self._min_logstd, self._max_logstd))
        return mu, sd
    
    def get_dist(self, x):
        mu, sd = self.forward(x)
        dist = torch_dist.Normal(mu, sd)
        return dist
    
    def sample(self, x):
        mu, sd = self.forward(x)
        dist = torch_dist.Normal(mu, sd)
        z = dist.rsample()
        return z


class TanhGaussianEncoder(nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            hidden_dims, 
            dropout=0., 
            out_min=-1., 
            out_max=1.,
        ):
        super().__init__()
        self.out_min = out_min
        self.out_max = out_max
        self.base = GaussianEncoder(
            input_dim, output_dim, hidden_dims, dropout
        )
    
    def get_dist(self, x):
        mu, sd = self.base.forward(x)
        dist = TanhNormal(mu, sd, min=self.out_min, max=self.out_max)
        return dist
    
    def sample(self, x):
        dist = self.get_dist(x)
        z = dist.rsample()
        return z


class BCTrainer(L.LightningModule):
    """Behavior cloning trainer"""
    def __init__(self, config: BaseConfig, env: NpGymEnv):
        super().__init__()
        self.lr = config.pretrain.lr
        self.x_dim = config.feature_size
        self.a_dim = config.action_size
        
        hidden_dims = [config.encoder.hidden_size for _ in range(config.encoder.n_layers)]
        self.act_max = torch.from_numpy(env.action_space.high[..., :-1]).to(torch.float32)
        self.act_min = torch.from_numpy(env.action_space.low[..., :-1]).to(torch.float32)

        self.encoder = TanhGaussianEncoder(
            self.x_dim, 
            self.a_dim, 
            hidden_dims, 
            dropout=config.encoder.dropout,
            out_min=self.act_min,
            out_max=self.act_max,
        )
        self.target_std = config.pretrain.target_std
        self.beta = 0.1

    # def compute_loss(self, dist, a):
    #     loss = -dist.log_prob(a).mean() / self.a_dim
    #     return loss
    
    def compute_loss(self, dist, a):
        # mae = torch.abs(dist.mode - a).mean()
        # std_loss = torch.abs(dist.scale - self.target_std).mean()
        mode_loss = nn.SmoothL1Loss().forward(dist.mode, a)
        std_loss = torch.pow(dist.scale - self.target_std, 2).mean()
        loss = mode_loss + self.beta * std_loss
        return loss, mode_loss, std_loss
    
    def training_step(self, batch, _):
        x = batch["emg_obs"]
        a = batch["act"]

        dist = self.encoder.get_dist(x)
        loss, _, _ = self.compute_loss(dist, a)
        
        mae = torch.abs(dist.mode - a).mean()
        std = dist.scale.mean()
        
        self.log("train_loss", loss.data.item())
        self.log("train_mae", mae.data.item())
        self.log("train_std", std.data.item())
        return loss

    def validation_step(self, batch, _):
        x = batch["emg_obs"]
        a = batch["act"]
        
        dist = self.encoder.get_dist(x)
        loss, _, _ = self.compute_loss(dist, a)
        
        mae = torch.abs(dist.mode - a).mean()
        std = dist.scale.mean()

        self.log("val_loss", loss.data.item())
        self.log("val_mae", mae.data.item(), prog_bar=True)
        self.log("val_std", std.data.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

class MITrainer(L.LightningModule):
    """Mutual information trainer"""
    def __init__(self, config: BaseConfig, env: NpGymEnv, teacher: SAC | None = None, supervise: bool = False):
        super().__init__()
        self.supervise = supervise
        self.sl_sd = config.mi.sl_sd
        self.num_neg_samples = config.mi.num_neg_samples
        self.lr = config.mi.lr
        self.beta_1 = config.mi.beta_1
        self.beta_2 = config.mi.beta_2
        self.beta_3 = config.mi.beta_3
        self.beta_4 = config.mi.beta_4
        self.kl_approx_method = config.mi.kl_approx_method

        assert self.kl_approx_method in ["logp", "abs", "mse"]

        self.x_dim = config.feature_size
        self.a_dim = config.action_size
        hidden_dims = [config.encoder.hidden_size for _ in range(config.encoder.n_layers)]
        self.act_max = torch.from_numpy(env.action_space.high[..., :-1]).to(torch.float32)
        self.act_min = torch.from_numpy(env.action_space.low[..., :-1]).to(torch.float32)

        self.encoder = TanhGaussianEncoder(
            self.x_dim, 
            self.a_dim, 
            hidden_dims, 
            dropout=config.encoder.dropout,
            out_min=self.act_min,
            out_max=self.act_max,
        )

        self.critic = MLP(
            self.x_dim + self.a_dim, 
            1, 
            hidden_dims, 
            dropout=0., 
            activation=nn.SiLU, 
            output_activation=None,
        )

        if teacher is not None:
            self.teacher = teacher.model.policy
        else:
            self.teacher = None
    
    def compute_mi_loss(self, x, z):
        """Compute infonce loss"""
        f = self.critic(torch.cat([x, z], dim=-1))

        sample_idx_neg = torch.randint(len(x), size=(self.num_neg_samples,))
        x_neg = x[sample_idx_neg]
        x_neg_ = x_neg.unsqueeze(0).repeat_interleave(len(x), dim=0)
        z_ = z.unsqueeze(1).repeat_interleave(len(x_neg), dim=1)
        f_neg_ = self.critic(torch.cat([x_neg_, z_], dim=-1))
        f_neg_ = torch.cat([f.unsqueeze(-2), f_neg_], dim=-2)

        labels = torch.zeros_like(f_neg_.squeeze(-1)).to(x.device)
        labels[:, 0] = 1
        p = torch.softmax(f_neg_.squeeze(-1), dim=-1)
        loss = cross_entropy(labels, p).mean()
        
        with torch.no_grad():
            accuracy = torch.sum(p.argmax(-1) == labels.argmax(-1)) / len(labels)
        
        stats = {
            "mi_loss": loss.data.cpu().item(),
            "mi_accuracy": accuracy.data.cpu().mean().item(),
        }
        return loss, stats
    
    def compute_kl_loss(self, o, z, z_dist):
        """Compute sample based kl divergence from teacher"""
        if isinstance(self.teacher, ProbabilisticActor):
            teacher_inputs = TensorDict({
                "observation": o,
                "action": z,
            })
            with torch.no_grad():
                teacher_dist = self.teacher.get_dist(teacher_inputs)
            log_prior = teacher_dist.log_prob(z)
        else:
            teacher_dist = torch_dist.Normal(
                torch.zeros(1, device=z.device), 
                torch.ones(1, device=z.device),
            )
            log_prior = teacher_dist.log_prob(z).sum(-1)

        log_post = z_dist.log_prob(z)
        ent = torch_dist.Normal(z_dist.loc, z_dist.scale).entropy().sum(-1)

        if self.kl_approx_method == "logp":
            # kl = -(log_prior - log_post).mean()
            kl = -(log_prior + ent).mean()
        elif self.kl_approx_method == "abs":
            kl = 0.5 * nn.SmoothL1Loss()(log_post, log_prior)
        elif self.kl_approx_method == "mse":
            with torch.no_grad():
                teacher_dist = self.teacher.get_dist(teacher_inputs)
                a_teacher = teacher_dist.mode
            kl = torch.pow(z - a_teacher, 2).mean()
        else:
            raise ValueError("kl approximation method much be one of [logp, abs, mse]")
        
        with torch.no_grad():
            mae_prior = torch.abs(teacher_dist.mode - z).mean()
            std_post = z_dist.sample((30,)).std(0).mean()

        stats = {
            "log_prior": log_prior.data.cpu().mean().item(),
            "log_post": log_post.data.cpu().mean().item(),
            "entropy": ent.data.cpu().mean().item(),
            "post_std": std_post.data.cpu().item(),
            "mae_prior": mae_prior.data.cpu().item(),
        }
        return kl, stats
    
    def compute_sl_loss(self, z, ideal_a):
        """Compute supervised loss"""


        if self.supervise:  # and y is not None:
            # y_dist = torch_dist.Normal(z, torch.ones(1, device=z.device) * self.sl_sd)
            # loss = -y_dist.log_prob(y).sum(-1).mean()
            # loss = -teacher_dist.log_prob(y).sum(-1).mean()

            loss = torch.pow(z - ideal_a, 2).mean()
        else:
            loss = torch.zeros(1, device=z.device)

        mae = torch.abs(z - ideal_a).mean()
        
        stats = {
            "sl_loss": loss.data.cpu().item(),
            "mae": mae.data.cpu().item()
        }
        return loss, stats

    def compute_cov(self, z):
        """Compute representation covariance"""
        z_mean = z.mean(0, keepdim=True)
        z_res = z - z_mean
        cov = torch.sum(z_res.unsqueeze(-2) * z_res.unsqueeze(-1), dim=0) / (len(z_res) - 1)
        return cov

    def compute_cov_loss(self, cov):
        mask = (1 - torch.eye(len(cov))).to(cov.device)
        loss = torch.sum(cov.abs() * mask) / mask.sum()
        return loss

    def compute_loss(self, batch):
        x = batch["emg_obs"]

        intended_a = batch["intended_action"]
        if "obs" in batch.keys():
            o = batch["obs"]
        else:
            o = None

        teacher_inputs = TensorDict({"observation": o})
        with set_exploration_type(ExplorationType.MODE), torch.no_grad():
            teacher_a = self.teacher(teacher_inputs)["action"]

        z_dist = self.encoder.get_dist(x)

        z = z_dist.rsample()
        mi_loss, mi_stats = self.compute_mi_loss(x, z)
        kl_loss, kl_stats = self.compute_kl_loss(o, z, z_dist)

        cov = self.compute_cov(z)
        cov_loss = self.compute_cov_loss(cov)

        sl_loss, sl_stats = self.compute_sl_loss(z, teacher_a)
        loss = self.beta_1 * mi_loss + self.beta_2 * kl_loss + self.beta_3 * cov_loss + self.beta_4 * sl_loss

        with torch.no_grad():
            pred_a = z_dist.mode

        mae = torch.abs(pred_a - intended_a).mean()
        missalignment_mae = torch.abs(teacher_a - intended_a).mean()

        stats = {
            "loss": loss.data.cpu().item(),
            "cov_loss": cov_loss.data.cpu().item(),
            "act_mae": mae.data.cpu().item(),
            "missalignment_mae": missalignment_mae.data.cpu().item(),
            "intended_magnitude": intended_a.abs().mean(),
            **mi_stats, **kl_stats, **sl_stats,
        }
        return loss, stats
    
    def training_step(self, batch, _):
        loss, stats = self.compute_loss(batch)

        for k, v in stats.items():
            self.log(f"train/{k}", v)
        return loss

    def validation_step(self, batch, _):
        loss, stats = self.compute_loss(batch)

        for k, v in stats.items():
            self.log(f"val/{k}", v)
        return loss

    def configure_optimizers(self):
        if self.optimizers() == []:
            # optimizer = torch.optim.Adam([self.encoder.parameters(), self.critic.parameters()], lr=self.lr)
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = self.optimizers()
        return optimizer


class EMGPolicy(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float):
        super(EMGPolicy, self).__init__()
        self.lr = lr
        self.model = model
        self.criterion = nn.MSELoss()

    def training_step(self, batch, _):
        x, y = batch
        predictions = self.model.sample(x)
        loss = self.criterion(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, _):
        x, y = val_batch
        predictions = self.model.sample(x)
        val_loss = self.criterion(predictions, y)
        mae = torch.abs(predictions - y).mean()
        self.log("val_loss", val_loss)
        self.log("val_mae", mae, prog_bar=True)

        return val_loss

    def configure_optimizers(self):
        # FIXME do we compute the gradient of the teacher here?
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class EMGAgent:
    """Wrapper for encoder action selection"""
    def __init__(self, policy: TanhGaussianEncoder | GaussianEncoder, obs_mu=0., obs_sd=1.):
        self.policy = policy
        self.obs_mu = obs_mu
        self.obs_sd = obs_sd

    def sample_action(self, observation: dict, sample_mean: bool = False) -> float:
        emg_obs = observation["emg_observation"]
        if not isinstance(emg_obs, torch.Tensor):
            emg_obs = torch.tensor(emg_obs, dtype=torch.float32)
            emg_obs = normalize(emg_obs, self.obs_mu, self.obs_sd)
        
        with torch.no_grad():
            dist = self.policy.get_dist(emg_obs)
            if sample_mean:
                act = dist.mode
            else:
                act = dist.sample()
        return act.detach().numpy()


if __name__ == "__main__":
    from lift.rl.env_utils import gym_env_maker, apply_env_transforms
    from lift.teacher import load_teacher
    torch.manual_seed(0)

    config = BaseConfig()
    env = NpGymEnv(
        "FetchReachDense-v2", 
        cat_obs=True, 
        cat_keys=config.teacher.env_cat_keys,
    )
    torchrl_env = apply_env_transforms(gym_env_maker("FetchReachDense-v2"))
    teacher = SAC(
        config.teacher, torchrl_env, torchrl_env
    )
    
    # synthetic data
    batch_size = 128
    batch = {
        "obs": torch.randn(batch_size, env.observation_space["observation"].shape[-1]),
        "emg_obs": 100 * torch.randn(batch_size, config.feature_size), # multiple 100 to get large action in encoder
        "act": torch.rand(batch_size, env.action_space.shape[-1]) * 2 - 1,
    }

    # test tanh encoder
    input_dim = config.feature_size
    output_dim = config.action_size
    hidden_dims = [config.encoder.hidden_size for _ in range(config.encoder.n_layers)]

    encoder = TanhGaussianEncoder(
        input_dim,
        output_dim,
        hidden_dims,
    )

    act_sample = encoder.sample(batch["emg_obs"])
    assert torch.all(act_sample.abs() < 1.)
    assert list(act_sample.shape) == [batch_size, 3]
    
    # test bc trainer
    bc_trainer = BCTrainer(config, env)
    with torch.no_grad():
        dist = bc_trainer.encoder.get_dist(batch["emg_obs"])
        loss, _, _ = bc_trainer.compute_loss(dist, batch["act"][..., :3])
    assert list(loss.shape) == []
    
    # test mi trainer
    mi_trainer = MITrainer(config, env, teacher)
    with torch.no_grad():
        z_dist = mi_trainer.encoder.get_dist(batch["emg_obs"])
        z = z_dist.sample()
        nce_loss = mi_trainer.compute_mi_loss(batch["emg_obs"], z)
        kl_loss = mi_trainer.compute_kl_loss(batch["obs"], z, z_dist)

    # test emg agent with tanh encoder
    agent = EMGAgent(encoder)
    act = agent.sample_action({"emg_observation": batch["emg_obs"].numpy()})
    assert list(act.shape) == [batch_size, 3]