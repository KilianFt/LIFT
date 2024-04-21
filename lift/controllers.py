import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as torch_dist
from tensordict import TensorDict
from torchrl.modules.distributions import TanhNormal

from configs import BaseConfig
from lift.neural_nets import MLP
from lift.environments.gym_envs import NpGymEnv
from lift.rl.sac import SAC
from lift.utils import cross_entropy

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

    def compute_loss(self, dist, a):
        loss = -dist.log_prob(a).mean()
        return loss
    
    def training_step(self, batch, _):
        x = batch["emg_obs"]
        a = batch["act"]

        dist = self.encoder.get_dist(x)
        loss = self.compute_loss(dist, a)
        mae = torch.abs(dist.mode - a).mean()
        
        self.log("train_loss", loss.data.item())
        self.log("train_mae", mae.data.item())
        return loss

    def validation_step(self, batch, _):
        x = batch["emg_obs"]
        a = batch["act"]
        
        dist = self.encoder.get_dist(x)
        loss = self.compute_loss(dist, a)
        mae = torch.abs(dist.mode - a).mean()

        self.log("val_loss", loss.data.item())
        self.log("val_mae", mae.data.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

class MITrainer(L.LightningModule):
    """Mutual information trainer"""
    def __init__(self, config: BaseConfig, env: NpGymEnv, teacher: SAC):
        super().__init__()
        self.lr = config.lr
        self.beta_1 = config.encoder.beta_1
        self.beta_2 = config.encoder.beta_2
        self.kl_approx_method = config.encoder.kl_approx_method

        self.x_dim = config.feature_size
        self.a_dim = config.action_size
        h_dim = config.encoder.h_dim
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
        self.critic_x = MLP(
            self.x_dim, 
            h_dim, 
            hidden_dims, 
            dropout=0., 
            activation=nn.SiLU, 
            output_activation=None,
        )
        self.critic_z = MLP(
            self.a_dim, 
            h_dim, 
            hidden_dims, 
            dropout=0., 
            activation=nn.SiLU, 
            output_activation=None,
        ) 
        self.teacher = teacher.model.policy
    
    def compute_infonce_loss(self, x, z):
        h_x = self.critic_x(x)
        h_z = self.critic_z(z)

        f = torch.einsum("ih, jh -> ij", h_z, h_x)
        p = torch.softmax(f, dim=-1)
        labels = torch.eye(len(h_x)).to(x.device)
        loss = cross_entropy(labels, p)
        return loss.mean()
    
    def compute_kl_loss(self, o, z, z_dist):
        """Compute sample based kl divergence from teacher"""
        teacher_inputs = TensorDict({
            "observation": o,
            "action": F.pad(z, (0, 1), value=0),
        })

        if self.kl_approx_method == "logp":
            log_prior = self.teacher.log_prob(teacher_inputs)
            ent = -z_dist.log_prob(z)
            kl = -(log_prior + ent).mean()
        elif self.kl_approx_method == "abs":
            # john schulman kl approximation
            log_prior = self.teacher.log_prob(teacher_inputs)
            log_post = z_dist.log_prob(z)
            kl = 0.5 * nn.SmoothL1Loss()(log_post, log_prior)
        elif self.kl_approx_method == "mse":
            with torch.no_grad():
                teacher_dist = self.teacher.get_dist(teacher_inputs)
                a_teacher = teacher_dist.mode[..., :-1]
            kl = torch.pow(z - a_teacher, 2).mean()
        else:
            raise ValueError("kl approximation method much be one of [logp, abs, mse]")
        
        return kl
    
    def compute_loss(self, x, o, z_dist):
        z = z_dist.rsample()
        nce_loss = self.compute_infonce_loss(x, z)
        kl_loss = self.compute_kl_loss(o, z, z_dist)
        loss = self.beta_1 * nce_loss + self.beta_2 * kl_loss
        return loss, nce_loss, kl_loss
    
    def training_step(self, batch, _):
        o = batch["obs"]
        x = batch["emg_obs"]
        a = batch["act"]
        a = a[:,:self.a_dim].clone() # remove last element from y

        z_dist = self.encoder.get_dist(x)
        loss, _, _ = self.compute_loss(x, o, z_dist)
        mae = torch.abs(z_dist.mode - a).mean()
        
        self.log("train_loss", loss.data.item())
        self.log("train_mae", mae.data.item())
        return loss

    def validation_step(self, batch, _):
        o = batch["obs"]
        x = batch["emg_obs"]
        a = batch["act"]
        a = a[:,:self.a_dim].clone() # remove last element from y
        
        z_dist = self.encoder.get_dist(x)
        loss, nce_loss, kl_loss = self.compute_loss(x, o, z_dist)

        mae = torch.abs(z_dist.mode - a).mean()

        self.log("val_loss", loss.data.item())
        self.log("val_nce_loss", nce_loss.data.item())
        self.log("val_kl_loss", kl_loss.data.item())
        self.log("val_mae", mae.data.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
        self.log("val_loss", val_loss)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class EMGAgent:
    """Wrapper for encoder action selection"""
    def __init__(self, policy: TanhGaussianEncoder | GaussianEncoder):
        self.policy = policy

    def sample_action(self, observation: dict, sample_mean: bool = False) -> float:
        emg_obs = observation["emg_observation"]
        if not isinstance(emg_obs, torch.Tensor):
            emg_obs = torch.tensor(emg_obs, dtype=torch.float32)
        
        with torch.no_grad():
            dist = self.policy.get_dist(emg_obs)
            if sample_mean:
                act = dist.mode
            else:
                act = dist.sample()
            """TODO: properly address action dimension mismatch in emg env"""
            act = F.pad(act, (0, 1), value=0) # pad zero to last action dimension
        return act.detach().numpy()


if __name__ == "__main__":
    from lift.rl.utils import gym_env_maker, apply_env_transforms
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
    h_dim = config.encoder.h_dim
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
        loss = bc_trainer.compute_loss(dist, batch["act"][..., :3])
    assert list(loss.shape) == []
    
    # test mi trainer
    mi_trainer = MITrainer(config, env, teacher)
    with torch.no_grad():
        z_dist = mi_trainer.encoder.get_dist(batch["emg_obs"])
        z = z_dist.sample()
        nce_loss = mi_trainer.compute_infonce_loss(batch["emg_obs"], z)
        kl_loss = mi_trainer.compute_kl_loss(batch["obs"], z, z_dist)

    # test emg agent with tanh encoder
    agent = EMGAgent(encoder)
    act = agent.sample_action({"emg_observation": batch["emg_obs"].numpy()})
    assert list(act.shape) == [batch_size, 4]
    assert sum(act[:, -1] == 0) == batch_size