import lightning as L
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as torch_dist

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, use_batch_norm=False, dropout=0.1,
                 activation=nn.ReLU, output_activation=nn.Tanh):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.Dropout(p=dropout))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(activation(inplace=True))

        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.Dropout(p=dropout))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(activation(inplace=True))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        if output_activation is not None:
            layers.append(output_activation())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.network(x)


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
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        self.mlp = MLP(
            input_dim, 
            output_dim * 2, 
            hidden_dims, 
            dropout=0., 
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


def compute_kl_loss(z, y):
    # estimate probability of z and y
    z_prob = F.log_softmax(z, dim=-1)
    y_prob = F.log_softmax(y, dim=-1)
    return F.kl_div(z_prob, y_prob, log_target=True, reduction='batchmean')


def cross_entropy(p, q, eps=1e-6):
    logq = torch.log(q + eps)
    ce = -torch.sum(p * logq, dim=-1)
    return ce


class EMGEncoder(L.LightningModule):
    """Encode EMG features using MI maximization"""
    def __init__(self, config):
        super(EMGEncoder, self).__init__()
        self.lr = config.lr
        self.beta = config.encoder.beta

        self.x_dim = config.feature_size
        self.z_dim = config.action_size
        self.h_dim = config.encoder.h_dim
        hidden_dims = [config.encoder.hidden_size for _ in range(config.encoder.n_layers)]

        self.encoder = GaussianEncoder(
            self.x_dim, self.z_dim, hidden_dims
        )
        self.critic_x = MLP(
            self.x_dim, 
            self.h_dim, 
            hidden_dims, 
            dropout=0., 
            activation=nn.SiLU, 
            output_activation=None,
        )
        self.critic_z = MLP(
            self.z_dim, 
            self.h_dim, 
            hidden_dims, 
            dropout=0., 
            activation=nn.SiLU, 
            output_activation=None,
        ) 

    def compute_infonce_loss(self, x, z):
        h_x = self.critic_x(x)
        h_z = self.critic_z(z)

        f = torch.einsum("ih, jh -> ij", h_z, h_x)
        p = torch.softmax(f, dim=-1)
        labels = torch.eye(len(h_x)).to(x.device)
        loss = cross_entropy(labels, p)
        return loss.mean()
    
    def compute_loss(self, x, z, y):
        nce_loss = self.compute_infonce_loss(x, z)
        kl_loss = torch.pow(z - y, 2).mean()
        loss = self.beta * nce_loss + kl_loss
        return loss, nce_loss, kl_loss
    
    def training_step(self, batch, _):
        x, y = batch
        y = y[:,:self.z_dim].clone() # remove last element from y

        z = self.encoder.sample(x)
        loss, _, _ = self.compute_loss(x, z, y)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, _):
        x, y = val_batch
        y = y[:,:self.z_dim].clone() # remove last element from y
        
        z = self.encoder.sample(x)
        val_loss, nce_loss, kl_loss = self.compute_loss(x, z, y)

        self.log("val_loss", val_loss)
        self.log("val_nce_loss", nce_loss)
        self.log("val_kl_loss", kl_loss)

        return val_loss

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
    def __init__(self, policy):
        self.policy = policy

    def sample_action(self, observation) -> float:
        emg_obs = torch.tensor(observation['emg_observation'], dtype=torch.float32)
        with torch.no_grad():
            action = self.policy.sample(emg_obs)
        np_action = action.detach().numpy()
        # add another dimension to match the action space (not used in reach)
        zeros = np.zeros((np_action.shape[0], 1))
        return np.concatenate((np_action, zeros), axis=1)

    def update(self):
        pass
