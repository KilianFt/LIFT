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
    
class Encoder(nn.Module):
    """Output a categorical distribution"""
    def __init__(self, input_dim, output_dim, hidden_size, hidden_dims, tau=0.5):
        super().__init__()
        self._mlp = MLP(input_dim, hidden_size, hidden_dims, dropout=0., activation=nn.SiLU, output_activation=None)
        self._mu = nn.Linear(hidden_size, output_dim)
        self._logstd = nn.Linear(hidden_size, output_dim)
        self._min_logstd = -20.0
        self._max_logstd = 2.0
        self.tau_ = tau
    
    def forward(self, x):
        z = self._mlp(x)
        mu = self._mu(z)
        log_std = self._logstd(z)
        clipped_logstd = log_std.clip(self._min_logstd, self._max_logstd)
        return mu, clipped_logstd
    
    def sample(self, x):
        mu, log_std = self.forward(x)
        std = torch.exp(log_std)
        dist = torch_dist.Normal(mu, std)
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
    def __init__(self, config):
        super(EMGEncoder, self).__init__()
        self.lr = config.lr
        self.beta = config.encoder.beta

        x_dim = config.feature_size
        z_dim = config.action_size
        h_dim = config.encoder.h_dim
        hidden_dims = [config.encoder.hidden_size for _ in range(config.encoder.n_layers)]
        tau = config.encoder.tau

        self.encoder = Encoder(x_dim, z_dim, h_dim, hidden_dims, tau=tau)
        self.critic_x = MLP(x_dim, h_dim, hidden_dims, dropout=0., activation=nn.SiLU, output_activation=None)
        self.critic_z = MLP(z_dim, h_dim, hidden_dims, dropout=0., activation=nn.SiLU, output_activation=None)

        self.criterion = nn.MSELoss()



    def compute_infonce_loss(self, x, z):
        h_x = self.critic_x(x)
        h_z = self.critic_z(z)

        f = torch.einsum("ih, jh -> ij", h_z, h_x)
        p = torch.softmax(f, dim=-1)
        labels = torch.eye(len(h_x)).to(x.device)
        loss = cross_entropy(labels, p)
        return loss.mean()

    def training_step(self, batch, _):
        x, y = batch
        # remove last element from y
        y = y[:,:-1].clone()
        z = self.encoder.sample(x)

        # nce_loss = self.compute_infonce_loss(x, z)
        # kl_loss = compute_kl_loss(z, y)
        # loss = nce_loss + self.beta * kl_loss
        loss = self.criterion(z, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, _):
        x, y = val_batch
        # remove last element from y
        y = y[:,:-1].clone()
        z = self.encoder.sample(x)

        # nce_loss = self.compute_infonce_loss(x, z)
        # kl_loss = compute_kl_loss(z, y)
        # val_loss = nce_loss + self.beta * kl_loss
        val_loss = self.criterion(z, y)

        self.log("val_loss", val_loss)

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
        action = self.policy.sample(emg_obs)
        np_action = action.detach().numpy()
        # add another dimension to match the action space (not used in reach)
        zeros = np.zeros((np_action.shape[0], 1))
        return np.concatenate((np_action, zeros), axis=1)

    def update(self):
        pass
