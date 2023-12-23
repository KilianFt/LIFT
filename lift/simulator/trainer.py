import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from lift.neural_nets import MLP
from lift.simulator.simulator import WindowSimulator

import nevergrad as ng

class Trainer:
    def __init__(
            self, 
            features_list, 
            num_actions, 
            num_channels, 
            window_size, 
            num_bursts,
            hidden_sizes,
            batch_size,
            grad_target,
            d_iters,
            g_iters,
        ):
        input_size = len(features_list) * num_channels + num_actions
        self.discriminator = MLP(input_size, hidden_sizes, 1)
        self.generator = WindowSimulator(
            num_actions=num_actions,
            num_bursts=num_bursts,
            num_channels=num_channels,
            window_size=window_size,
        )

        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters())
        
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.grad_target = grad_target
        self.d_iters = d_iters
        self.g_iters = g_iters
    
    def sample_batch(self):
        idx = torch.randint(len(self.data), (self.batch_size,))
        return self.data[idx]
    
    def compute_gradient_penalty(self, real_inputs, fake_inputs):
        # interpolate data
        alpha = torch.rand(len(real_inputs), 1)
        interpolated = alpha * real_inputs + (1 - alpha) * fake_inputs
        interpolated = Variable(interpolated, requires_grad=True)

        prob = torch.sigmoid(self.discriminator(interpolated))
        
        grad = torch_grad(
            outputs=prob, inputs=interpolated, 
            grad_outputs=torch.ones_like(prob),
            create_graph=True, retain_graph=True
        )[0]

        grad_norm = torch.linalg.norm(grad, dim=-1)
        grad_pen = torch.pow(grad_norm - self.grad_target, 2).mean()
        return grad_pen
    
    def train_discriminator(self):
        epoch_loss = []
        for _ in range(self.d_iters):
            # sample real
            real_data = self.sample_batch()

            # sample fake
            actions = real_data[..., -self.num_actions:]
            windows = self.generator(actions)
            fake_features = self.generator.compute_features(windows)
            fake_data = torch.cat([fake_features, actions], dim=-1)

            # train discriminator
            data = torch.cat([real_data, fake_data], dim=0)
            labels = torch.zeros(len(data), 1)
            labels[-len(fake_data):] = 1

            d_out = torch.sigmoid(self.discriminator.forward(data))
            d_loss = F.binary_cross_entropy(d_out, labels)
            d_total_loss = d_loss + 10 * self.compute_gradient_penalty(real_data, fake_data)

            self.d_optimizer.zero_grad()
            d_total_loss.backward()
            self.d_optimizer.step()

            epoch_loss.append(d_loss.data.item())
        epoch_loss = np.mean(epoch_loss)
        return epoch_loss
    
    def train_generator(self):
        def obj_func(bias_range, emg_range):
            # sample actions uniformly
            actions = torch.randint(self.num_actions, (self.batch_size,))
            actions = F.one_hot(actions, num_classes=self.num_actions).float()

            self.generator.set_params(
                torch.from_numpy(bias_range).to(torch.float32), 
                torch.from_numpy(emg_range).to(torch.float32),
            )
            windows = self.generator(actions)
            features = self.generator.compute_features(windows)
            fake_data = torch.cat([features, actions], dim=-1)
            with torch.no_grad():
                d_out = torch.sigmoid(self.discriminator.forward(fake_data))
            return torch.mean(d_out).numpy()

        instrum = ng.p.Instrumentation(
            bias_range=ng.p.Array(shape=self.generator.bias_range_shape), 
            emg_range=ng.p.Array(shape=self.generator.emg_range_shape), 
        )
        optimizer = ng.optimizers.NGOpt(
            parametrization=instrum, 
            budget=self.d_iters, 
            num_workers=1,
        )

        epoch_loss = []
        for _ in range(optimizer.budget):
            p = optimizer.ask()
            loss = obj_func(**p.kwargs)
            optimizer.tell(p, loss)
            epoch_loss.append(loss)

        recommendation = optimizer.provide_recommendation()
        with open('best_params.pkl', 'wb') as f:
            pickle.dump(recommendation.kwargs, f)

        epoch_loss = np.mean(epoch_loss)
        return epoch_loss
    
    def train(self, data, epochs):
        self.data = data

        history = {
            "d_loss": [],
            "g_loss": [],
        }
        for e in range(epochs):
            d_loss = self.train_discriminator()
            g_loss = self.train_generator()

            print(e, d_loss, g_loss)
            history["d_loss"].append(d_loss)
            history["g_loss"].append(g_loss)
        return history