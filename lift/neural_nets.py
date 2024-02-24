import torch
import torch.nn as nn
    
class MLP(nn.Module):
    def __init__(
            self, 
            input_size, 
            output_size, 
            hidden_sizes, 
            activation=nn.ReLU, 
            output_activation=nn.Tanh,
            use_batch_norm=False, 
            dropout=0.1,
        ):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.Dropout(p=dropout))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(activation(inplace=True))

        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(activation(inplace=True))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        if output_activation is not None:
            layers.append(output_activation())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)