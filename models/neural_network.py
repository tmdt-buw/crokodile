import numpy as np
import torch
import torch.nn as nn
from pytorch3d import transforms
from torch.nn.functional import mse_loss

"""
        Helper function to generate neural network from hyperparameters.
        Activations between layers are ReLU.

        Params:
            in_dim: Input dimension of the network
            out_dim: Output dimension of the network
            network_width
            network_depth
            dropout
            out_activation: What activation should be used on the network output
    """


def create_network(
    self,
    in_dim,
    out_dim,
    network_width,
    network_depth,
    dropout,
    out_activation=None,
):
    network_structure = [
        ("linear", network_width),
        ("relu", None),
        ("dropout", dropout),
    ] * network_depth
    network_structure.append(("linear", out_dim))

    if out_activation:
        network_structure.append((out_activation, None))

    return NeuralNetwork(in_dim, network_structure)


class Clamp(torch.nn.Module):
    def __init__(self, min, max):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.sigmoid(x) * (self.max - self.min) + self.min


def init_xavier_uniform(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.xavier_uniform_(m.weight)


class NeuralNetwork(nn.Module):
    def __init__(self, in_dims, network_structure):
        super(NeuralNetwork, self).__init__()

        if type(in_dims) == int:
            in_dim = in_dims
        else:
            in_dim = int(np.sum([np.product(dim) for dim in in_dims]))

        assert type(in_dim) == int

        self.operators = nn.ModuleList([nn.Flatten()])

        current_dim = in_dim

        for name, params in network_structure:
            module, current_dim = self.get_module(name, params, current_dim)
            self.operators.append(module)

    def get_module(self, name, params, current_dim):
        if name == "res_block":
            module = ResBlock(current_dim, params)
        elif name == "linear":
            module = nn.Linear(current_dim, params)
            current_dim = params
        elif name == "relu":
            assert params is None, "No argument for ReLU please"
            module = nn.ReLU()
        elif name == "leaky_relu":
            assert params is None, "No argument for ReLU please"
            module = nn.LeakyReLU()
        elif name == "selu":
            assert params is None, "No argument for SeLU please"
            module = nn.SELU()
        elif name == "tanh":
            assert params is None, "No argument for Tanh please"
            module = nn.Tanh()
        elif name == "gelu":
            assert params is None, "No argument for GreLU please"
            module = nn.GELU()
        elif name == "dropout":
            module = nn.Dropout(params)
        elif name == "batchnorm":
            module = nn.BatchNorm1d(current_dim)
        else:
            raise NotImplementedError(f"{name} not known")

        return module, current_dim

    def forward(self, *args, **kwargs):
        x = torch.cat(args, dim=-1)

        for operator in self.operators:
            x = operator(x)
        return x

    def get_weights(self):
        weights = []

        for operator in self.operators:
            if type(operator) == nn.Linear:
                weights.append(operator.weight)

        return weights

    def get_activations(self, x):
        activations = []

        for operator in self.operators:
            x = operator(x)

            if type(operator) == nn.ReLU:
                activations.append(x)

        return activations
