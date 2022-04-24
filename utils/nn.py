import numpy as np
import torch
import torch.nn as nn
from pytorch3d import transforms
from torch.nn.functional import mse_loss


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

        self.operators = nn.ModuleList([
            nn.Flatten()
        ])

        current_dim = in_dim

        for name, params in network_structure:
            module, current_dim = self.get_module(name, params, current_dim)
            self.operators.append(module)

    def get_module(self, name, params, current_dim):
        if name == 'res_block':
            module = ResBlock(current_dim, params)
        elif name == 'linear':
            module = nn.Linear(current_dim, params)
            current_dim = params
        elif name == 'relu':
            assert params is None, 'No argument for ReLU please'
            module = nn.ReLU()
        elif name == "leaky_relu":
            assert params is None, 'No argument for ReLU please'
            module = nn.LeakyReLU()
        elif name == 'selu':
            assert params is None, 'No argument for SeLU please'
            module = nn.SELU()
        elif name == 'tanh':
            assert params is None, 'No argument for Tanh please'
            module = nn.Tanh()
        elif name == 'gelu':
            assert params is None, 'No argument for GreLU please'
            module = nn.GELU()
        elif name == 'dropout':
            module = nn.Dropout(params)
        elif name == 'batchnorm':
            module = nn.BatchNorm1d(current_dim)
        else:
            raise NotImplementedError(f'{name} not known')

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


class ResBlock(NeuralNetwork):
    def __init__(self, in_dim, network_structure):
        network_structure.append(('linear', in_dim))
        self.network = super(ResBlock, self).__init__(in_dim, network_structure)

    def forward(self, *args, **kwargs):
        x = torch.cat(args, dim=-1)

        out = super(ResBlock, self).forward(x)

        return x + out


class Critic(NeuralNetwork):
    def __init__(self, state_dims, action_dim, network_structure):
        in_dim = int(np.sum([np.product(arg) for arg in state_dims]) + np.product(action_dim))

        super(Critic, self).__init__(in_dim, network_structure)

        dummy = super(Critic, self).forward(torch.zeros((1, in_dim)))

        self.operators.append(nn.Linear(dummy.shape[1], 1))

        self.operators.apply(init_xavier_uniform)

    def forward(self, *args):
        return super(Critic, self).forward(*args)


class Normalize(torch.nn.Module):
    def __init__(self, mean=torch.zeros(1), std=torch.ones(1)):
        super(Normalize, self).__init__()

        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.std = torch.nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class UnNormalize(torch.nn.Module):
    def __init__(self, mean=torch.zeros(1), std=torch.ones(1)):
        super(UnNormalize, self).__init__()

        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.std = torch.nn.Parameter(std, requires_grad=False)

    def forward(self, z):
        return (z * self.std) + self.mean


class Rescale(torch.nn.Module):
    def __init__(self, min_from, max_from, min_to, max_to):
        super(Rescale, self).__init__()
        m = (max_to - min_to) / (max_from - min_from)
        c = min_to - min_from * m

        self.m = torch.nn.Parameter(m, requires_grad=True)
        self.c = torch.nn.Parameter(c, requires_grad=True)

    def forward(self, x):
        return self.m * x + self.c

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return f'm={self.m.data.tolist()}, c={self.c.data.tolist()}'


class SawtoothFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, min_x, max_x, min_y, max_y):
        slope = (max_y - min_y) / (max_x - min_x)
        ctx.save_for_backward(slope)

        result = (i - min_x) % (max_x - min_x) * slope + min_y
        return result

    @staticmethod
    def backward(ctx, grad_output):
        slope, = ctx.saved_tensors
        return grad_output * slope, None, None, None, None


class Sawtooth(torch.nn.Module):
    def __init__(self, min_x=-1, max_x=1, min_y=-1, max_y=1):
        super(Sawtooth, self).__init__()

        self.min_x = nn.Parameter(torch.tensor(min_x), requires_grad=False)
        self.max_x = nn.Parameter(torch.tensor(max_x), requires_grad=False)
        self.min_y = nn.Parameter(torch.tensor(min_y), requires_grad=False)
        self.max_y = nn.Parameter(torch.tensor(max_y), requires_grad=False)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return SawtoothFunction.apply(input, self.min_x, self.max_x, self.min_y, self.max_y)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return f'min_x={self.min_x.data.tolist()}, max_x={self.max_x.data.tolist()}, min_y={self.min_y.data.tolist()}, max_y={self.max_y.data.tolist()}'


class PoseLoss(torch.nn.Module):
    def __init__(self, weight_orientation=1.):
        super(PoseLoss, self).__init__()
        self.weight_orientation = weight_orientation

    def forward(self, output, target):
        loss_position = mse_loss(output[:, -1, :3, -1], target[:, -1, :3, -1])
        # todo: use angle between quaternions (screw)
        q_output = transforms.matrix_to_quaternion(output[:, -1, :3, :3])
        q_target = transforms.matrix_to_quaternion(target[:, -1, :3, :3])

        q_diff = transforms.quaternion_multiply(q_output, transforms.quaternion_invert(q_target))

        angles = 2 * torch.atan2(torch.norm(q_diff[:, 1:]), q_diff[:, 0])

        loss_orientation = torch.mean(torch.abs(angles))
        loss = loss_position + loss_orientation

        return loss, loss_position, loss_orientation


class KinematicChainLoss(PoseLoss):
    def __init__(self, weight_matrix_positions=None, weight_matrix_orientations=None, eps=1e-7):
        super(KinematicChainLoss, self).__init__()
        if weight_matrix_positions is not None:
            self.weight_matrix_positions = torch.nn.Parameter(weight_matrix_positions, requires_grad=False)

        if weight_matrix_orientations is not None:
            self.weight_matrix_orientations = torch.nn.Parameter(weight_matrix_orientations, requires_grad=False)

        self.eps = eps

    def forward(self, output, target):

        if hasattr(self, "weight_matrix_positions"):
            distance_positions = torch.cdist(output[:, :, :3, -1].contiguous(), target[:, :, :3, -1].contiguous())
            loss_positions = torch.einsum("bxy,xy", distance_positions, self.weight_matrix_positions).mean()
        else:
            loss_positions = torch.norm(output[:, -1, :3, -1] - target[:, -1, :3, -1], p=2, dim=-1).mean()

        if hasattr(self, "weight_matrix_orientations"):
            # http://www.boris-belousov.net/2016/12/01/quat-dist/
            # R = P * Q^T
            # tr_R = R * eye(3)

            output_ = output[:, :, :3, :3]  # size: bxsx3x3
            target_ = torch.transpose(target[:, :, :3, :3], -1, -2)  # size: bxtx3x3

            tr_R = torch.einsum("bsxy,btyz,xz->bst", output_, target_,  torch.eye(3, device=output_.device))

            # calculate angle
            # add eps to make input to arccos (-1, 1) to avoid numerical instability
            # scale between 0 and 1
            distance_orientations = torch.acos(torch.clamp((tr_R - 1) / (2 + self.eps), -1 + self.eps, 1 - self.eps)) / np.pi

            loss_orientations = torch.einsum("bxy,xy", distance_orientations, self.weight_matrix_orientations).mean()
        else:
            loss_orientations = torch.norm(transforms.matrix_to_quaternion(output[:, -1, :3, :3]) -
                                           transforms.matrix_to_quaternion(target[:, -1, :3, :3]), p=2, dim=-1).mean()

        loss = loss_positions + loss_orientations

        return loss, loss_positions, loss_orientations


class WeightedPoseLoss(PoseLoss):
    def __init__(self, weight_position=1, weight_orientation=1):
        super(WeightedPoseLoss, self).__init__()
        self.weight_position = weight_position
        self.weight_orientation = weight_orientation

    def forward(self, output, target):
        _, loss_position, loss_orientation = super(WeightedPoseLoss, self).forward(output, target)
        loss = self.weight_position * loss_position + \
               self.weight_orientation * loss_orientation

        return loss, loss_position, loss_orientation


class HomoscedasticLoss(PoseLoss):
    """PAPER: Kendall, Alex, and Roberto Cipolla. "Geometric loss functions for camera pose regression with deep learning." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017."""

    def __init__(self):
        super(HomoscedasticLoss, self).__init__()
        self.s_x = torch.nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=True)
        self.s_q = torch.nn.Parameter(torch.tensor([-3], dtype=torch.float32), requires_grad=True)

    def forward(self, output, target):
        _, loss_position, loss_orientation = super(HomoscedasticLoss, self).forward(output, target)

        loss = loss_position * (-self.s_x).exp() + self.s_x + \
               loss_orientation * (-self.s_q).exp() + self.s_q

        return loss, loss_position, loss_orientation


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    network_structure = [('res_block', [('linear', 64), ('relu', None), ('dropout', 0.2), ('linear', 32)]),
                         ('linear', 64), ('relu', None), ('dropout', 0.2), ('linear', 32)]

    neural_network = NeuralNetwork(21, network_structure).to(device)

    print(neural_network)

    x = torch.rand((1, 21)).to(device)
    y = neural_network(x)

    print(y.shape)
    from copy import deepcopy

    sawtooth = SawtoothFunction.apply
    #
    x = torch.linspace(-5, 5, 100, requires_grad=True)
    x_init = deepcopy(x)
    # x = torch.range(1, 5, requires_grad=True)
    # y = sawtooth(x, -.5, .7, -1, 1.5)

    st = Sawtooth(-1, 1, -np.pi, np.pi)

    y = st(x)

    import matplotlib.pyplot as plt

    #
    y.retain_grad()
    target = torch.ones_like(y)
    # target = torch.zeros_like(y)

    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()

    plt.scatter(x.detach(), y.detach(), label="y")
    plt.figure()

    plt.scatter(x.detach(), x.grad, label="dL/dx")

    plt.show()

    plt.figure()
    optimizer = torch.optim.Adam([x], 1e-3)

    losses = []

    from tqdm import tqdm

    for step in tqdm(range(5_000)):
        optimizer.zero_grad()

        y = st(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        # plt.scatter(x_init.detach(), x.detach(), label=f"{step}")
        if step % 500:
            plt.scatter(x_init.detach(), y.detach(), label=f"{step}")
        if loss.item() < 1e-3:
            break

    plt.figure()
    plt.plot(losses)
    plt.show()

    pass
