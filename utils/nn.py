import numpy as np
import torch
import torch.nn as nn

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


def get_weight_matrices(
    link_positions_X, link_positions_Y, weight_matrix_exponent_p, norm=True
):
    """
    Generate weights based on distances between relative positions of robot links

    Params:
        link_positions_{X, Y}: Positions of both robots in 3D space.
        weight_matrix_exponent_p: Parameter used to shape the position weight matrix by emphasizing similarity.
            weight = exp(-weight_matrix_exponent_p * distance)

    Returns:
        weight_matrix_XY_p: Weight matrix for positions
        weight_matrix_XY_p: Weight matrix for orientations. All zeros except weight which corresponds to the end effectors.
    """

    link_positions_X = torch.cat(
        (torch.zeros(1, 3).to(link_positions_X), link_positions_X)
    )
    link_positions_Y = torch.cat(
        (torch.zeros(1, 3).to(link_positions_Y), link_positions_Y)
    )

    link_lenghts_X = torch.norm(
        link_positions_X[1:] - link_positions_X[:-1], p=2, dim=-1
    )
    link_lenghts_Y = torch.norm(
        link_positions_Y[1:] - link_positions_Y[:-1], p=2, dim=-1
    )

    link_order_X = link_lenghts_X.cumsum(0)
    link_order_X = link_order_X / link_order_X[-1]

    link_order_Y = link_lenghts_Y.cumsum(0)
    link_order_Y = link_order_Y / link_order_Y[-1]

    weight_matrix_XY_p = torch.exp(
        -weight_matrix_exponent_p
        * torch.cdist(link_order_X.unsqueeze(-1), link_order_Y.unsqueeze(-1))
    )
    weight_matrix_XY_p = torch.nan_to_num(weight_matrix_XY_p, 1.0)

    weight_matrix_XY_o = torch.zeros_like(weight_matrix_XY_p)
    weight_matrix_XY_o[-1, -1] = 1

    if norm:
        weight_matrix_XY_p = weight_matrix_XY_p.softmax(0)
        # weight_matrix_XY_o = weight_matrix_XY_o.softmax(0)

    return weight_matrix_XY_p, weight_matrix_XY_o


def create_network(
    in_dim,
    out_dim,
    network_width,
    network_depth,
    dropout,
    out_activation=None,
    **kwargs,
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


class ResBlock(NeuralNetwork):
    def __init__(self, in_dim, network_structure):
        network_structure.append(("linear", in_dim))
        self.network = super(ResBlock, self).__init__(in_dim, network_structure)

    def forward(self, *args, **kwargs):
        x = torch.cat(args, dim=-1)

        out = super(ResBlock, self).forward(x)

        return x + out


class Critic(NeuralNetwork):
    def __init__(self, state_dims, action_dim, network_structure):
        in_dim = int(
            np.sum([np.product(arg) for arg in state_dims]) + np.product(action_dim)
        )

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

        self.m = torch.nn.Parameter(m, requires_grad=False)
        self.c = torch.nn.Parameter(c, requires_grad=False)

    def forward(self, x):
        return self.m * x + self.c

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return f"m={self.m.data.tolist()}, c={self.c.data.tolist()}"


class SawtoothFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, min_x, max_x, min_y, max_y):
        slope = (max_y - min_y) / (max_x - min_x)
        ctx.save_for_backward(slope)

        result = (i - min_x) % (max_x - min_x) * slope + min_y
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (slope,) = ctx.saved_tensors
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
        return SawtoothFunction.apply(
            input, self.min_x, self.max_x, self.min_y, self.max_y
        )

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return f"min_x={self.min_x.data.tolist()}, max_x={self.max_x.data.tolist()}, min_y={self.min_y.data.tolist()}, max_y={self.max_y.data.tolist()}"


class Pos2Pose(nn.Module):
    def forward(self, x):
        positions = x.reshape(x.shape[0], -1, 3, 1)

        orientations = torch.eye(3, device=x.device).expand(
            positions.shape[0], positions.shape[1], 3, 3
        )

        poses = torch.cat((orientations, positions), dim=-1)

        pad = torch.tensor([0, 0, 0, 1], device=x.device)
        pad = pad.expand(positions.shape[0], positions.shape[1], 1, 4)

        out = torch.cat((poses, pad), dim=-2)

        return out


class KinematicChainLoss(torch.nn.Module):
    def __init__(
        self,
        weight_matrix_positions,
        weight_matrix_orientations,
        reduction=True,
        verbose_output=False,
        eps=1e-7,
    ):
        super(KinematicChainLoss, self).__init__()

        self.weight_matrix_positions = torch.nn.Parameter(
            weight_matrix_positions, requires_grad=False
        )

        self.weight_matrix_orientations = torch.nn.Parameter(
            weight_matrix_orientations, requires_grad=False
        )

        self.reduction = reduction
        self.verbose_output = verbose_output
        self.eps = eps

    def forward(self, X, Y):
        """
        - X: :math:`(B, S, M, 4, 4)`
        - Y: :math:`(B, T, N, 4, 4)`

        - Y: :math:`(B, S, T)`
        """

        if len(X.shape) == 4:
            # S == 1
            X = X.unsqueeze(1)

        if len(Y.shape) == 4:
            # T == 1
            Y = Y.unsqueeze(1)

        assert len(X.shape) == 5 and len(Y.shape) == 5

        s, m = X.shape[1:3]
        t, n = Y.shape[1:3]

        # b(s*n)44
        X = X.view(X.shape[0], -1, *X.shape[-2:])
        Y = Y.view(Y.shape[0], -1, *Y.shape[-2:])

        # compute position loss
        # b(s*n)(t*m)
        distance_positions = torch.cdist(
            X[..., :3, -1].contiguous(), Y[..., :3, -1].contiguous()
        )
        distance_positions = distance_positions.view(-1, s, m, t, n)

        # todo: make scaling a parameter
        # distance_positions = 1 - torch.exp(-10 * distance_positions)

        loss_positions = torch.einsum(
            "bsmtn,mn->bst", distance_positions, self.weight_matrix_positions
        )

        # compute orientation loss

        # http://www.boris-belousov.net/2016/12/01/quat-dist/
        # R = P * Q^T
        # tr_R = R * eye(3)
        tr_R = torch.einsum(
            "bsxy,btyz,xz->bst",
            X[..., :3, :3],
            torch.transpose(Y[..., :3, :3], -1, -2),
            torch.eye(3, device=X.device),
        )

        # calculate angle
        # add eps to make input to arccos (-1, 1) to avoid numerical instability
        # scale between 0 and 1
        # todo is acos (0,pi) or (-pi,pi)?
        distance_orientations = (
            torch.acos(
                torch.clamp((tr_R - 1) / (2 + self.eps), -1 + self.eps, 1 - self.eps)
            )
            / np.pi
        )
        distance_orientations = distance_orientations.view(-1, s, m, t, n)

        loss_orientations = torch.einsum(
            "bsmtn,mn->bst",
            distance_orientations,
            self.weight_matrix_orientations,
        )

        loss = loss_positions + loss_orientations

        if self.reduction:
            loss = loss.mean()
            loss_positions = loss_positions.mean()
            loss_orientations = loss_orientations.mean()

        if self.verbose_output:
            return loss, loss_positions, loss_orientations
        else:
            return loss


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    network_structure = [
        (
            "res_block",
            [("linear", 64), ("relu", None), ("dropout", 0.2), ("linear", 32)],
        ),
        ("linear", 64),
        ("relu", None),
        ("dropout", 0.2),
        ("linear", 32),
    ]

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
