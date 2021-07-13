import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear


class Flatten(torch.nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class DH_Transformation(torch.nn.Module):
    THETA_COS = torch.tensor([[1, 0, 0, 1],
                              [0, 1, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]], requires_grad=False)

    THETA_SIN = torch.tensor([[0, -1, 1, 0],
                              [1, 0, 0, 1],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]], requires_grad=False)

    ALPHA_COS = torch.tensor([[0, 1, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 0]], requires_grad=False)

    ALPHA_SIN = torch.tensor([[0, 0, 1, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 0]], requires_grad=False)

    A = torch.tensor([[0, 0, 0, 1],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]], requires_grad=False)

    D = torch.tensor([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]], requires_grad=False)

    CONST = torch.tensor([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]], requires_grad=False)

    def __init__(self, theta=None, d=None, a=None, alpha=None, mode=None):
        super(DH_Transformation, self).__init__()

        if theta is None:
            self.theta = Linear(1, 1)
        else:
            self.theta = theta

        if d is None:
            self.d = Linear(1, 1)
        else:
            self.d = d

        # mode defines, if joint is rotational or translational
        self.mode = mode
        if mode is None:
            self.parameter_attribution = Variable(torch.randn(2),
                                                  requires_grad=True)
        elif mode is "rot":
            self.parameter_attribution = Variable(torch.FloatTensor([1e6, 0]),
                                                  requires_grad=False)
        elif mode is "trans":
            self.parameter_attribution = Variable(torch.FloatTensor([0, 1e6]),
                                                  requires_grad=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if a is None:
            self.a = Variable(torch.randn(1), requires_grad=True)
        else:
            self.a = Variable(torch.tensor(a), requires_grad=False)

        if alpha is None:
            self.alpha = Variable(torch.randn(1), requires_grad=True)
        else:
            self.alpha = Variable(torch.tensor(alpha), requires_grad=False)

    def forward(self, pose_init, param):

        action_rottrans = param.unsqueeze(-1).repeat(1,2) * self.parameter_attribution.softmax(-1)

        theta = self.theta(action_rottrans[:, :1])
        d = self.d(action_rottrans[:, 1:])

        rot1 =



        theta_cos = theta[:,None] * self.THETA_COS
        theta_sin = theta[:,None] * self.THETA_SIN

        d = d[:,None] * self.D

        alpha_cos = self.alpha[:,None] * self.ALPHA_COS
        alpha_sin = self.alpha[:,None] * self.ALPHA_SIN

        a = self.a[:,None] * self.A


        transformation = theta_cos * theta_sin * alpha_cos * alpha_sin * a * d + self.CONST

        return torch.sigmoid(d) * (self.max - self.min) + self.min


tf = DH_Transformation(mode="rot")

poses_init = torch.eye(4).unsqueeze(-1).repeat(1,1,5)
params = torch.tensor([-1,-.5,0,.5,1])

print(tf(poses_init, params))

exit()


def init_xavier_uniform(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.xavier_uniform_(m.weight)


class DH_Model(nn.Module):
    def __init__(self, config=tuple([None] * 5)):
        super(DH_Model, self).__init__()

        self.operators = []

        for transformation_config in config:
            if transformation_config is None:
                transformation = None

        current_layer_size = in_dim

        for layer, params in network_structure:
            if layer == 'linear':
                self.operators.append(nn.Linear(current_layer_size, params))
                current_layer_size = params
            elif layer == 'relu':
                assert params is None, 'No argument for ReLU please'
                self.operators.append(nn.ReLU())
            elif layer == 'selu':
                assert params is None, 'No argument for SeLU please'
                self.operators.append(nn.SELU())
            elif layer == 'tanh':
                assert params is None, 'No argument for Tanh please'
                self.operators.append(nn.Tanh())
            elif layer == 'sigmoid':
                assert params is None, 'No argument for Sigmoid please'
                self.operators.append(nn.Sigmoid())
            elif layer == 'gelu':
                assert params is None, 'No argument for GreLU please'
                self.operators.append(nn.GELU())
            elif layer == 'dropout':
                self.operators.append(nn.Dropout(params))
            elif layer == 'batchnorm':
                self.operators.append(nn.BatchNorm1d(current_layer_size))
            else:
                raise NotImplementedError(f'{layer} not known')

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

    def get_activations(self, state):
        x = state

        activations = []

        for operator in self.operators:
            x = operator(x)

            if type(operator) == nn.ReLU:
                activations.append(x)

        return activations


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    pol_struct = [('linear', 64), ('relu', None), ('dropout', 0.2),
                  ('linear', 32)]

    neural_network = NeuralNetwork([21], [7], pol_struct).to(device)

    print(neural_network)

    print(neural_network(torch.rand((1, 21)).to(device)))

    policy = Policy([21], [7], pol_struct).to(device)

    print(policy(torch.rand((1, 21)).to(device)))
    print(policy)

    val_struct = [('linear', 32), ('relu', None), ('dropout', 0.2),
                  ('linear', 32)]

    # val = Critic([21], [7], val_struct).to(device)

    # print(val.operators)
