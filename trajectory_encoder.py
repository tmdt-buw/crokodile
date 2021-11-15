import torch
import torch.nn as nn

from utils.nn import NeuralNetwork


class TrajectoryEncoder(nn.Module):
    """
    The TrajectoryEncoder encodes a trajectory with elements of arbitrary continuous spaces.
    """
    def __init__(self, type_dims: list, encoder_structs: list, d_model: int):
        """
        :param type_dims: dimensions of spaces. Spaces are enumerated starting with 1.
        :param encoder_structs: structure of encoder neural networks to be trained for each space.
        Two tokens are introduced: BOT (beginning of trajectory) and EOT (end of trajectory).
        """
        super(TrajectoryEncoder, self).__init__()

        encoder_structs = [struct + [('linear', d_model)] for struct in encoder_structs]

        self.d_model = d_model
        self.translators = nn.ModuleList(
            [NeuralNetwork(type_dim, struct) for type_dim, struct in zip(type_dims, encoder_structs)])

        self.id_bot = len(self.translators)
        self.id_eot = self.id_bot + 1
        self.token_embedding = nn.Embedding(2, d_model)

    def forward(self, x, mask: list):
        """
        Encode a trajectory

        :param x: Trajectory
        :param mask: List
        :return:
        """
        encodings = []
        permutation = []

        for type_id, translator in enumerate(self.translators):
            type_mask = mask == type_id
            if type_mask.any():
                type_batch = torch.stack([x[idx] for idx in torch.where(type_mask)[0]])
                type_encoding = translator(type_batch)

                encodings.append(type_encoding)
                permutation.append(type_mask.nonzero())

        mask_bos = mask == self.id_bot

        if mask_bos.any():
            bos_encoding = self.token_embedding(torch.IntTensor([0] * mask_bos.sum()))

            encodings.append(bos_encoding)
            permutation.append(mask_bos.nonzero())

        mask_eos = mask == self.id_eot

        if mask_eos.any():
            eos_encoding = self.token_embedding(torch.IntTensor([1] * mask_eos.sum()))

            encodings.append(eos_encoding)
            permutation.append(mask_eos.nonzero())

        permutation = torch.cat(permutation).squeeze()

        trajectory_encoding = torch.cat(encodings)
        trajectory_encoding = trajectory_encoding[permutation, :]

        return trajectory_encoding


def generate_trajectory(dims, mask, amt_trajectories=1):
    trajectory = []

    for item in mask:
        dim = dims[item]

        if dim is None:
            trajectory.append(None)
        else:
            trajectory.append(torch.rand(amt_trajectories, dim))

    return trajectory


if __name__ == '__main__':
    max_seq_length = 9
    seq_length = 5

    dims = [11, 22]
    input_dim = 15

    batch_size = 64

    lr = 1e-1

    encoder_structs = [[], []]

    te = TrajectoryEncoder(dims, encoder_structs, input_dim)

    types = torch.tensor([2, 0, 1, 0, 1, 0, 3, 3, 3])
    trajectories = generate_trajectory(dims + [None, None], types, amt_trajectories=batch_size)

    [print(tt.shape) for tt in trajectories if tt is not None]

    input_encoding = te(trajectories, types)

    exit()

    optimizer = torch.optim.Adam(te.parameters(), lr=lr)
    loss_function = torch.nn.MSELoss()

    for _ in range(10):
        types = torch.tensor([2, 0, 1, 0, 1, 0, 3, 3, 3])
        trajectory = generate_trajectory([state_dim, action_dim, None, None], types, 32)
        target = torch.ones((batch_size, max_seq_length, input_dim))

        input_encoding = te(trajectory, types)

        # print(input_encoding)
        print(input_encoding.shape)

        loss = loss_function(input_encoding, target)
        print(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()