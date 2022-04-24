import torch
import torch.nn as nn
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nn import NeuralNetwork


class TrajectoryDecoder(nn.Module):
    """
    The TrajectoryEncoder encodes a trajectory with elements of arbitrary continuous spaces.
    """
    def __init__(self, type_dims: list, decoder_structs: list, d_model: int):
        """
        :param type_dims: dimensions of spaces. Spaces are enumerated starting with 1.
        :param decoder_structs: structure of encoder neural networks to be trained for each space.
        Two tokens are introduced: BOT (beginning of trajectory) and EOT (end of trajectory).
        """
        super(TrajectoryDecoder, self).__init__()

        self.d_model = d_model
        self.decoders = {}

        token_id = 0

        for idecoder, (type_dim, struct) in enumerate(zip(type_dims, decoder_structs)):
            if type_dim is not None:
                struct += [('linear', type_dim)]
                self.decoders[idecoder] = NeuralNetwork(d_model, struct)
            else:
                self.decoders[idecoder] = token_id
                token_id += 1

        if token_id:
            self.token_embedding = nn.Embedding(token_id, d_model)

    def forward(self, x, types: torch.Tensor):
        """
        Encode a trajectory

        :param x: Trajectory
        :param types: List
        :return:
        """
        decodings = [None] * x.shape[0]

        for type_id, translator in self.decoders.items():
            type_mask = types == type_id
            if type_mask.any():
                if type(translator) == int:
                    decoding = [[None] * type_mask.sum()]
                else:
                    decoding = translator(x[type_mask])

                decodings.append(decoding)
                permutation.append(type_mask.nonzero())

        permutation = torch.cat(permutation).squeeze()

        trajectory_encoding = torch.cat(encodings)
        trajectory_encoding = trajectory_encoding[torch.argsort(permutation), :]

        return trajectory_encoding


def generate_trajectory(dims, mask, device=None):
    trajectory = []

    for item in mask:
        dim = dims[item]

        if dim is None:
            trajectory.append(None)
        else:
            trajectory.append(torch.rand(1, dim, device=device))

    return trajectory


if __name__ == '__main__':
    from trajectory_encoder import TrajectoryEncoder

    max_seq_length = 9
    seq_length = 5

    dims = [11, 22] + [None, None]
    embedding_dim = 15

    batch_size = 64

    lr = 1e-1

    encoder_structs = [[], []] + [None, None]

    encoder = TrajectoryEncoder(dims, encoder_structs, embedding_dim)

    types_in = torch.tensor([2, 0, 1, 0, 1, 0, 3, 3, 3])
    trajectories = generate_trajectory(dims, types_in)

    print([tuple(tt.shape) if tt is not None else -1 for tt in trajectories])

    input_encoding = encoder(trajectories, types_in)

    ## decode
    decoder_structs = [[], []] + [None, None]
    types_out = types_in

    decoder = TrajectoryDecoder(dims, decoder_structs, embedding_dim)
    output = decoder(input_encoding, types_out)

    assert input_encoding.shape == torch.Size((max_seq_length, embedding_dim))

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    loss_function = torch.nn.MSELoss()

    target = torch.ones((max_seq_length, embedding_dim))

    print(input_encoding.shape)
    print(target.shape)

    losses = []

    for _ in range(10):

        input_encoding = encoder(trajectories, types_in)

        loss = loss_function(input_encoding, target)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    losses = torch.tensor(losses)

    assert (losses[:-1] > losses[1:]).all() # continuous decrease of loss