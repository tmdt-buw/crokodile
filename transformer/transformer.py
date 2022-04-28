import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 in_dim, out_dim, number_heads, number_hidden_layers, number_layers, max_seq_length, dropout=0.):
        super(Seq2SeqTransformer, self).__init__()

        self.encoder = Encoder(in_dim, out_dim, number_heads, number_hidden_layers, number_layers, dropout)
        self.decoder = Decoder(in_dim, out_dim, number_heads, number_hidden_layers, number_layers, dropout)

        self.positional_encoding = PositionalEncoding(in_dim, max_seq_length, dropout)


    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(src)
        tgt_emb = self.positional_encoding(trg)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return outs

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, number_heads, number_hidden_layers, number_layers, dropout=0.):
        super(Encoder, self).__init__()
        self.number_layers = number_layers
        encoder_layers = TransformerEncoderLayer(in_dim, number_heads, number_hidden_layers, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, number_layers)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        x = self.transformer_encoder(src, mask, src_key_padding_mask)

        return x


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, number_heads, number_hidden_layers, number_layers, dropout=0.):
        super(Decoder, self).__init__()
        self.number_layers = number_layers
        decoder_layers = TransformerDecoderLayer(in_dim, number_heads, number_hidden_layers, dropout)
        self.decoder = TransformerDecoder(decoder_layers, number_layers)

    def forward(self, trg, e_outputs):
        x = self.decoder(x, e_outputs)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5, dropout=0.):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_div_term = position * div_term
        pe[:, 0::2] = torch.sin(pos_div_term[:, :(d_model + 2) // 2])
        pe[:, 1::2] = torch.cos(pos_div_term[:, :d_model // 2])
        pe = pe.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


if __name__ == '__main__':
    from trajectory_encoder import TrajectoryEncoder, generate_trajectory
    import numpy as np

    max_seq_length = 9
    seq_length = 5

    dims = [11, 22]
    input_embedding_dim = 16
    embedding_dim = 32

    batch_size = 64

    lr = 1e-1

    encoder_structs = [[], []]

    te = TrajectoryEncoder(dims, encoder_structs, input_embedding_dim)

    # generate trajectory encodings
    trajectory_encodings = []

    for _ in range(batch_size):
        types = [2] + [0, 1] * np.random.randint(1, max_seq_length // 2 - 1) + [1]
        types += [3] * (max_seq_length - len(types))
        types = torch.tensor(types)
        trajectory = generate_trajectory(dims + [None, None], types)

        input_encoding = te(trajectory, types)

        trajectory_encodings.append(input_encoding)

    trajectory_encodings = torch.stack(trajectory_encodings)
    print(trajectory_encodings.shape)

    encoder = Encoder(input_embedding_dim, embedding_dim, 2, 128, 2, max_seq_length, 0)
    encoding = encoder(trajectory_encodings)
    print(encoding.shape)

    decoder = Decoder(input_embedding_dim, embedding_dim, 2, 128, 2, max_seq_length, 0)
    decoding = decoder(trajectory_encodings, encoding)

    print(decoding.shape)

    exit()

    optimizer = torch.optim.Adam(te.parameters(), lr=lr)
    loss_function = torch.nn.MSELoss()

    for _ in range(1_000):

        trajectory_batch = []
        target = torch.ones((batch_size, max_seq_length, input_dim))

        for _ in range(batch_size):
            types = torch.tensor([2, 0, 1, 0, 1, 0, 3, 3, 3])

            trajectory = generate_trajectory([state_dim, action_dim, None, None], types)

            trajectory_encoding = te(trajectory, types)

            trajectory_batch.append(trajectory_encoding)

        trajectory_batch = torch.stack(trajectory_batch)

        encoder = Encoder(8, 4, 2, 128, 2, max_seq_length, 0)
        decoder = Decoder(6, 6, 2, 128, 2, max_seq_length, 0)

        loss = loss_function(trajectory_batch, target)
        print(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # print("Trajectory encoding", trajectory_encoding.shape)
