import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer


class Seq2SeqTransformer(nn.Module):
    def __init__(self, max_len=10, num_domains=2, d_model=64, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, **kwargs):
        super(Seq2SeqTransformer, self).__init__()

        self.max_len = max_len

        self.encoder_src = nn.Conv1d(1, d_model, 1)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_tgt = nn.Conv1d(1, d_model, 1)

        self.domain_embedding = nn.Embedding(num_domains, d_model)

        self.encoder_sot = nn.Conv1d(3, d_model, 1)

        self.transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                       dropout, batch_first=True)

        self.decoder_out = nn.Conv1d(d_model, 1, 1)

        tgt_mask = (torch.tril(torch.ones((max_len, max_len))) == 0)
        self.register_buffer('tgt_mask', tgt_mask)

    def forward(self, src: Tensor, sot: Tensor, tgt: Tensor,
                src_domain: Tensor, tgt_domain: Tensor,
                src_key_padding_mask: Tensor = None, tgt_key_padding_mask: Tensor = None,
                ):
        src = src.unsqueeze(1)
        tgt = tgt.unsqueeze(1)
        sot = sot.unsqueeze(-1)

        src_domain = src_domain.unsqueeze(1)
        tgt_domain = tgt_domain.unsqueeze(1)

        src_nan_mask = ~src.isnan()
        torch.nan_to_num_(src)

        src_domain_emb = self.domain_embedding(src_domain)
        tgt_domain_emb = self.domain_embedding(tgt_domain)

        src_enc = self.encoder_src(src)  # NCL -> NEL
        src_enc = torch.einsum("nel,nil->nel", src_enc, src_nan_mask)
        src_enc = src_enc.swapdims(1, 2)  # NEL -> NLE
        src_emb = self.positional_encoding(src_enc)

        src_emb = torch.concat((src_domain_emb, src_emb), dim=1)

        tgt_nan_mask = ~tgt.isnan()
        torch.nan_to_num_(tgt)

        tgt_enc = self.encoder_tgt(tgt)  # tgt_enc: NCL -> NEL
        tgt_enc = torch.einsum("nel,nil->nel", tgt_enc, tgt_nan_mask)
        tgt_enc = tgt_enc.swapdims(1, 2)  # NEL -> NLE
        tgt_emb = self.positional_encoding(tgt_enc)

        sot_nan_mask = ~sot.isnan().any(1, keepdims=True)
        torch.nan_to_num_(sot)

        sot_enc = self.encoder_sot(sot)
        sot_enc = torch.einsum("nel,nil->nel", sot_enc, sot_nan_mask)
        sot_enc = sot_enc.swapdims(1, 2)  # NEL -> NLE

        sot_enc = sot_enc + tgt_domain_emb

        sot_tgt = torch.concat((sot_enc, tgt_emb), dim=1)

        # sot_tgt_emb = self.positional_encoding(sot_tgt_enc)

        out_emb = self.transformer(src_emb, sot_tgt,
                                   src_key_padding_mask=src_key_padding_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   tgt_mask=self.tgt_mask,
                                   # memory_key_padding_mask=self.memory_padding_mask
                                   )
        out_emb = out_emb.swapdims(1, 2)  # NLE -> NEL

        out = self.decoder_out(out_emb)  # NEL -> NCL

        out = out.squeeze(1)  # NL

        return out


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

    model = Seq2SeqTransformer(6, 7)

    src = torch.rand(10, 1, 6)  # NCL
    tgt = torch.rand(10, 1, 7)

    out = model(src, tgt)

    print(src.shape)
    print(out.shape)

    exit()

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
