import math

import torch
from click.core import batch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Seq2SeqTransformerEncoder(nn.Module):
    def __init__(
        self, src_len, tgt_len, d_model=64, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, **kwargs
    ):
        super(Seq2SeqTransformerEncoder, self).__init__()

        self.src_len = src_len
        self.tgt_len = tgt_len
        self.max_len = max(src_len, tgt_len)

        self.encoder_src = nn.Conv1d(1, d_model, 1)
        self.positional_encoding = PositionalEncoding(d_model, self.max_len, dropout)

        self.encoder_tgt = nn.Conv1d(1, d_model, 1)

        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)

        self.transformer_encoder = TransformerEncoder(layer, num_encoder_layers)

        self.decoder_out = nn.Conv1d(d_model, 1, 1)

        src_key_padding_mask = torch.zeros(1, self.max_len)
        src_key_padding_mask[:, src_len:] = 1.0

        self.register_buffer("src_key_padding_mask", src_key_padding_mask)

    def forward(self, src: Tensor):
        src = torch.nn.functional.pad(src, (0, self.max_len - self.src_len), mode="constant", value=torch.nan)

        src = src.unsqueeze(1)

        src_enc = self.encoder_src(src)  # NCL -> NEL
        src_enc = src_enc.swapdims(1, 2)  # NEL -> NLE
        src_emb = self.positional_encoding(src_enc)
        torch.nan_to_num_(src_emb)

        out_emb = self.transformer_encoder(
            src_enc,
            # src_key_padding_mask=self.src_key_padding_mask.repeat(src.shape[0],1)
        )
        out_emb = out_emb.swapdims(1, 2)  # NLE -> NEL

        out = self.decoder_out(out_emb)  # NEL -> NCL

        out = out.squeeze(1)  # NL

        out = out[:, : self.tgt_len]

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5, dropout=0.0):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_div_term = position * div_term
        pe[:, 0::2] = torch.sin(pos_div_term[:, : (d_model + 2) // 2])
        pe[:, 1::2] = torch.cos(pos_div_term[:, : d_model // 2])
        pe = pe.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


if __name__ == "__main__":

    model = Seq2SeqTransformerEncoder(6, 7)

    src = torch.rand(10, 1, 6)  # NCL
    tgt = torch.rand(10, 1, 7)

    out = model(src, tgt)

    print(src.shape)
    print(out.shape)
