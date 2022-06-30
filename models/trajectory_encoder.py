import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
import math
from torch.nn import TransformerEncoderLayer, TransformerEncoder

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nn import NeuralNetwork


class TrajectoryEncoder(nn.Module):
    """
    The TrajectoryEncoder encodes a trajectory with elements of arbitrary continuous spaces.
    """

    def __init__(self, state_dim: int, action_dim: int, behavior_dim: int, max_len: int, d_model=64, nhead=8,
                 num_layers=6, dim_feedforward=2048, dropout=0.1, **kwargs):
        super(TrajectoryEncoder, self).__init__()

        self.max_len = max_len
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_state = nn.Conv1d(state_dim, d_model, 1)
        self.encoder_action = nn.Conv1d(action_dim, d_model, 1)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation="gelu",
                                                 batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.behavior_decoder = nn.Conv1d(d_model, behavior_dim, 1)

    def forward(self, states, actions):
        states_ = self.encoder_state(states.swapdims(1, 2))
        actions_ = self.encoder_action(actions.swapdims(1, 2))
        actions_ = torch.nn.functional.pad(actions_, (0, 1), value=torch.nan)

        states_actions = torch.stack((states_, actions_), dim=-1).view(*states_.shape[:2], -1)
        states_actions = states_actions[:, :, :-1]

        # padding = self.max_len - states_actions.shape[-1]
        # states_actions = torch.nn.functional.pad(actions_, (0, padding))
        states_actions.swapdims_(1, 2)

        out = self.transformer_encoder(states_actions)

        out.swapdims_(1, 2)

        behavior = self.behavior_decoder(out).mean(-1)

        return behavior


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


states = torch.ones(2, 3, 3) * 2 - 1  # bld
actions = torch.zeros(2, 2, 2) * 2 - 1  # bmd

encoder = TrajectoryEncoder(
    state_dim=states.shape[-1],
    action_dim=actions.shape[-1],
    behavior_dim=32,
    max_len=states.shape[1] + actions.shape[1],
    d_model=2,
    nhead=2,
)

behavior = encoder(states, actions)
