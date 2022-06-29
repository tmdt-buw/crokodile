"""
Verify differentiability of dht module by performing inverse kinematics.
"""

import os

import numpy as np
import torch
from torch.multiprocessing import Process, set_start_method
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import wandb
from models.dht import get_dht_model
from utils.nn import NeuralNetwork, KinematicChainLoss

try:
    set_start_method('spawn')
except RuntimeError:
    pass

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def validate_loss_fn():
    # desired loss behavior
    def tcp_loss(poses, y):
        return torch.norm(poses[:, -1, :3, -1] - y[:, -1, :3, -1], dim=-1).mean()

    x_dim, y_dim = 4, 5

    weight_matrix_p = torch.zeros(x_dim, y_dim)
    weight_matrix_p[-1, -1] = 1
    weight_matrix_o = torch.zeros(x_dim, y_dim)

    kinematic_chain_loss_tcp_ = KinematicChainLoss(weight_matrix_p, weight_matrix_o)

    def kinematic_chain_loss_tcp(poses, y):
        loss, _, _ = kinematic_chain_loss_tcp_(poses, y)
        return loss

    x_data = torch.randn(1000, x_dim, 4, 4)
    y_data = torch.randn(1000, y_dim, 4, 4)

    loss1 = tcp_loss(x_data, y_data)
    loss2 = kinematic_chain_loss_tcp(x_data, y_data)

    assert torch.isclose(loss1, loss2)


if __name__ == '__main__':
    validate_loss_fn()