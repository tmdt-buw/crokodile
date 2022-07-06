"""
Checks if dht model generates correct tcp pose
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from tqdm import tqdm

from models.dht import get_dht_model
from utils.nn import KinematicChainLoss

wandb_mode = "online"
# wandb_mode = "disabled"

sys.path.append(str(Path(__file__).resolve().parents[1]))

def perform_gradient_decent_ik(config=None, project=None, visualize=False, num_robots=2):
    with wandb.init(config=config, project=project, entity="bitter", mode=wandb_mode):
        config = wandb.config

        data = torch.load(config.data_file)

        dht_params = data["dht_params"]
        joint_limits = data["joint_limits"]

        dht_model = get_dht_model(dht_params, joint_limits)

        X = torch.rand(1000, len(joint_limits))
        y = dht_model(X).detach()

        weight_matrix_p = torch.zeros(len(data["dht_params"]), len(data["dht_params"]))
        weight_matrix_p[-1, -1] = 1
        weight_matrix_o = torch.zeros(len(data["dht_params"]), len(data["dht_params"]))
        weight_matrix_o[-1, -1] = 1

        loss_fn = KinematicChainLoss(weight_matrix_p, weight_matrix_o, reduction=False, verbose_output=True)

        angles_ = []
        for _ in range(config.ensemble):
            angles = torch.nn.Parameter(torch.randn_like(X) * 2 - 1)
            angles_.append(angles)
        optimizer = torch.optim.AdamW(angles_, lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=50, min_lr=1e-6)

        angles_history = torch.empty((config.epochs, *angles.shape))

        for step in tqdm(range(config.epochs)):
            angles_history[step] = angles

            losses = []
            for angles in angles_:
                poses = dht_model(angles)

                loss, loss_p, loss_o = loss_fn(poses, y)
                losses.append(loss)

            losses = torch.stack(losses)
            loss = losses.mean()
            loss_test = losses.min(dim=0)[0].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
                'loss': loss.item(),
                'loss_test': loss_test.item(),
                # 'loss_p': loss_p.item(),
                # 'loss_o': loss_o.item(),
                'lr': optimizer.param_groups[0]["lr"],
            }, step=step)

            scheduler.step(loss.item())

            if loss < 1e-3 or loss.isnan().any():
                break

        angles_history = angles_history[:step]

        torch.save({
            'angles': angles_history.cpu().detach(),
            'target': X.cpu().detach(),
        }, os.path.join(wandb.run.dir, "results.pt"))



if __name__ == '__main__':

    data_file = "../data/panda_5_10000_1000.pt"

    config = {
        "data_file": data_file,
        "lr": 1e-1,
        "epochs": 2_500,
        "ensemble": 1,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perform_gradient_decent_ik(config, project="dht_ik", visualize=True)
