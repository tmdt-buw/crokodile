import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from environments.environments_robot_task.robots import get_robot

from config import *

import pybullet as p
from models.dht import get_dht_model
from torch.utils.data import DataLoader, TensorDataset
import wandb
from utils.nn import KinematicChainLoss

if __name__ == '__main__':
    # with wandb.init(mode="disabled"):
    with wandb.init():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_file_A = "panda_5_10000_1000.pt"
        data_file_B = "ur5_5_10000_1000.pt"

        data_file_out = "panda-ur5_5_10000_1000.pt"

        data_A = torch.load(os.path.join(data_folder, data_file_A))
        data_B = torch.load(os.path.join(data_folder, data_file_B))

        weight_matrix_p = torch.zeros(len(data_A["dht_params"]), len(data_B["dht_params"]))
        weight_matrix_p[-1, -1] = 1
        weight_matrix_o = torch.zeros(len(data_A["dht_params"]), len(data_B["dht_params"]))
        weight_matrix_o[-1, -1] = 1

        loss_fn = KinematicChainLoss(weight_matrix_p, weight_matrix_o, reduction=False).to(device)

        dht_model_A = get_dht_model(data_A["dht_params"], data_A["joint_limits"])
        dht_model_B = get_dht_model(data_B["dht_params"], data_B["joint_limits"]).to(device)

        batch_size = 1000
        seeds = 5
        ik_steps = 1000

        states_A = data_A["trajectories_states_train"].reshape(-1, data_A["trajectories_states_train"].shape[-1])

        # states_A = states_A[:10]
        # states_B = data_B["trajectories_states_train"].reshape(-1, data_B["trajectories_states_train"].shape[-1])
        link_poses_A = dht_model_A(states_A).detach()

        states_B = []

        for batch_id, (link_poses_A_,) in tqdm(enumerate(DataLoader(TensorDataset(link_poses_A), batch_size=batch_size, shuffle=False))):
            link_poses_A_ = link_poses_A_.to(device).repeat_interleave(seeds, dim=0)

            states_B_ = torch.rand(link_poses_A_.shape[0], *data_B["trajectories_states_train"].shape[2:]).to(device)
            states_B_ = states_B_ * 2 - 1
            states_B_ = torch.nn.Parameter(states_B_)

            optimizer = torch.optim.AdamW([states_B_])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=50, min_lr=1e-6)

            for step in range(ik_steps):
                link_poses_B_ = dht_model_B(states_B_)
                loss = loss_fn(link_poses_A_, link_poses_B_)
                loss = loss.reshape(-1, seeds)

                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

                wandb.log({
                    'loss_mean': loss.mean().item(),
                    'loss_best': loss.min(dim=0)[0].mean().item(),
                    'lr': optimizer.param_groups[0]["lr"],
                }, step=step + batch_id * ik_steps)

            states_B_ = states_B_.reshape(-1, seeds, *states_B_.shape[1:])
            states_B_ = states_B_[range(states_B_.shape[0]), loss.argmin(1)]
            states_B.append(states_B_)

        states_B = torch.concat(states_B)

        link_poses_B = dht_model_B(states_B)

        # loss = loss_fn(link_poses_A, link_poses_B)

        # print(loss.mean())

        # p.connect(p.GUI)
        #
        # robot = get_robot({
        #     "name": "ur5",
        #     **{"name": "ur5", "sim_time": .1, "scale": .1}
        # }, bullet_client=p)
        #
        # robot.reset({"arm": {"joint_positions": states_B[0].cpu().detach().numpy()}})


        data = {
            "A": data_A,
            "B": data_B,
        }

        data["A"]["corresponding_states"] = states_B.cpu().detach()

        torch.save(
            data,
            os.path.join(data_folder, data_file_out)
        )
