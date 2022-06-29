"""
Train state mappings with dht models.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.multiprocessing import Process, set_start_method, cpu_count
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nn import NeuralNetwork, KinematicChainLoss
from models.dht import get_dht_model

try:
    set_start_method('spawn')
except RuntimeError:
    pass

wandb_mode = "online"

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
state_mapper_file = "model.pt"


wandb_mode = "disabled"

def create_network(in_dim, out_dim, network_width, network_depth, dropout):
    network_structure = [('linear', network_width), ('relu', None),
                         ('dropout', dropout)] * network_depth
    network_structure.append(('linear', out_dim))

    return NeuralNetwork(in_dim, network_structure)


def train_action_mapping(config=None, project=None):
    print(device)

    # Initialize a new wandb run
    with wandb.init(config=config, project=project, mode=wandb_mode):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        data_file_A, data_file_B = config.data_files

        data_path_A = os.path.join(data_folder, data_file_A)
        data_path_B = os.path.join(data_folder, data_file_B)

        data_A = torch.load(data_path_A)
        data_B = torch.load(data_path_B)

        states_train = data_A["states_train"]
        states_test = data_A["states_test"]

        dht_model_A = get_dht_model(data_A["dht_params"], data_A["joint_limits"])
        dht_model_B = get_dht_model(data_B["dht_params"], data_B["joint_limits"])

        link_poses_train = dht_model_A(states_train).detach()
        link_poses_test = dht_model_A(states_test).detach()

        states_test = states_test.to(device)
        link_poses_test = link_poses_test.to(device)
        dht_model_B = dht_model_B.to(device)

        loader_train = DataLoader(TensorDataset(states_train, link_poses_train), batch_size=config.batch_size,
                                  shuffle=True)

        link_poses_A = dht_model_A(torch.zeros((1, *data_A["states_train"].shape[1:]))).to(device)
        link_poses_B = dht_model_B(torch.zeros((1, *data_B["states_train"].shape[1:]), device=device))

        link_positions_A = link_poses_A[0, :, :3, -1]
        link_positions_A = torch.cat((torch.zeros(1, 3, device=device), link_positions_A))
        link_lenghts_A = torch.norm(link_positions_A[1:] - link_positions_A[:-1], p=2, dim=-1)
        link_order_A = link_lenghts_A.cumsum(0)
        link_order_A = link_order_A / link_order_A[-1]

        link_positions_B = link_poses_B[0, :, :3, -1]
        link_positions_B = torch.cat((torch.zeros(1, 3, device=device), link_positions_B))
        link_lenghts_B = torch.norm(link_positions_B[1:] - link_positions_B[:-1], p=2, dim=-1)
        link_order_B = link_lenghts_B.cumsum(0)
        link_order_B = link_order_B / link_order_B[-1]

        weight_matrix_exponent = config.get("weight_matrix_exponent", torch.inf)

        weight_matrix_p = torch.exp(
            -weight_matrix_exponent * torch.cdist(link_order_B.unsqueeze(-1), link_order_A.unsqueeze(-1)))
        weight_matrix_p = torch.nan_to_num(weight_matrix_p, 1.)

        weight_matrix_o = torch.zeros(len(data_B["dht_params"]), len(data_A["dht_params"]))
        weight_matrix_o[-1, -1] = 1

        loss_fn = KinematicChainLoss(weight_matrix_p, weight_matrix_o).to(device)

        wandb.config.update({
            "loss_fn": str(loss_fn)
        })

        action_mapper_AB_config = {
            "in_dim": len(data_A["joint_limits"]),
            "out_dim": len(data_B["joint_limits"]),
            "network_width": config.network_width,
            "network_depth": config.network_depth,
            "dropout": config.dropout
        }

        action_mapper_AB = create_network(
            **action_mapper_AB_config
        ).to(device)

        state_mapper_AB_data = torch.load(os.path.join(data_folder, config.state_mapper_AB))

        if config.optimizer == "sgd":
            optimizer = torch.optim.SGD(action_mapper_AB.parameters(), lr=config.lr, momentum=0.9)
        elif config.optimizer == "adam":
            optimizer = torch.optim.Adam(action_mapper_AB.parameters(), lr=config.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.95, patience=200, min_lr=1e-5)

        best_loss = np.inf
        patience = config.get("patience", np.inf)

        for epoch in tqdm(range(config.epochs)):
            action_mapper_AB.train()
            for states, link_poses in loader_train:
                states, link_poses = states.to(device), link_poses.to(device)
                optimizer.zero_grad()

                states_target = action_mapper_AB(states)
                link_poses_target = dht_model_B(states_target)

                loss, loss_p, loss_o = loss_fn(link_poses_target, link_poses)

                # todo: integrate valid state discriminator (Seminar Elias / Mark)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                action_mapper_AB.eval()
                states_target = action_mapper_AB(states_test)
                link_poses_target = dht_model_B(states_target)

                loss, loss_p, loss_o = loss_fn(link_poses_target, link_poses_test)

                error_tcp_p = torch.norm(link_poses_target[:, -1, :3, -1] - link_poses_test[:, -1, :3, -1],
                                         p=2, dim=-1).mean()

                wandb.log({
                    'loss': loss.item(),
                    'loss_p': loss_p.item(),
                    'loss_o': loss_o.item(),
                    'error_tcp_p': error_tcp_p.item(),
                    'lr': optimizer.param_groups[0]["lr"],
                }, step=epoch)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save({
                        'model_config': action_mapper_AB_config,
                        'model_state_dict': action_mapper_AB.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(wandb.run.dir, "model.pt"))
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
                    if steps_since_improvement > patience:
                        break

            scheduler.step(loss.item())

            if loss < 1e-3 or loss.isnan().any():
                break


def launch_agent(sweep_id, device_id, count):
    global device
    device = device_id
    wandb.agent(sweep_id, function=train_action_mapping, count=count)


if __name__ == '__main__':
    data_file_A = "panda_10000_1000.pt"
    data_file_B = "ur5_10000_1000.pt"

    state_mapper_AB = "state_mapper_panda_ur5.pt"
    dynamics_B = "dynamics_ur5.pt"

    project = "action_mapper"

    # wandb.login()

    sweep = False

    if sweep:
        # sweep_id = None
        sweep_id = "robot2robot/state_mapper/uv04xo6w"
        runs_per_agent = 10

        if sweep_id is None:
            sweep_config = {
                # "name": "first-sweep",
                "method": "bayes",
                'metric': {
                    'name': 'loss',
                    'goal': 'minimize'
                },
                "parameters": {
                    "data_files": {
                        "value": (data_file_A, data_file_B)
                    },
                    "weight_matrix_exponent": {
                        "value": 10
                    },
                    "network_width": {
                        "min": 64,
                        "max": 2048
                    },
                    "network_depth": {
                        "min": 2,
                        "max": 8
                    },
                    "dropout": {
                        "min": 0.,
                        "max": .3
                    },
                    "optimizer": {
                        "values": ["sgd", "adam"]
                    },
                    "lr": {
                        "min": 0.0001,
                        "max": 0.01
                    },
                    "epochs": {
                        "value": 2000
                    },
                    "patience": {
                        "value": 200
                    },
                    "batch_size": {
                        "value": 1024
                        # "values": [32, 64, 128, 256, 512, 1024]
                    }
                }
            }

            sweep_id = wandb.sweep(sweep_config, project=project)

        processes = []

        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                process = Process(target=launch_agent, args=(sweep_id, device_id, runs_per_agent))
                process.start()
                processes.append(process)
        else:
            for device_id in range(cpu_count() // 2):
                process = Process(target=launch_agent, args=(sweep_id, "cpu", runs_per_agent))
                process.start()
                processes.append(process)

        for process in processes:
            process.join()

    else:
        config = {
            "data_files": (data_file_A, data_file_B),
            "state_mapper_AB": state_mapper_AB,
            "dynamics_B": dynamics_B,
            "network_width": 512,
            "network_depth": 4,
            "dropout": 0.,
            "optimizer": "adam",
            "lr": 3e-4,
            "epochs": 10_000,
            "batch_size": 128,
            "weight_matrix_exponent": 10,
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_action_mapping(config, project=project)
