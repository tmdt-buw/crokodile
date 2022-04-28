"""
Train state mappings with dht models.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.multiprocessing import Process, set_start_method
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nn import NeuralNetwork, KinematicChainLoss
from utils.dht import get_dht_model

try:
    set_start_method('spawn')
except RuntimeError:
    pass

wandb_mode = "online"
# wandb_mode = "disabled"

def create_network(in_dim, out_dim, network_width, network_depth, dropout):
    network_structure = [('linear', network_width), ('relu', None),
                         ('dropout', dropout)] * network_depth
    network_structure.append(('linear', out_dim))

    return NeuralNetwork(in_dim, network_structure)


def train_state_mapping(config=None, project=None):
    print(device)

    # Initialize a new wandb run
    with wandb.init(config=config, project=project, mode=wandb_mode):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        data_file_A, data_file_B = config.data_files

        data_path_A = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_file_A)
        data_path_B = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_file_B)

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

        weight_matrix_p = torch.zeros(len(data_B["dht_params"]), len(data_A["dht_params"]))
        weight_matrix_p[-1, -1] = 1
        weight_matrix_o = torch.zeros(len(data_B["dht_params"]), len(data_A["dht_params"]))
        weight_matrix_o[-1, -1] = 1

        loss_fn = KinematicChainLoss(weight_matrix_p, weight_matrix_o).to(device)

        wandb.config.update({
            "loss_fn": str(loss_fn)
        })

        model = create_network(
            len(data_A["joint_limits"]),
            len(data_B["joint_limits"]),
            config.network_width,
            config.network_depth,
            config.dropout
        ).to(device)

        if config.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
        elif config.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=50, min_lr=1e-6)

        best_loss = np.inf
        patience = config.get("patience", np.inf)

        for epoch in tqdm(range(config.epochs)):
            model.train()
            for states, link_poses in loader_train:
                states, link_poses = states.to(device), link_poses.to(device)
                optimizer.zero_grad()

                states_target = model(states)
                link_poses_target = dht_model_B(states_target)

                loss, loss_p, loss_o = loss_fn(link_poses_target, link_poses)

                # todo: integrate valid state discriminator (Seminar Elias / Mark)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                states_target = model(states_test)
                link_poses_target = dht_model_B(states_target)

                loss, loss_p, loss_o = loss_fn(link_poses_target, link_poses_test)

                # error_tcp_p = torch.norm(link_poses_target[:, -1, :3, -1] - link_poses_test[:, -1, :3, -1], p=2, dim=-1).mean()

                wandb.log({
                    'loss': loss.item(),
                    'loss_p': loss_p.item(),
                    'loss_o': loss_o.item(),
                    # 'error_tcp_p': error_tcp_p.item(),
                    'lr': optimizer.param_groups[0]["lr"],
                }, step=epoch)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch
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
    wandb.agent(sweep_id, function=train_state_mapping, count=count)


if __name__ == '__main__':
    data_file_A = "data/panda_10000_1000.pt"
    data_file_B = "data/ur5_10000_1000.pt"
    # data_file_A = "data/panda_10_10.pt"
    # data_file_B = "data/ur5_10_10.pt"

    project = "robot2robot_state_mapper"

    wandb.login()

    sweep = True

    if sweep:
        sweep_config = {
            "name": "state_mapper_sweep",
            "method": "bayes",
            'metric': {
                'name': 'loss',
                'goal': 'minimize'
            },
            "parameters": {
                "data_files": {
                    "value": (data_file_A, data_file_B)
                },
                "network_width": {
                    "min": 128,
                    "max": 2048
                },
                "network_depth": {
                    "min": 2,
                    "max": 16
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
                    "values": [32, 64, 128, 256, 512, 1024]
                },
            }
        }

        runs_per_agent = 10

        sweep_id = wandb.sweep(sweep_config, project=project)
        # sweep_id = "bitter/robot2robot_state_mapper/ola3rf5f"

        processes = []

        for device_id in range(torch.cuda.device_count()):
            process = Process(target=launch_agent, args=(sweep_id, device_id, runs_per_agent))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

    else:
        config = {
            "data_files": (data_file_A, data_file_B),
            "network_width": 64,
            "network_depth": 8,
            "dropout": 0.,
            "optimizer": "adam",
            "lr": 1e-3,
            "epochs": 10_000,
            "batch_size": 32,
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_state_mapping(config, project=project)
