"""
Learn dht model from data
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

from utils.dht import get_dht_model, DHT_Transform
from utils.nn import KinematicChainLoss, Rescale, NeuralNetwork, Pos2Pose

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

    model = torch.nn.Sequential(
        NeuralNetwork(in_dim, network_structure),
        Pos2Pose()
    )

    return model


def train_dht_model(config=None, project=None):
    print(device)

    # Initialize a new wandb run
    with wandb.init(config=config, project=project, entity="bitter", mode=wandb_mode):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        data_file = config.data_file

        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_file)

        data = torch.load(data_path)

        states_train = data["states_train"]
        states_test = data["states_test"]

        dht_model = get_dht_model(data["dht_params"], data["joint_limits"])

        link_poses_train = dht_model(states_train).detach().detach()
        link_poses_test = dht_model(states_test).detach().detach()

        states_test = states_test.to(device)
        link_poses_test = link_poses_test.to(device)

        loader_train = DataLoader(TensorDataset(states_train, link_poses_train), batch_size=config.batch_size,
                                  shuffle=True)

        weight_matrix_p = torch.zeros(len(data["dht_params"]), len(data["dht_params"]))
        weight_matrix_p[-1, -1] = 1
        weight_matrix_o = torch.zeros(len(data["dht_params"]), len(data["dht_params"]))
        weight_matrix_o[-1, -1] = 1

        loss_fn = KinematicChainLoss(weight_matrix_p, weight_matrix_o).to(device)

        wandb.config.update({
            "loss_fn": str(loss_fn)
        })

        if model_type == "vanilla":
            model = create_network(
                len(data["joint_limits"]),
                len(data["dht_params"]) * 3,
                config.network_width,
                config.network_depth,
                config.dropout
            ).to(device)
        elif model_type == "PINN":
            model = get_dht_model(data["dht_params"], data["joint_limits"], config.upscale_dim).to(device)

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
            # elif isinstance(m, Rescale):
            #     torch.nn.init.normal_(m.m)
            #     torch.nn.init.normal_(m.c)
            elif isinstance(m, DHT_Transform) or isinstance(m, Rescale):
                for p in m.parameters():
                    if p.requires_grad:
                        torch.nn.init.normal_(p)

        model.apply(init_weights)

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

                link_poses_predicted = model(states)
                loss, loss_p, loss_o = loss_fn(link_poses_predicted, link_poses)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                link_poses_predicted = model(states_test)
                loss, loss_p, loss_o = loss_fn(link_poses_predicted, link_poses_test)

                error_tcp_p = torch.norm(link_poses_predicted[:, -1, :3, -1] - link_poses_test[:, -1, :3, -1], p=2,
                                         dim=-1).mean()

                wandb.log({
                    'loss': loss.item(),
                    'loss_p': loss_p.item(),
                    'loss_o': loss_o.item(),
                    'error_tcp_p': error_tcp_p.item(),
                    'lr': optimizer.param_groups[0]["lr"],
                }, step=epoch)

                if loss.item() < best_loss:
                    best_loss = error_tcp_p.item()
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
    wandb.agent(sweep_id, function=train_dht_model, count=count)


if __name__ == '__main__':
    data_file = "data/panda_10000_1000.pt"
    project = f"dht_{os.path.basename(data_file).replace('.pt', '')}"

    wandb.login()

    model_type = "vanilla"
    # model_type = "PINN"

    assert model_type in ["vanilla", "PINN"]

    sweep = True

    if sweep:
        sweep_config = {
            "name": "dht_model_sweep",
            "method": "bayes",
            'metric': {
                'name': 'loss',
                'goal': 'minimize'
            },
            "parameters": {
                "data_file": {
                    "value": data_file
                },
                "dropout": {
                    "min": 0.,
                    "max": .15
                },
                "optimizer": {
                    "values": ["sgd", "adam"]
                },
                "lr": {
                    "min": 0.0001,
                    "max": 0.005
                },
                "epochs": {
                    "value": 1000
                },
                "patience": {
                    "value": 200
                },
                "batch_size": {
                    "values": [64, 128, 256, 512]
                },
            }
        }

        if model_type == "vanilla":
            sweep_config["parameters"].update(
                {
                    "network_width": {
                        "min": 128,
                        "max": 2048
                    },
                    "network_depth": {
                        "min": 2,
                        "max": 16
                    },
                }
            )
        else:
            sweep_config["parameters"].update(
                {
                    "upscale_dim": {
                        "values": [False, 8, 32, 64]
                    }
                }
            )

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
            "data_file": data_file,
            "optimizer": "adam",
            "lr": 1e-3,
            "epochs": 10_000,
            "batch_size": 32,
            "dropout": 0.,
        }

        if model_type == "vanilla":
            config.update(
                {
                    "network_width": 32,
                    "network_depth": 4,
                }
            )
        else:
            config.update(
                {
                    "upscale_dim": False
                }
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dht_model(config, project=project)
