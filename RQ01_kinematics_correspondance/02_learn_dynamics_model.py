"""
Learn dht model from data
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.multiprocessing import Process, set_start_method
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import matplotlib.pyplot as plt

import wandb

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.dht import get_dht_model, DHT_Transform
from utils.nn import Rescale, NeuralNetwork, Pos2Pose

try:
    set_start_method('spawn')
except RuntimeError:
    pass

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

# wandb_mode = "online"
wandb_mode = "disabled"


def create_network(in_dim, out_dim, network_width, network_depth, dropout):
    network_structure = [('linear', network_width), ('relu', None),
                         ('dropout', dropout)] * network_depth

    network_structure.append(('linear', out_dim))

    model = NeuralNetwork(in_dim, network_structure)

    return model


def train_dynamics_model(config=None, project=None):
    print(device)

    # Initialize a new wandb run
    with wandb.init(config=config, project=project, entity="bitter", mode=wandb_mode):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        data_file = config.data_file
        data_path = os.path.join(data_folder, data_file)

        data = torch.load(data_path)

        states_train = data["states_train"]
        actions_train = data["actions_train"]
        next_states_train = data["next_states_train"]

        states_test = data["states_test"]
        actions_test = data["actions_test"]
        next_states_test = data["next_states_test"]

        # dht_model = get_dht_model(data["dht_params"], data["joint_limits"])
        # link_poses_train = dht_model(states_train).detach()

        states_test = states_test.to(device)
        actions_test = actions_test.to(device)
        next_states_test = next_states_test.to(device)

        loss_fn = MSELoss().to(device)

        wandb.config.update({
            "loss_fn": str(loss_fn)
        })

        if config.model_type == "SAS":
            X_train = torch.concat((states_train, actions_train), dim=-1)
            y_train = next_states_train
            X_test = torch.concat((states_test, actions_test), dim=-1)
            y_test = next_states_test
        elif config.model_type == "SSA":
            X_train = torch.concat((states_train, next_states_train), dim=-1)
            y_train = actions_train
            X_test = torch.concat((states_test, next_states_test), dim=-1)
            y_test = actions_test

        loader_train = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=config.batch_size,
                                  shuffle=True)

        model_config = {
            "in_dim": X_train.shape[1],
            "out_dim": y_train.shape[1],
            "network_width": config.network_width,
            "network_depth": config.network_depth,
            "dropout": config.dropout
        }

        model = create_network(**model_config).to(device)

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

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.95, patience=50, min_lr=1e-6)

        best_loss = np.inf
        steps_since_improvement = 0
        patience = config.get("patience", np.inf)

        for epoch in tqdm(range(config.epochs)):
            model.train()
            for X, y in loader_train:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()

                y_predicted = model(X)
                loss = loss_fn(y_predicted, y)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()

                y_predicted = model(X_test)
                loss = loss_fn(y_predicted, y_test)

                wandb.log({
                    'loss': loss.item(),
                    'lr': optimizer.param_groups[0]["lr"],
                }, step=epoch)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save({
                        'model_config': model_config,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch
                    }, os.path.join(wandb.run.dir, "dynamics_model.pt"))
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
                    if steps_since_improvement > 2 * patience:
                        break

            scheduler.step(loss.item())

            if loss < 1e-4 or loss.isnan().any():
                break


def launch_agent(sweep_id, device_id, count):
    global device
    device = device_id
    wandb.agent(sweep_id, function=train_dynamics_model, count=count)


if __name__ == '__main__':
    data_file = "ur5_10000_1000.pt"
    project = f"dynamics_{os.path.basename(data_file).replace('.pt', '')}"

    # wandb.login()

    sweep = False

    if sweep:
        sweep_config = {
            "method": "bayes",
            'metric': {
                'name': 'loss',
                'goal': 'minimize'
            },
            "parameters": {
                "data_file": {
                    "value": data_file
                },
                "model_type": {
                    "value": "SSA"
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
            "model_type": "SSA",
            "optimizer": "adam",
            "lr": 1e-3,
            "epochs": 10_000,
            "batch_size": 32,
            "dropout": 0.,
            "patience": 200,
            "network_width": 32,
            "network_depth": 2,
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dynamics_model(config, project=project)
