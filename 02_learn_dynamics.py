"""
Verify differentiability of dht module by performing inverse kinematics.
"""

import os

import numpy as np
import torch
import wandb
from torch.multiprocessing import Process, set_start_method
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.dht import DHT_Transform
from utils.nn import NeuralNetwork, KinematicChainLoss, Rescale

try:
    set_start_method('spawn')
except RuntimeError:
    pass

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

wandb_mode = "online"
# wandb_mode = "disabled"
sweep = True
# sweep = False


def train_dynamics_model(config=None, project=None):
    print(device)

    # Initialize a new wandb run
    with wandb.init(config=config, project=project, entity="bitter", mode=wandb_mode):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        data_file = config.data_file

        data = torch.load(data_file)

        states_train, actions_train, next_states_train = data["states_train"], data["actions_train"], data[
            "next_states_train"]
        states_test, actions_test, next_states_test = data["states_train"].to(device), \
                                                      data["actions_train"].to(device),\
                                                      data["next_states_train"].to(device)

        loader_train = DataLoader(TensorDataset(states_train, actions_train, next_states_train),
                                  batch_size=config.batch_size, shuffle=True)

        loss_fn = lambda x, y: 1 - torch.nn.functional.cosine_similarity(x, y).mean()

        wandb.config.update({
            "loss_fn": str(loss_fn)
        })

        network_structure = [('linear', config.network_width), ('relu', None),
                             ('dropout', config.dropout)] * config.network_depth
        network_structure.append(('linear', actions_train.shape[-1]))

        model = NeuralNetwork((states_train.shape[1:], next_states_train.shape[1:]), network_structure).to(device)

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

        best_error = np.inf
        patience = config.get("patience", np.inf)

        for epoch in tqdm(range(config.epochs)):
            model.train()
            for s, a, s_ in loader_train:
                s, a, s_ = s.to(device), a.to(device), s_.to(device)
                optimizer.zero_grad()

                prediction = model(s, s_)
                loss = loss_fn(prediction, a)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                prediction = model(states_test, next_states_test)
                loss = loss_fn(prediction, actions_test)

                wandb.log({
                    'loss': loss.item(),
                    'lr': optimizer.param_groups[0]["lr"],
                }, step=epoch)

                if loss.item() < best_error:
                    best_error = loss.item()
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch
                    }, os.path.join(wandb.run.dir, "model.pt"))
                    best_patience = 0
                else:
                    best_patience += 1
                    if best_patience > patience:
                        break

            if loss < 1e-3 or loss.isnan().any():
                break


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


def launch_agent(sweep_id, device_id, count):
    global device
    device = device_id
    wandb.agent(sweep_id, function=train_dynamics_model, count=count)


if __name__ == '__main__':
    # validate_loss_fn()
    # perform_gradient_decent_ik(data_file_A, notes="test pos + ori tcp loss")

    data_file = "data/transitions/panda_10000_1000.pt"
    project = f"dynamics_{os.path.basename(data_file).replace('.pt', '')}"

    wandb.login()

    if sweep:
        sweep_config = {
            "name": "dynamics_model_sweep",
            "method": "bayes",
            'metric': {
                'name': 'loss',
                'goal': 'minimize'
            },
            "parameters": {
                "data_file": {
                    "value": data_file
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
            "network_width": 64,
            "network_depth": 8,
            "dropout": 0.,
            "optimizer": "adam",
            "lr": 1e-3,
            "epochs": 10_000,
            "batch_size": 32,
            "upscale_dim": False,
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dynamics_model(config, project=project)
