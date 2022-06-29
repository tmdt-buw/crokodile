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
from torch.nn import MSELoss
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

wandb_mode = "disabled"


def create_network(in_dim, out_dim, network_width, network_depth, dropout):
    network_structure = [('linear', network_width), ('relu', None),
                         ('dropout', dropout)] * network_depth
    network_structure.append(('linear', out_dim))

    return NeuralNetwork(in_dim, network_structure)


def get_weight_matrices(link_positions_X, link_positions_Y, weight_matrix_exponent_p):
    link_positions_X = torch.cat((torch.zeros(1, 3, device=device), link_positions_X))
    link_lenghts_X = torch.norm(link_positions_X[1:] - link_positions_X[:-1], p=2, dim=-1)
    link_order_X = link_lenghts_X.cumsum(0)
    link_order_X = link_order_X / link_order_X[-1]

    link_positions_Y = torch.cat((torch.zeros(1, 3, device=device), link_positions_Y))
    link_lenghts_Y = torch.norm(link_positions_Y[1:] - link_positions_Y[:-1], p=2, dim=-1)
    link_order_Y = link_lenghts_Y.cumsum(0)
    link_order_Y = link_order_Y / link_order_Y[-1]

    weight_matrix_XY_p = torch.exp(
        -weight_matrix_exponent_p * torch.cdist(link_order_X.unsqueeze(-1), link_order_Y.unsqueeze(-1)))
    weight_matrix_XY_p = torch.nan_to_num(weight_matrix_XY_p, 1.)

    weight_matrix_XY_o = torch.zeros(len(link_positions_X), len(link_positions_Y))
    weight_matrix_XY_o[-1, -1] = 1

    return weight_matrix_XY_p, weight_matrix_XY_p


def train_state_mapping(config=None, project=None):
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

        states_A_train = data_A["states_train"]
        actions_A_train = data_A["actions_train"]
        next_states_A_train = data_A["next_states_train"]
        states_A_test = data_A["states_test"].to(device)
        actions_A_test = data_A["actions_test"].to(device)
        next_states_A_test = data_A["next_states_test"].to(device)

        states_B_train = data_B["states_train"]
        actions_B_train = data_B["actions_train"]
        next_states_B_train = data_B["next_states_train"]
        states_B_test = data_B["states_test"].to(device)
        actions_B_test = data_B["actions_test"].to(device)
        next_states_B_test = data_B["next_states_test"].to(device)

        # states_actions_B_train = torch.concat((states_B_train, actions_B_train), axis=-1)
        states_actions_B_test = torch.concat((states_B_test, actions_B_test), axis=-1).to(device)

        dht_model_A = get_dht_model(data_A["dht_params"], data_A["joint_limits"]).to(device)
        dht_model_B = get_dht_model(data_B["dht_params"], data_B["joint_limits"]).to(device)

        loader_train_A = DataLoader(TensorDataset(states_A_train, actions_A_train, next_states_A_train),
                                    batch_size=config.batch_size, shuffle=True)

        loader_train_B = DataLoader(TensorDataset(states_B_train, actions_B_train, next_states_B_train),
                                    batch_size=config.batch_size, shuffle=True)

        # Dynamics Models
        dynamics_model_A = create_network(
            in_dim=states_A_train.shape[1] + actions_A_train.shape[1],
            out_dim=states_A_train.shape[1],
            network_width=config.network_width,
            network_depth=config.network_depth,
            dropout=config.dropout
        ).to(device)
        optimizer_dynamics_model_A = torch.optim.Adam(dynamics_model_A.parameters(), lr=config.lr)
        loss_fn_dynamics_model_A = MSELoss().to(device)
        scheduler_dynamics_model_A = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_dynamics_model_A, factor=.95,
                                                                                patience=200, min_lr=1e-5)

        dynamics_model_B = create_network(
            in_dim=states_B_train.shape[1] + actions_B_train.shape[1],
            out_dim=states_B_train.shape[1],
            network_width=config.network_width,
            network_depth=config.network_depth,
            dropout=config.dropout
        ).to(device)
        optimizer_dynamics_model_B = torch.optim.Adam(dynamics_model_B.parameters(), lr=config.lr)
        loss_fn_dynamics_model_B = MSELoss().to(device)
        scheduler_dynamics_model_B = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_dynamics_model_B, factor=.95,
                                                                                patience=200, min_lr=1e-5)

        # State Mappers
        link_positions_A = dht_model_A(torch.zeros((1, *data_A["states_train"].shape[1:]), device=device))[0, :, :3, -1]
        link_positions_B = dht_model_B(torch.zeros((1, *data_B["states_train"].shape[1:]), device=device))[0, :, :3, -1]

        weight_matrix_exponent_p = config.get("weight_matrix_exponent", torch.inf)

        weight_matrix_AB_p, weight_matrix_AB_o = get_weight_matrices(link_positions_A, link_positions_B,
                                                                     weight_matrix_exponent_p)


        state_mapper_AB = create_network(
            in_dim=states_A_train.shape[1],
            out_dim=states_B_train.shape[1],
            network_width=config.network_width,
            network_depth=config.network_depth,
            dropout=config.dropout
        ).to(device)
        optimizer_state_mapper_AB = torch.optim.Adam(state_mapper_AB.parameters(), lr=config.lr)
        loss_fn_kinematics_AB = KinematicChainLoss(weight_matrix_AB_p, weight_matrix_AB_o).to(device)
        scheduler_state_mapper_AB = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_state_mapper_AB, factor=.95,
                                                                               patience=200,
                                                                               min_lr=1e-5)

        state_mapper_BA = create_network(
            in_dim=states_B_train.shape[1],
            out_dim=states_A_train.shape[1],
            network_width=config.network_width,
            network_depth=config.network_depth,
            dropout=config.dropout
        ).to(device)
        optimizer_state_mapper_BA = torch.optim.Adam(state_mapper_BA.parameters(), lr=config.lr)
        loss_fn_kinematics_BA = KinematicChainLoss(weight_matrix_AB_p.T, weight_matrix_AB_o.T).to(device)
        scheduler_state_mapper_BA = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_state_mapper_BA, factor=.95,
                                                                               patience=200,
                                                                               min_lr=1e-5)

        action_mapper_AB = create_network(
            in_dim=actions_A_train.shape[1],
            out_dim=actions_B_train.shape[1],
            network_width=config.network_width,
            network_depth=config.network_depth,
            dropout=config.dropout
        ).to(device)
        optimizer_action_mapper_AB = torch.optim.Adam(action_mapper_AB.parameters(), lr=config.lr)
        loss_fn_kinematics_AB = KinematicChainLoss(weight_matrix_AB_p, weight_matrix_AB_o).to(device)
        scheduler_action_mapper_AB = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_action_mapper_AB, factor=.95,
                                                                               patience=200,
                                                                               min_lr=1e-5)

        action_mapper_BA = create_network(
            in_dim=actions_B_train.shape[1],
            out_dim=actions_A_train.shape[1],
            network_width=config.network_width,
            network_depth=config.network_depth,
            dropout=config.dropout
        ).to(device)
        optimizer_action_mapper_BA = torch.optim.Adam(action_mapper_BA.parameters(), lr=config.lr)
        loss_fn_kinematics_BA = KinematicChainLoss(weight_matrix_AB_p.T, weight_matrix_AB_o.T).to(device)
        scheduler_action_mapper_BA = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_action_mapper_BA, factor=.95,
                                                                               patience=200,
                                                                               min_lr=1e-5)

        best_loss = np.inf
        patience = config.get("patience", np.inf)

        def train(state_mapper_XY, optimizer_state_mapper_XY,
                  dynamics_model_X, optimizer_dynamics_model_X,
                  action_mapper_XY, optimizer_action_mapper_XY,
                  dht_model_X, dht_model_Y, dynamics_model_Y,
                  loss_fn_kinematics_XY, loss_fn_dynamics_model_X,
                  loader_train_X):

            state_mapper_XY.train()
            dynamics_model_X.train()
            action_mapper_XY.train()

            for states_X, actions_X, next_states_X in loader_train_X:
                states_X, actions_X, next_states_X = states_X.to(device), actions_X.to(device), next_states_X.to(device)

                # Train dynamics model pX(sX|sX,aX)
                states_actions_X = torch.concat((states_X, actions_X), axis=-1)

                optimizer_dynamics_model_X.zero_grad()
                next_states_X_predicted = dynamics_model_X(states_actions_X)

                loss_dynamics_model = loss_fn_dynamics_model_X(next_states_X_predicted, next_states_X)
                loss_dynamics_model.backward()
                optimizer_dynamics_model_X.step()

                # Train state mapper dhtY(smXY(sX)) ~ dhtX(sX)
                optimizer_state_mapper_XY.zero_grad()
                states_Y = state_mapper_XY(states_X)

                link_poses_X = dht_model_X(states_X)
                link_poses_Y = dht_model_Y(states_Y)

                loss_state_mapper_XY, loss_state_mapper_XY_position, loss_state_mapper_XY_orientation = \
                    loss_fn_kinematics_XY(link_poses_X, link_poses_Y)

                loss_state_mapper_XY.backward()
                optimizer_state_mapper_XY.step()

                # Train action mapper A->B dhtY(pY(smXY(sX),amXY(aY))) ~ dhtX(s_X)
                optimizer_action_mapper_XY.zero_grad()
                states_Y = state_mapper_XY(states_X)

                actions_Y = action_mapper_XY(actions_X)
                states_actions_Y = torch.concat((states_Y, actions_Y), axis=-1)

                next_states_Y = dynamics_model_Y(states_actions_Y)

                link_poses_X = dht_model_X(next_states_X)
                link_poses_Y = dht_model_Y(next_states_Y)

                loss_action_mapper_XY, loss_action_mapper_XY_position, loss_action_mapper_XY_orientation = \
                    loss_fn_kinematics_XY(link_poses_X, link_poses_Y)

                loss_action_mapper_XY.backward()
                optimizer_action_mapper_XY.step()

        def test(state_mapper_XY, optimizer_state_mapper_XY,
                  dynamics_model_X, optimizer_dynamics_model_X,
                  action_mapper_XY, optimizer_action_mapper_XY,
                  dht_model_X, dht_model_Y, dynamics_model_Y,
                  loss_fn_kinematics_XY, loss_fn_dynamics_model_X,
                  loader_test_X):

            state_mapper_XY.train()
            dynamics_model_X.train()
            action_mapper_XY.train()

            for states_X, actions_X, next_states_X in loader_test_X:
                states_X, actions_X, next_states_X = states_X.to(device), actions_X.to(device), next_states_X.to(device)

                # Train dynamics model pX(sX|sX,aX)
                states_actions_X = torch.concat((states_X, actions_X), axis=-1)

                optimizer_dynamics_model_X.zero_grad()
                next_states_X_predicted = dynamics_model_X(states_actions_X)

                loss_dynamics_model = loss_fn_dynamics_model_X(next_states_X_predicted, next_states_X)
                loss_dynamics_model.backward()
                optimizer_dynamics_model_X.step()

                # Train state mapper dhtY(smXY(sX)) ~ dhtX(sX)
                optimizer_state_mapper_XY.zero_grad()
                states_Y = state_mapper_XY(states_X)

                link_poses_X = dht_model_X(states_X)
                link_poses_Y = dht_model_Y(states_Y)

                loss_state_mapper_XY, loss_state_mapper_XY_position, loss_state_mapper_XY_orientation = \
                    loss_fn_kinematics_XY(link_poses_X, link_poses_Y)

                loss_state_mapper_XY.backward()
                optimizer_state_mapper_XY.step()

                # Train action mapper A->B dhtY(pY(smXY(sX),amXY(aY))) ~ dhtX(s_X)
                optimizer_action_mapper_XY.zero_grad()
                states_Y = state_mapper_XY(states_X)

                actions_Y = action_mapper_XY(actions_X)
                states_actions_Y = torch.concat((states_Y, actions_Y), axis=-1)

                next_states_Y = dynamics_model_Y(states_actions_Y)

                link_poses_X = dht_model_X(next_states_X)
                link_poses_Y = dht_model_Y(next_states_Y)

                loss_action_mapper_XY, loss_action_mapper_XY_position, loss_action_mapper_XY_orientation = \
                    loss_fn_kinematics_XY(link_poses_X, link_poses_Y)

                loss_action_mapper_XY.backward()
                optimizer_action_mapper_XY.step()

        for epoch in tqdm(range(config.epochs)):

            # todo: integrate valid state discriminator (Seminar Elias / Mark)
            # todo: train cycle-consistency of state mappers (& action mappers)

            # Train dynamics model pB(sB,aB) ~ s_B
            # Train state mapper B->A dhtA(smBA(sB)) ~ dhtB(sB)
            # Train state mapper A->B dhtB(smAB(sA)) ~ dhtA(sA) [only required for action mapper]
            # Train action mapper A->B smBA(pB(smAB(sA),amAB(aA))) ~ s_A

            train(state_mapper_AB, optimizer_state_mapper_AB,
                  dynamics_model_A, optimizer_dynamics_model_A,
                  action_mapper_AB, optimizer_action_mapper_AB,
                  dht_model_A, dht_model_B, dynamics_model_B,
                  loss_fn_kinematics_AB, loss_fn_dynamics_model_A,
                  loader_train_A)

            train(state_mapper_BA, optimizer_state_mapper_BA,
                  dynamics_model_B, optimizer_dynamics_model_B,
                  action_mapper_BA, optimizer_action_mapper_BA,
                  dht_model_B, dht_model_A, dynamics_model_A,
                  loss_fn_kinematics_BA, loss_fn_dynamics_model_B,
                  loader_train_B)

            test(state_mapper_AB, optimizer_state_mapper_AB,
                  dynamics_model_A, optimizer_dynamics_model_A,
                  action_mapper_AB, optimizer_action_mapper_AB,
                  dht_model_A, dht_model_B, dynamics_model_B,
                  loss_fn_kinematics_AB, loss_fn_dynamics_model_A,
                  loader_train_A)

            test(state_mapper_BA, optimizer_state_mapper_BA,
                  dynamics_model_B, optimizer_dynamics_model_B,
                  action_mapper_BA, optimizer_action_mapper_BA,
                  dht_model_B, dht_model_A, dynamics_model_A,
                  loss_fn_kinematics_BA, loss_fn_dynamics_model_B,
                  loader_train_B)

            with torch.no_grad():
                # Evaluate dynamics model
                dynamics_model_B.eval()
                next_states_B_predicted = dynamics_model_B(states_actions_B_test)

                loss_dynamics_model = loss_fn_dynamics_model_B(next_states_B_predicted, next_states_B)

                wandb.log({
                    'loss_dynamics_model': loss_dynamics_model.item(),
                }, step=epoch)

                # Evaluate state mapper B->A
                state_mapper_BA.eval()
                states_A = state_mapper_BA(states_B_test)

                link_poses_B = dht_model_B(states_B_test)
                link_poses_A = dht_model_A(states_A)

                loss_state_mapper_BA, loss_state_mapper_BA_position, loss_state_mapper_BA_orientation = \
                    loss_fn_kinematics_BA(link_poses_B, link_poses_A)

                error_state_mapper_BA_tcp_position = torch.norm(link_poses_B[:, -1, :3, -1] -
                                                                link_poses_A[:, -1, :3, -1], p=2, dim=-1).mean()

                wandb.log({
                    'loss_state_mapper_BA': loss_state_mapper_BA.item(),
                    'loss_state_mapper_BA_position': loss_state_mapper_BA_position.item(),
                    'loss_state_mapper_BA_orientation': loss_state_mapper_BA_orientation.item(),
                    'error_state_mapper_BA_tcp_position': error_state_mapper_BA_tcp_position.item(),
                    'lr_state_mapper_BA': optimizer_state_mapper_BA.param_groups[0]["lr"],
                }, step=epoch)

                # Evaluate state mapper A->B
                state_mapper_AB.eval()
                states_B = state_mapper_AB(states_A_test)

                link_poses_A = dht_model_A(states_A_test)
                link_poses_B = dht_model_B(states_B)

                loss_state_mapper_AB, loss_state_mapper_AB_position, loss_state_mapper_AB_orientation = \
                    loss_fn_kinematics_AB(link_poses_A, link_poses_B)

                error_state_mapper_AB_tcp_position = torch.norm(link_poses_A[:, -1, :3, -1] -
                                                                link_poses_B[:, -1, :3, -1], p=2, dim=-1).mean()

                wandb.log({
                    'loss_state_mapper_AB': loss_state_mapper_AB.item(),
                    'loss_state_mapper_AB_position': loss_state_mapper_AB_position.item(),
                    'loss_state_mapper_AB_orientation': loss_state_mapper_AB_orientation.item(),
                    'error_state_mapper_AB_tcp_position': error_state_mapper_AB_tcp_position.item(),
                    'lr_state_mapper_AB': optimizer_state_mapper_AB.param_groups[0]["lr"],
                }, step=epoch)

                # if loss_state.item() < best_loss:
                #     best_loss = loss_state.item()
                #     torch.save({
                #         'model_config': state_mapper_BA_config,
                #         'model_state_dict': state_mapper_BA.state_dict(),
                #     }, os.path.join(wandb.run.dir, "state_mapper_BA.pt"))
                #     steps_since_improvement = 0
                # else:
                #     steps_since_improvement += 1
                #     if steps_since_improvement > patience:
                #         break

            scheduler_state.step(loss_state_mapper_BA.item())

            if loss_state_mapper_BA < 1e-3 or loss_state_mapper_BA.isnan().any():
                break


def launch_agent(sweep_id, device_id, count):
    global device
    device = device_id
    wandb.agent(sweep_id, function=train_state_mapping, count=count)


if __name__ == '__main__':
    data_file_A = "panda_10000_1000.pt"
    data_file_B = "ur5_10000_1000.pt"

    project = "state_mapper_BA"

    sweep = False

    if sweep:
        sweep_id = None
        # sweep_id = "robot2robot/_/uv04xo6w"
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

        # if torch.cuda.is_available():
        for device_id in range(torch.cuda.device_count()):
            process = Process(target=launch_agent, args=(sweep_id, device_id, runs_per_agent))
            process.start()
            processes.append(process)
        # else:
        #     for device_id in range(cpu_count() // 2):
        #         process = Process(target=launch_agent, args=(sweep_id, "cpu", runs_per_agent))
        #         process.start()
        #         processes.append(process)

        for process in processes:
            process.join()

    else:
        config = {
            "data_files": (data_file_A, data_file_B),
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
        train_state_mapping(config, project=project)
