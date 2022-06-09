import os
from copy import deepcopy

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import wandb
from dht import DHT_Model
from nn import NeuralNetwork, KinematicChainLoss, Sawtooth

# wandb.init(project="state-mapper", entity="bitter")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb_mode = "online"


# wandb_mode = "disabled"

def generate_weight_matrix():
    if config_loss.get("weight_matrix_positions", "") == "auto":
        poses_A = dht_model_A(torch.zeros((1, len(joint_limits_A)))).squeeze()
        positions_A = poses_A[:, :3, -1]
        positions_A = torch.cat((torch.zeros(1, 3), positions_A))
        link_lenghts_A = torch.norm(positions_A[1:] - positions_A[:-1], p=2, dim=-1)
        link_lenghts_A = link_lenghts_A.cumsum(0)
        link_lenghts_A = link_lenghts_A / link_lenghts_A[-1]

        poses_B = dht_model_B(torch.zeros((1, len(joint_limits_B)))).squeeze()
        positions_B = poses_B[:, :3, -1]
        positions_B = torch.cat((torch.zeros(1, 3), positions_B))
        link_lenghts_B = torch.norm(positions_B[1:] - positions_B[:-1], p=2, dim=-1)
        link_lenghts_B = link_lenghts_B.cumsum(0)
        link_lenghts_B = link_lenghts_B / link_lenghts_B[-1]

        weight_matrix = (1 - torch.cdist(link_lenghts_A.unsqueeze(-1), link_lenghts_B.unsqueeze(-1))) ** 2
        weight_matrix = weight_matrix * link_lenghts_A.unsqueeze(-1)
        weight_matrix = weight_matrix * link_lenghts_B.unsqueeze(0)

        config_loss["weight_matrix_positions"] = weight_matrix
    elif config_loss.get("weight_matrix_positions", "") == "tcp":
        poses_A = dht_model_A(torch.zeros((1, len(joint_limits_A)))).squeeze()
        poses_B = dht_model_B(torch.zeros((1, len(joint_limits_B)))).squeeze()

        weight_matrix = torch.zeros(len(poses_A), len(poses_B))
        weight_matrix[-1, -1] = 1

        config_loss["weight_matrix_positions"] = weight_matrix

    if config_loss.get("weight_matrix_orientations", "") == "auto":
        poses_A = dht_model_A(torch.zeros((1, len(joint_limits_A)))).squeeze()
        poses_B = dht_model_B(torch.zeros((1, len(joint_limits_B)))).squeeze()

        weight_matrix = torch.zeros((len(poses_A), len(poses_B)))
        weight_matrix[-1, -1] = 1.
        config_loss["weight_matrix_orientations"] = weight_matrix  # .to(device)

    if any([v is False for v in config_loss.values()]):
        poses_A = dht_model_A(torch.zeros((1, len(joint_limits_A)))).squeeze()
        poses_B = dht_model_B(torch.zeros((1, len(joint_limits_B)))).squeeze()

        weight_matrix = torch.zeros((len(poses_A), len(poses_B)))

        if config_loss.get("weight_matrix_positions", "") is False:
            config_loss["weight_matrix_positions"] = weight_matrix
        if config_loss.get("weight_matrix_orientations", "") is False:
            config_loss["weight_matrix_orientations"] = weight_matrix


if __name__ == '__main__':
    import numpy as np

    import sys
    from pathlib import Path

    wandb.login()

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    # network_structure = [('linear', 512), ('relu', None), ('dropout', 0.1)] * 8
    network_structure = [('linear', 512), ('relu', None), ('dropout', 0.1)] * 8

    configs = [
        {
            "epochs": 10_000,
            "batch_size": 32,
            "optimizer": "adam",
            "lr": 1e-3,
            "network_structure": network_structure,
            "data_file_A": "data/2link_10000_1000.pt",
            "data_file_B": "data/3link_10000_1000.pt",
            "config_loss": {"weight_matrix_positions": "auto"},
            # "scheduler": "reduce_plateau",
            "patience": 2000,
            # "warm_start": "results/learn_state_mapper/20220324-171736/0/model_4.313.pt"
        }
    ]

    for cid, config in enumerate(configs):
        with wandb.init(config=config,
                        project="robot2robot_state_mapper", entity="bitter",
                        mode=wandb_mode):
            model_path = os.path.join("results", os.path.basename(__file__).replace('.py', ''), wandb.run.name,
                                      "model.pt")

            epochs = config.get("epochs", 500)
            batch_size = config.get("batch_size", 512)
            network_structure = config.get("network_structure", [])
            lr = config.get("lr", 3e-4)
            optimizer = config.get("optimzier", "sgd")
            data_file_A = config["data_file_A"]
            data_file_B = config["data_file_B"]
            patience = config.get("patience", None)
            scheduler = config.get("scheduler", None)
            warm_start = config.get("warm_start", None)

            best_performance = np.inf
            steps_since_best = 0
            best_state_dict = None

            # if torch.cuda.device_count() > 1:
            #     batch_size *= torch.cuda.device_count()

            data_A = torch.load(data_file_A)

            dht_params_A = data_A["dht_params"]
            joint_limits_A = data_A["joint_limits"]

            data_B = torch.load(data_file_B)
            dht_params_B = data_B["dht_params"]
            joint_limits_B = data_B["joint_limits"]


            def get_dht_model(dht_params):
                return torch.nn.Sequential(
                    Sawtooth(-1, 1, -np.pi, np.pi),
                    DHT_Model(dht_params)
                )


            dht_model_A = get_dht_model(dht_params_A)
            dht_model_B = get_dht_model(dht_params_B)

            X_train, X_test = data_A["X_train"], data_A["X_test"]

            y_train = dht_model_A(X_train).detach()
            y_test = dht_model_A(X_test).detach()

            data_loader_train = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            network_structure.append(('linear', len(joint_limits_B)))
            model = torch.nn.Sequential(
                NeuralNetwork(len(joint_limits_A), network_structure),
                dht_model_B
            )

            if warm_start:
                model.load_state_dict(torch.load(warm_start))
            else:
                def init_weights(m):
                    if isinstance(m, torch.nn.Linear):
                        torch.nn.init.xavier_uniform_(m.weight)


                model.apply(init_weights)

            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = torch.nn.DataParallel(model)

            model.to(device)
            dht_model_B.to(device)

            weight_matrix_p = torch.zeros(len(joint_limits_B), len(joint_limits_A))
            weight_matrix_p[-1, -1] = 1
            weight_matrix_o = torch.zeros(len(joint_limits_B), len(joint_limits_A))
            weight_matrix_o[-1, -1] = 1

            loss_function = KinematicChainLoss(weight_matrix_p, weight_matrix_o).to(device)

            if optimizer == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            elif optimizer == "adam":
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-4)

            if scheduler == "coswr":
                scheduler = CosineAnnealingWarmRestarts(optimizer, 2_000)
            elif scheduler == "reduce_plateau":
                scheduler = ReduceLROnPlateau(optimizer, factor=.9, patience=150)

            try:
                for epoch in tqdm(range(epochs)):
                    model.train()

                    for x, y in data_loader_train:
                        x = x.to(device)
                        y = y.to(device)

                        prediction_poses = model(x)

                        optimizer.zero_grad()
                        loss, _, _ = loss_function(prediction_poses, y)

                        loss.backward()

                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        optimizer.step()

                    with torch.no_grad():
                        model.eval()
                        prediction_poses = model(X_test)
                        loss_test, loss_position, loss_orientation = loss_function(prediction_poses, y_test)

                        deviation_tcp_position = torch.norm(prediction_poses[:, -1, :3, -1] - y_test[:, -1, :3, -1],
                                                            p=2, dim=-1).mean()

                        wandb.log(
                            {'loss': loss_test.item(),
                             'loss_p': loss_position.item(),
                             'loss_o': loss_orientation.item(),
                             'tcp_p error': deviation_tcp_position.item(),
                             'lr': optimizer.param_groups[0]["lr"],
                             }, step=epoch)

                        if scheduler is not None:
                            if type(scheduler) == ReduceLROnPlateau:
                                scheduler.step(deviation_tcp_position.item())
                            else:
                                scheduler.step()

                        if deviation_tcp_position.item() < best_performance:
                            best_performance = deviation_tcp_position.item()
                            steps_since_best = 0
                            if isinstance(model, torch.nn.DataParallel):
                                best_state_dict = deepcopy(model.module.state_dict())
                            else:
                                best_state_dict = deepcopy(model.state_dict())

                        elif patience is not None:
                            steps_since_best += 1
                            if steps_since_best > patience:
                                break


            except KeyboardInterrupt:
                print("Shutting down")

            if best_state_dict is not None:
                os.path.dirname(model_path)
                torch.save(best_state_dict, model_path)
