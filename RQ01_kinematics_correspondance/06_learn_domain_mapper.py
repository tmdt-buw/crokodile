"""
Train state mappings with dht models.
"""

import os
import sys
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import MSELoss
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import wandb
from torch.multiprocessing import Process

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nn import NeuralNetwork, KinematicChainLoss
from utils.dht import get_dht_model
from multiprocessing import cpu_count, set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

class LitDomainMapper(pl.LightningModule):
    def __init__(self,
                 data_file_A, data_file_B,
                 dynamics_model_network_width=32, dynamics_model_network_depth=1, dynamics_model_dropout=0.,
                 dynamics_model_lr=3e-4,
                 state_mapper_network_width=32, state_mapper_network_depth=1, state_mapper_dropout=0.,
                 state_mapper_lr=3e-4,
                 action_mapper_network_width=32, action_mapper_network_depth=1, action_mapper_dropout=0.,
                 action_mapper_lr=3e-4,
                 weight_matrix_exponent=np.inf,
                 batch_size=32,
                 **kwargs
                 ):
        super(LitDomainMapper, self).__init__()
        self.save_hyperparameters()

        self.dht_model_A, self.dynamics_model_A, self.state_mapper_AB, self.action_mapper_AB = \
            self.get_models(data_file_A, data_file_B,
                            dynamics_model_network_width, dynamics_model_network_depth, dynamics_model_dropout,
                            state_mapper_network_width, state_mapper_network_depth, state_mapper_dropout,
                            action_mapper_network_width, action_mapper_network_depth, action_mapper_dropout,
                            )
        self.dht_model_B, self.dynamics_model_B, self.state_mapper_BA, self.action_mapper_BA = \
            self.get_models(data_file_B, data_file_A,
                            dynamics_model_network_width, dynamics_model_network_depth, dynamics_model_dropout,
                            state_mapper_network_width, state_mapper_network_depth, state_mapper_dropout,
                            action_mapper_network_width, action_mapper_network_depth, action_mapper_dropout,
                            )

        self.loss_fn_dynamics_model, self.loss_fn_kinematics_AB, self.loss_fn_kinematics_BA = self.get_loss_functions()

        self.automatic_optimization = False

    def forward(self, x):
        return self.state_mapper_AB(x)

    def get_models(self, data_file_X, data_file_Y,
                   dynamics_model_network_width, dynamics_model_network_depth, dynamics_model_dropout,
                   state_mapper_network_width, state_mapper_network_depth, state_mapper_dropout,
                   action_mapper_network_width, action_mapper_network_depth, action_mapper_dropout,
                   ):
        data_path_X = os.path.join(data_folder, data_file_X)
        data_X = torch.load(data_path_X)

        data_path_Y = os.path.join(data_folder, data_file_Y)
        data_Y = torch.load(data_path_Y)

        dht_model_X = get_dht_model(data_X["dht_params"], data_X["joint_limits"])

        dynamics_model_X = self.create_network(
            in_dim=data_X["states_train"].shape[1] + data_X["actions_train"].shape[1],
            out_dim=data_X["states_train"].shape[1],
            network_width=dynamics_model_network_width,
            network_depth=dynamics_model_network_depth,
            dropout=dynamics_model_dropout,
            out_activation='tanh'
        )

        state_mapper_XY = self.create_network(
            in_dim=data_X["states_train"].shape[1],
            out_dim=data_Y["states_train"].shape[1],
            network_width=state_mapper_network_width,
            network_depth=state_mapper_network_depth,
            dropout=state_mapper_dropout,
            out_activation='tanh'
        )

        action_mapper_XY = self.create_network(
            in_dim=data_X["actions_train"].shape[1],
            out_dim=data_Y["actions_train"].shape[1],
            network_width=action_mapper_network_width,
            network_depth=action_mapper_network_depth,
            dropout=action_mapper_dropout,
            out_activation='tanh'
        )

        return dht_model_X, dynamics_model_X, state_mapper_XY, action_mapper_XY

    def get_loss_functions(self):
        data_path_A = os.path.join(data_folder, self.hparams.data_file_A)
        data_A = torch.load(data_path_A)

        data_path_B = os.path.join(data_folder, self.hparams.data_file_B)
        data_B = torch.load(data_path_B)

        loss_fn_dynamics_model = MSELoss()

        link_positions_A = self.dht_model_A(torch.zeros((1, *data_A["states_train"].shape[1:])))[0, :, :3, -1]
        link_positions_B = self.dht_model_B(torch.zeros((1, *data_B["states_train"].shape[1:])))[0, :, :3, -1]

        weight_matrix_AB_p, weight_matrix_AB_o = self.get_weight_matrices(link_positions_A, link_positions_B,
                                                                          self.hparams.weight_matrix_exponent)

        loss_fn_kinematics_AB = KinematicChainLoss(weight_matrix_AB_p, weight_matrix_AB_o)
        loss_fn_kinematics_BA = KinematicChainLoss(weight_matrix_AB_p.T, weight_matrix_AB_o.T)

        return loss_fn_dynamics_model, loss_fn_kinematics_AB, loss_fn_kinematics_BA

    def create_network(self, in_dim, out_dim, network_width, network_depth, dropout, out_activation=None):
        network_structure = [('linear', network_width), ('relu', None),
                             ('dropout', dropout)] * network_depth
        network_structure.append(('linear', out_dim))

        if out_activation:
            network_structure.append((out_activation, None))

        return NeuralNetwork(in_dim, network_structure)

    def get_weight_matrices(self, link_positions_X, link_positions_Y, weight_matrix_exponent_p, norm=True):
        link_positions_X = torch.cat((torch.zeros(1, 3), link_positions_X))
        link_lenghts_X = torch.norm(link_positions_X[1:] - link_positions_X[:-1], p=2, dim=-1)
        link_order_X = link_lenghts_X.cumsum(0)
        link_order_X = link_order_X / link_order_X[-1]

        link_positions_Y = torch.cat((torch.zeros(1, 3), link_positions_Y))
        link_lenghts_Y = torch.norm(link_positions_Y[1:] - link_positions_Y[:-1], p=2, dim=-1)
        link_order_Y = link_lenghts_Y.cumsum(0)
        link_order_Y = link_order_Y / link_order_Y[-1]

        weight_matrix_XY_p = torch.exp(
            -weight_matrix_exponent_p * torch.cdist(link_order_X.unsqueeze(-1), link_order_Y.unsqueeze(-1)))
        weight_matrix_XY_p = torch.nan_to_num(weight_matrix_XY_p, 1.)

        weight_matrix_XY_o = torch.zeros(len(link_positions_X), len(link_positions_Y))
        weight_matrix_XY_o[-1, -1] = 1

        if norm:
            weight_matrix_XY_p /= weight_matrix_XY_p.sum()
            weight_matrix_XY_o /= weight_matrix_XY_o.sum()

        return weight_matrix_XY_p, weight_matrix_XY_p

    def get_train_dataloader(self, data_file):
        data_path = os.path.join(data_folder, data_file)
        data = torch.load(data_path)

        states_train = data["states_train"]
        actions_train = data["actions_train"]
        next_states_train = data["next_states_train"]

        dataloader_train = DataLoader(TensorDataset(states_train, actions_train, next_states_train),
                                      batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                                      shuffle=True, pin_memory=True)

        return dataloader_train

    def get_validation_dataloader(self, data_file):
        data_path = os.path.join(data_folder, data_file)
        data = torch.load(data_path)

        states_validation = data["states_test"]
        actions_validation = data["actions_test"]
        next_states_validation = data["next_states_test"]

        dataloader_validation = DataLoader(TensorDataset(states_validation, actions_validation, next_states_validation),
                                           batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                                           pin_memory=True)

        return dataloader_validation

    def train_dataloader(self):
        dataloader_train_A = self.get_train_dataloader(self.hparams.data_file_A)
        dataloader_train_B = self.get_train_dataloader(self.hparams.data_file_B)
        return CombinedLoader({"A": dataloader_train_A, "B": dataloader_train_B})

    def val_dataloader(self):
        dataloader_validation_A = self.get_validation_dataloader(self.hparams.data_file_A)
        dataloader_validation_B = self.get_validation_dataloader(self.hparams.data_file_B)

        return CombinedLoader({"A": dataloader_validation_A, "B": dataloader_validation_B})

    def loss_dynamics_model(self, batch, dynamics_model, loss_fn):
        states, actions, next_states = batch
        states_actions_X = torch.concat((states, actions), axis=-1)
        next_states_X_predicted = dynamics_model(states_actions_X)
        loss_dynamics_model = loss_fn(next_states_X_predicted, next_states)
        return loss_dynamics_model

    def loss_state_mapper(self, batch, state_mapper_XY, dht_model_X, dht_model_Y, loss_fn):
        states_X, _, _ = batch

        states_Y = state_mapper_XY(states_X)

        link_poses_X = dht_model_X(states_X)
        link_poses_Y = dht_model_Y(states_Y)

        loss_state_mapper_XY, _, _ = loss_fn(link_poses_X, link_poses_Y)

        return loss_state_mapper_XY

    def loss_action_mapper(self, batch, action_mapper_XY, state_mapper_XY, dht_model_X, dht_model_Y, dynamics_model_Y,
                           loss_fn):
        states_X, actions_X, next_states_X = batch

        states_Y = state_mapper_XY(states_X)

        actions_Y = action_mapper_XY(actions_X)

        states_actions_Y = torch.concat((states_Y, actions_Y), axis=-1)

        next_states_Y = dynamics_model_Y(states_actions_Y)

        link_poses_X = dht_model_X(next_states_X)
        link_poses_Y = dht_model_Y(next_states_Y)

        loss_action_mapper_XY, _, _ = loss_fn(link_poses_X, link_poses_Y)

        return loss_action_mapper_XY

    def training_step(self, batch, batch_idx):

        cumulated_loss = 0.

        for optimizer_idx, optimizer in enumerate(self.optimizers()):
            optimizer.zero_grad()
            loss = self.step(batch, batch_idx, optimizer_idx, "train_")
            self.manual_backward(loss)
            optimizer.step()

            cumulated_loss += loss.item()

        self.log("train_loss", cumulated_loss, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):

        cumulated_loss = 0.

        for optimizer_idx in range(len(self.optimizers())):
            loss = self.step(batch, batch_idx, optimizer_idx, "validation_")
            cumulated_loss += loss.item()

        self.log("validation_loss", cumulated_loss, on_step=False, on_epoch=True)

    def step(self, batch, batch_idx, optimizer_idx, log_prefix=""):

        if optimizer_idx == 0:
            loss_dynamics_model = self.loss_dynamics_model(batch["A"], self.dynamics_model_A,
                                                           self.loss_fn_dynamics_model)
            self.log(log_prefix + 'loss_dynamics_model_A', loss_dynamics_model, on_step=False, on_epoch=True)
            return loss_dynamics_model
        elif optimizer_idx == 1:
            loss_dynamics_model = self.loss_dynamics_model(batch["B"], self.dynamics_model_B,
                                                           self.loss_fn_dynamics_model)
            self.log(log_prefix + 'loss_dynamics_model_B', loss_dynamics_model, on_step=False, on_epoch=True)
            return loss_dynamics_model
        elif optimizer_idx == 2:
            loss_state_mapper = self.loss_state_mapper(batch["A"], self.state_mapper_AB,
                                                       self.dht_model_A, self.dht_model_B, self.loss_fn_kinematics_AB)
            self.log(log_prefix + 'loss_state_mapper_AB', loss_state_mapper, on_step=False, on_epoch=True)
            return loss_state_mapper
        elif optimizer_idx == 3:
            loss_state_mapper = self.loss_state_mapper(batch["B"], self.state_mapper_BA,
                                                       self.dht_model_B, self.dht_model_A, self.loss_fn_kinematics_BA)
            self.log(log_prefix + 'loss_state_mapper_BA', loss_state_mapper, on_step=False, on_epoch=True)
            return loss_state_mapper
        elif optimizer_idx == 4:
            loss_action_mapper = self.loss_action_mapper(batch["A"], self.action_mapper_AB, self.state_mapper_AB,
                                                         self.dht_model_A, self.dht_model_B, self.dynamics_model_B,
                                                         self.loss_fn_kinematics_AB)
            self.log(log_prefix + 'loss_action_mapper_AB', loss_action_mapper, on_step=False, on_epoch=True)
            return loss_action_mapper
        elif optimizer_idx == 5:
            loss_action_mapper = self.loss_action_mapper(batch["B"], self.action_mapper_BA, self.state_mapper_BA,
                                                         self.dht_model_B, self.dht_model_A, self.dynamics_model_A,
                                                         self.loss_fn_kinematics_BA)
            self.log(log_prefix + 'loss_action_mapper_BA', loss_action_mapper, on_step=False, on_epoch=True)
            return loss_action_mapper

    def configure_optimizers(self):
        optimizer_dynamics_model_A = torch.optim.Adam(self.dynamics_model_A.parameters(),
                                                      lr=self.hparams.dynamics_model_lr)
        optimizer_dynamics_model_B = torch.optim.Adam(self.dynamics_model_B.parameters(),
                                                      lr=self.hparams.dynamics_model_lr)
        optimizer_state_mapper_AB = torch.optim.Adam(self.state_mapper_AB.parameters(),
                                                     lr=self.hparams.state_mapper_lr)
        optimizer_state_mapper_BA = torch.optim.Adam(self.state_mapper_BA.parameters(),
                                                     lr=self.hparams.state_mapper_lr)
        optimizer_action_mapper_AB = torch.optim.Adam(self.action_mapper_AB.parameters(),
                                                      lr=self.hparams.action_mapper_lr)
        optimizer_action_mapper_BA = torch.optim.Adam(self.action_mapper_BA.parameters(),
                                                      lr=self.hparams.action_mapper_lr)

        return [optimizer_dynamics_model_A, optimizer_dynamics_model_B,
                optimizer_state_mapper_AB, optimizer_state_mapper_BA,
                optimizer_action_mapper_AB, optimizer_action_mapper_BA], []


def run_training(config, wandb_config={}):

    if "warm_start" in config:

        with tempfile.TemporaryDirectory() as dir:
            api = wandb.Api()
            artifact = api.artifact(config["warm_start"])
            artifact_dir = artifact.download(dir)

            domain_mapper = LitDomainMapper.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))
    else:
        domain_mapper = LitDomainMapper(
            **config,
            dynamics_model_network_width=config["network_width"],
            dynamics_model_network_depth=config["network_depth"],
            dynamics_model_dropout=config["dropout"],
            dynamics_model_lr=config["lr"],
            state_mapper_network_width=config["network_width"],
            state_mapper_network_depth=config["network_depth"],
            state_mapper_dropout=config["dropout"],
            state_mapper_lr=config["lr"],
            action_mapper_network_width=config["network_width"],
            action_mapper_network_depth=config["network_depth"],
            action_mapper_dropout=config["dropout"],
            action_mapper_lr=config["lr"],
        )

    # trainer = pl.Trainer(strategy="ddp", accelerator="gpu", logger=wandb_logger)
    # trainer = pl.Trainer(strategy=DDPSpawnStrategy(), accelerator="gpu", logger=wandb_logger)
    callbacks = [
        ModelCheckpoint(monitor="validation_loss", mode="min"),
        EarlyStopping(monitor="validation_loss", mode="min", patience=1000)
    ]

    wandb_logger = WandbLogger(**wandb_config, log_model="all")

    trainer = pl.Trainer(strategy=DDPStrategy(), accelerator="gpu",
                         # gpus=-1,
                         devices=devices,
                         logger=wandb_logger, max_epochs=config["max_epochs"], callbacks=callbacks)
    trainer.fit(domain_mapper)


def launch_agent(sweep_id, devices_ids, count):
    global devices
    devices = devices_ids
    wandb.agent(sweep_id, function=run_training, count=count)


if __name__ == '__main__':
    CPU_COUNT = cpu_count()
    GPU_COUNT = torch.cuda.device_count()

    data_file_A = "panda_10000_1000.pt"
    data_file_B = "ur5_10000_1000.pt"

    wandb_config = {
        "project": "PITL",
        "group": "domain_mapper",
        "entity": "robot2robot",

        # "mode": "disabled"
    }

    sweep = False

    if sweep:

        sweep_config = {
            "method": "bayes",
            'metric': {
                'name': 'validation_loss',
                'goal': 'minimize'
            },
            "parameters": {
                "data_file_A": {
                    "value": data_file_A
                },
                "data_file_B": {
                    "value": data_file_B
                },
                "network_width": {
                    "min": 8,
                    "max": 256
                },
                "network_depth": {
                    "min": 1,
                    "max": 5
                },
                # "weight_matrix_exponent": {
                #     "value": 10
                # },
                "dropout": {
                    "min": 0.,
                    "max": .15
                },
                "lr": {
                    "min": 0.0001,
                    "max": 0.005
                },
                "batch_size": {
                    "values": [32, 64, 128, 256, 512]
                },
                "num_workers": {
                    "value": CPU_COUNT // GPU_COUNT
                },
                "max_epochs": {
                    "value": 100
                },
            }
        }

        runs_per_agent = 5

        sweep_id = wandb.sweep(sweep_config, **wandb_config)
        # sweep_id = "bitter/robot2robot_state_mapper/ola3rf5f"

        processes = []

        for device_id in range(GPU_COUNT):
            process = Process(target=launch_agent, args=(sweep_id, [device_id], runs_per_agent))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

    else:
        config = {
            "data_file_A": data_file_A,
            "data_file_B": data_file_B,
            "network_width": 32,
            "network_depth": 8,
            "weight_matrix_exponent": 10,
            "dropout": 0.1,
            "lr": 0.01,
            "batch_size": 64,
            "num_workers": 10,
            "max_epochs": 10_000,
            # "warm_start": "robot2robot/PITL/model-6o48z2nx:best"
        }

        # pl.utilities.rank_zero.rank_zero_only(wandb.init)(config=config, **wandb_config)
        # wandb.init(config=config, **wandb_config)

        torch.cuda.empty_cache()
        devices = -1 if torch.cuda.is_available() else None
        # devices = list(range(1,8))
        run_training(config, wandb_config)