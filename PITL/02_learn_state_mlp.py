"""
Train state mappings with dht models.
"""

import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import MSELoss
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nn import NeuralNetwork, KinematicChainLoss, create_network
from models.dht import get_dht_model
from models.transformer_encoder import Seq2SeqTransformerEncoder
import tempfile
import wandb
from config import data_folder
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from multiprocessing import cpu_count, set_start_method

from config import wandb_config


class LitStateMapper(pl.LightningModule):
    def __init__(self,
                 data_file_A, data_file_B,
                 state_mapper_config={},
                 batch_size=32,
                 num_workers=1,
                 **kwargs
                 ):
        super(LitStateMapper, self).__init__()

        self.save_hyperparameters()

        data_path_A = os.path.join(data_folder, data_file_A)
        data_A = torch.load(data_path_A)

        data_path_B = os.path.join(data_folder, data_file_B)
        data_B = torch.load(data_path_B)

        self.dht_model_A = get_dht_model(data_A["dht_params"], data_A["joint_limits"])
        self.dht_model_B = get_dht_model(data_B["dht_params"], data_B["joint_limits"])

        self.state_mapper_AB = create_network(
            in_dim=data_A["states_train"].shape[1],
            out_dim=data_B["states_train"].shape[1],
            **state_mapper_config
        )

        self.state_mapper_BA = create_network(
            in_dim=data_B["states_train"].shape[1],
            out_dim=data_A["states_train"].shape[1],
            **state_mapper_config
        )

        link_positions_A = self.dht_model_A(torch.zeros((1, *data_A["states_train"].shape[1:])))[0, :, :3, -1]
        link_positions_B = self.dht_model_B(torch.zeros((1, *data_B["states_train"].shape[1:])))[0, :, :3, -1]

        weight_matrix_AB_p = torch.zeros(link_positions_A.shape[0], link_positions_B.shape[0])
        weight_matrix_AB_p[-1, -1] = 1.
        weight_matrix_AB_o = torch.zeros(link_positions_A.shape[0], link_positions_B.shape[0])
        weight_matrix_AB_o[-1, -1] = 1.

        self.loss_fn_kinematics_AB = KinematicChainLoss(weight_matrix_AB_p, weight_matrix_AB_o)
        self.loss_fn_kinematics_BA = KinematicChainLoss(weight_matrix_AB_p.T, weight_matrix_AB_o.T)

        # manual optimization in training_step
        self.automatic_optimization = False

    """
        Maps states and actions A -> B
        Required by super class LightningModule
        
        Args:
            state_A
            action_A
        Returns:
            Mapped state and action
    """

    def forward(self, states_A):
        with torch.no_grad():
            states_B_ = self.state_mapper_AB.get_dummB_tgt(states_A)

            for _ in range(states_B_.shape[-1] - 1):
                states_B = self.state_mapper_AB(states_A, states_B_)
                states_B_ = torch.nn.functional.pad(states_B[:, :-1], (1, 0), mode="constant", value=torch.nan)

        states_B = self.state_mapper_AB(states_A, states_B_)

        return states_B

    """
        Generate weights based on distances between relative positions of robot links
        
        Params:
            link_positions_{X, Y}: Positions of both robots in 3D space.
            weight_matrix_exponent_p: Parameter used to shape the position weight matrix by emphasizing similarity.
                weight = exp(-weight_matrix_exponent_p * distance)
        
        Returns:
            weight_matrix_XY_p: Weight matrix for positions
            weight_matrix_XY_p: Weight matrix for orientations. All zeros except weight which corresponds to the end effectors.
    """

    @staticmethod
    def get_weight_matrices(link_positions_X, link_positions_Y, weight_matrix_exponent_p, norm=True):
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

    """
        Generate dataloader used for training.
        Refer to pytorch lightning docs.
    """

    def train_dataloader(self):
        dataloader_train_A = self.get_train_dataloader(self.hparams.data_file_A)
        dataloader_train_B = self.get_train_dataloader(self.hparams.data_file_B)
        return CombinedLoader({"A": dataloader_train_A, "B": dataloader_train_B})

    """
        Generate dataloader used for validation.
        Refer to pytorch lightning docs.
    """

    def val_dataloader(self):
        dataloader_validation_A = self.get_validation_dataloader(self.hparams.data_file_A)
        dataloader_validation_B = self.get_validation_dataloader(self.hparams.data_file_B)

        return CombinedLoader({"A": dataloader_validation_A, "B": dataloader_validation_B})

    """
        Determine loss of state mapper on batch.
    """

    def loss_state_mapper(self, batch, state_mapper_XY, dht_model_X, dht_model_Y):
        states_X, _, _ = batch
        states_Y = state_mapper_XY(states_X)

        link_poses_X = dht_model_X(states_X)
        link_poses_Y = dht_model_Y(states_Y)

        loss_state_mapper_XY_p = torch.norm(link_poses_X[:, -1, :3, -1] - link_poses_Y[:, -1, :3, -1], p=2, dim=-1).mean()


        # http://www.boris-belousov.net/2016/12/01/quat-dist/
        # R = P * Q^T
        # tr_R = R * eye(3)

        output_ = link_poses_X[:, -1, :3, :3]  # size: bx3x3
        target_ = torch.transpose(link_poses_Y[:, -1, :3, :3], -1, -2)  # size: bx3x3

        tr_R = torch.einsum("bxy,byz,xz->b", output_, target_, torch.eye(3, device=output_.device))

        # calculate angle
        # add eps to make input to arccos (-1, 1) to avoid numerical instability
        # scale between 0 and 1
        loss_state_mapper_XY_o = torch.acos(torch.clamp((tr_R - 1) / (2 + 1e-7), -1 + 1e-7, 1 - 1e-7)) / np.pi
        loss_state_mapper_XY_o = loss_state_mapper_XY_o.mean()

        loss_state_mapper_XY = loss_state_mapper_XY_p + loss_state_mapper_XY_o

        # error_tcp_p = torch.norm(link_poses_X[:, -1, :3, -1] - link_poses_Y[:, -1, :3, -1], p=2, dim=-1).mean()
        # error_tcp_o = torch.norm(link_poses_X[:, -1, :3, -1] - link_poses_Y[:, -1, :3, -1], p=2, dim=-1).mean()

        return loss_state_mapper_XY, loss_state_mapper_XY_p, loss_state_mapper_XY_o

    # def training_epoch_end(self, outputs) -> None:
    #     self.log("growth", self.growth_fn(self.current_epoch))

    """
        Perform training step. Customized behavior to enable accumulation of all losses into one variable.
        Refer to pytorch lightning docs.
    """

    def training_step(self, batch, batch_idx):

        cumulated_loss = 0.

        for optimizer_idx, optimizer in enumerate(self.optimizers()):
            optimizer.zero_grad()
            loss = self.step(batch, batch_idx, optimizer_idx, "train_")
            self.manual_backward(loss)
            optimizer.step()

            cumulated_loss += loss.item()

        self.log("train_loss", cumulated_loss, on_step=False, on_epoch=True)

    # def training_epoch_end(self, outputs):
    #     super(LitStateMapper, self).training_epoch_end(outputs)
    #
    #     for scheduler_idx, scheduler in enumerate(self.lr_schedulers()):
    #         scheduler.step()

    #         if scheduler_idx == 0:
    #             metric_name = "validation_loss_state_mapper_AB"
    #         elif scheduler_idx == 1:
    #             metric_name = "validation_loss_state_mapper_BA"
    #         else:
    #             raise ValueError(f"Metric for scheduler_idx {scheduler_idx} unknown!")
    #
    #         metric = self.trainer.callback_metrics[metric_name]
    #         scheduler.step(metric)

    """
        Perform validation step. Customized behavior to enable accumulation of all losses into one variable.
        Refer to pytorch lightning docs.
    """

    def validation_step(self, batch, batch_idx):

        cumulated_loss = 0.

        for optimizer_idx in range(len(self.optimizers())):
            loss = self.step(batch, batch_idx, optimizer_idx, "validation_")
            cumulated_loss += loss.item()

        self.log("validation_loss", cumulated_loss, on_step=False, on_epoch=True)

    """
        Perform one step (compute loss) for a model corresponding to an optimizer_idx.
        
        Params:
            batch: data
            batch_idx: not used
            optimizer_idx: Optimizer (and respective model) to evaluate
            log_prefix: used to log training and validation losses under different names
        
        Returns:
            loss
    """

    def step(self, batch, batch_idx, optimizer_idx, log_prefix=""):
        if optimizer_idx == 0:
            loss_state_mapper, loss_state_mapper_p, loss_state_mapper_o = self.loss_state_mapper(batch["A"],
                                                                                                 self.state_mapper_AB,
                                                                                                 self.dht_model_A,
                                                                                                 self.dht_model_B)
            self.log(log_prefix + 'loss_state_mapper_AB', loss_state_mapper, on_step=False, on_epoch=True)
            self.log(log_prefix + 'loss_state_mapper_AB_p', loss_state_mapper_p, on_step=False, on_epoch=True)
            self.log(log_prefix + 'loss_state_mapper_AB_o', loss_state_mapper_o, on_step=False, on_epoch=True)
            return loss_state_mapper
        elif optimizer_idx == 1:
            loss_state_mapper, loss_state_mapper_p, loss_state_mapper_o = self.loss_state_mapper(batch["B"],
                                                                                                 self.state_mapper_BA,
                                                                                                 self.dht_model_B,
                                                                                                 self.dht_model_A)
            self.log(log_prefix + 'loss_state_mapper_BA', loss_state_mapper, on_step=False, on_epoch=True)
            self.log(log_prefix + 'loss_state_mapper_BA_p', loss_state_mapper_p, on_step=False, on_epoch=True)
            self.log(log_prefix + 'loss_state_mapper_BA_o', loss_state_mapper_o, on_step=False, on_epoch=True)
            return loss_state_mapper

    """
        Helper function to generate all optimizers.
        Refer to pytorch lightning docs.
    """

    def configure_optimizers(self):
        optimizer_state_mapper_AB = torch.optim.AdamW(self.state_mapper_AB.parameters(),
                                                     lr=self.hparams.state_mapper_config.get("lr", 3e-4))
        optimizer_state_mapper_BA = torch.optim.AdamW(self.state_mapper_BA.parameters(),
                                                     lr=self.hparams.state_mapper_config.get("lr", 3e-4))

        scheduler_state_mapper_AB = {"scheduler": OneCycleLR(optimizer_state_mapper_AB, 1e-2, 50),
                                     "monitor": "validation_loss_state_mapper_AB",
                                     "name": "scheduler_optimizer_state_mapper_AB"
                                     }

        scheduler_state_mapper_BA = {"scheduler": OneCycleLR(optimizer_state_mapper_BA, 1e-2, 50),
                                     "monitor": "validation_loss_state_mapper_BA",
                                     "name": "scheduler_optimizer_state_mapper_BA"
                                     }

        return [optimizer_state_mapper_AB, optimizer_state_mapper_BA], [] # [scheduler_state_mapper_AB, scheduler_state_mapper_BA]


if __name__ == '__main__':
    CPU_COUNT = cpu_count()
    GPU_COUNT = torch.cuda.device_count()

    data_file_A = "panda_10000_1000.pt"
    data_file_B = "ur5_10000_1000.pt"

    wandb_config.update(
        {
            # "notes": "add softmax loss weight",
            "group": "state_mapper",
            "tags": ["mlp", "warm start"]
        }
    )

    config = {
        "data_file_A": data_file_A,
        "data_file_B": data_file_B,
        "state_mapper_config": {
            "network_width": 256,
            "network_depth": 8,
            "dropout": .1,
            "out_activation": "tanh",
            "lr": 3e-3,
        },
        "batch_size": 32,
        "max_epochs": 3_000,
    }

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        devices = -1  # torch.cuda.device_count()
        config["num_workers"] = cpu_count() // torch.cuda.device_count()
        # config["num_workers"] = cpu_count()
    else:
        devices = None
        config["num_workers"] = cpu_count()

    # config["warm_start"] = "robot2robot/PITL/model-2kdezjba:best"
    #
    # if "warm_start" in config:
    #     with tempfile.TemporaryDirectory() as dir:
    #         api = wandb.Api()
    #         artifact = api.artifact(config["warm_start"])
    #         artifact_dir = artifact.download(dir)
    #
    #         state_mapper = LitStateMapper.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))

    state_mapper = LitStateMapper(
        **config
    )

    # state_mapper.loss_fn_kinematics_AB.weight_matrix_orientations.data[-1,-1] = 1.
    # state_mapper.loss_fn_kinematics_BA.weight_matrix_orientations.data[-1,-1] = 1.

    callbacks = [
        ModelCheckpoint(monitor="validation_loss", mode="min"),
        LearningRateMonitor(logging_interval='epoch'),
        # EarlyStopping(monitor="validation_loss", mode="min", patience=1000)
    ]

    wandb_logger = WandbLogger(**wandb_config, log_model="all")

    # trainer = pl.Trainer()
    # trainer.fit(state_mapper)
    # 
    # exit()

    # try:
    trainer = pl.Trainer(strategy=DDPStrategy(), accelerator="gpu", devices=devices,
                         logger=wandb_logger, max_epochs=config["max_epochs"], callbacks=callbacks)
    trainer.fit(state_mapper)
    # except:
    #     trainer = pl.Trainer(accelerator="gpu", devices=devices,
    #                          logger=wandb_logger, max_epochs=config["max_epochs"], callbacks=callbacks)
    #     trainer.fit(state_mapper)
