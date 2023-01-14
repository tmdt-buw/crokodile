from copy import deepcopy


class StateMapper(LitTrainer):
    discriminator = None

    def __init__(self, config):
        self.model_cls = config["StateMapper"]["model_cls"]
        self.model_config = config["StateMapper"]

        super(StateMapper, self).__init__(config)

    def generate(self):
        super(StateMapper, self).generate()
        self.discriminator = Discriminator(self.config)
        self.model.discriminator = deepcopy(self.discriminator.model)
        del self.discriminator
        super(StateMapper, self).train()

    @classmethod
    def get_relevant_config(cls, config):
        return super(StateMapper, cls).get_relevant_config(config)
import os
import sys
from pathlib import Path

from torch.nn.functional import relu
import torch
from pytorch_lightning.trainer.supporters import CombinedLoader
from lit_models.discriminator import LitDiscriminator
from models.dht import get_dht_model


sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nn import create_network, KinematicChainLoss
from lit_models.lit_model import LitModel
from config import data_folder


class LitStateMapper(LitModel):
    def __init__(self, config):
        super(LitStateMapper, self).__init__(config["StateMapper"])
        self.dht_model_A, self.dht_model_B = self.get_dht_models()
        self.loss_function = self.get_kinematic_chain_loss()
        self.discriminator = LitDiscriminator(config)

    def get_model(self):
        data_path_X = os.path.join(data_folder, self.lit_config["data"]["data_file_X"])
        data_X = torch.load(data_path_X)

        data_path_Y = os.path.join(data_folder, self.lit_config["data"]["data_file_Y"])
        data_Y = torch.load(data_path_Y)

        state_mapper_XY = create_network(
            in_dim=data_X["trajectories_states_train"].shape[-1],
            out_dim=data_Y["trajectories_states_train"].shape[-1],
            **self.lit_config["model"],
        )

        return state_mapper_XY

    @staticmethod
    def get_weight_matrices(
        link_positions_X, link_positions_Y, weight_matrix_exponent_p, norm=True
    ):
        link_positions_X = torch.cat((torch.zeros(1, 3), link_positions_X))
        link_lenghts_X = torch.norm(
            link_positions_X[1:] - link_positions_X[:-1], p=2, dim=-1
        )
        link_order_X = link_lenghts_X.cumsum(0)
        link_order_X = link_order_X / link_order_X[-1]

        link_positions_Y = torch.cat((torch.zeros(1, 3), link_positions_Y))
        link_lenghts_Y = torch.norm(
            link_positions_Y[1:] - link_positions_Y[:-1], p=2, dim=-1
        )
        link_order_Y = link_lenghts_Y.cumsum(0)
        link_order_Y = link_order_Y / link_order_Y[-1]

        weight_matrix_XY_p = torch.exp(
            -weight_matrix_exponent_p
            * torch.cdist(link_order_X.unsqueeze(-1), link_order_Y.unsqueeze(-1))
        )
        weight_matrix_XY_p = torch.nan_to_num(weight_matrix_XY_p, 1.0)

        weight_matrix_XY_o = torch.zeros(len(link_positions_X), len(link_positions_Y))
        weight_matrix_XY_o[-1, -1] = 1

        if norm:
            weight_matrix_XY_p /= weight_matrix_XY_p.sum()
            weight_matrix_XY_o /= weight_matrix_XY_o.sum()

        return weight_matrix_XY_p, weight_matrix_XY_p

    def get_dht_models(self):
        data_path_X = os.path.join(data_folder, self.lit_config["data"]["data_file_X"])
        data_X = torch.load(data_path_X)

        data_path_Y = os.path.join(data_folder, self.lit_config["data"]["data_file_Y"])
        data_Y = torch.load(data_path_Y)

        dht_model_X = get_dht_model(data_X["dht_params"], data_X["joint_limits"])
        dht_model_Y = get_dht_model(data_Y["dht_params"], data_Y["joint_limits"])

        return dht_model_X, dht_model_Y

    def get_kinematic_chain_loss(self):
        data_path_A = os.path.join(data_folder, self.lit_config["data"]["data_file_X"])
        data_A = torch.load(data_path_A)

        data_path_B = os.path.join(data_folder, self.lit_config["data"]["data_file_Y"])
        data_B = torch.load(data_path_B)

        link_positions_A = self.dht_model_A(
            torch.zeros((1, data_A["trajectories_states_train"].shape[-1]))
        )[0, :, :3, -1]
        link_positions_B = self.dht_model_B(
            torch.zeros((1, data_B["trajectories_states_train"].shape[-1]))
        )[0, :, :3, -1]

        weight_matrix_AB_p, weight_matrix_AB_o = self.get_weight_matrices(
            link_positions_A,
            link_positions_B,
            self.lit_config["model"]["weight_matrix_exponent_p"],
        )
        loss_fn_kinematics_AB = KinematicChainLoss(
            weight_matrix_AB_p, weight_matrix_AB_o, verbose_output=True
        )

        return loss_fn_kinematics_AB

    def configure_optimizers(self):
        optimizer_state_mapper = torch.optim.AdamW(
            self.model.parameters(), lr=self.lit_config["train"].get("lr", 3e-4)
        )
        return optimizer_state_mapper

    def train_dataloader(self):
        dataloader_train_A = self.get_dataloader(
            self.lit_config["data"]["data_file_X"], "train"
        )
        dataloader_train_B = self.get_dataloader(
            self.lit_config["data"]["data_file_Y"], "train"
        )
        return CombinedLoader({"A": dataloader_train_A, "B": dataloader_train_B})

    def val_dataloader(self):
        dataloader_validation_A = self.get_dataloader(
            self.lit_config["data"]["data_file_X"], "test", False
        )
        dataloader_validation_B = self.get_dataloader(
            self.lit_config["data"]["data_file_Y"], "test", False
        )
        return CombinedLoader(
            {"A": dataloader_validation_A, "B": dataloader_validation_B}
        )

    def forward(self, states_A):
        states_B = self.model(states_A)
        return states_B

    def loss(self, batch):
        trajectories_states_X, _ = batch

        states_X = trajectories_states_X.reshape(-1, trajectories_states_X.shape[-1])

        states_Y = self.model(states_X)

        link_poses_X = self.dht_model_A(states_X)
        link_poses_Y = self.dht_model_B(states_Y)

        (
            loss_state_mapper_XY,
            loss_state_mapper_XY_p,
            loss_state_mapper_XY_o,
        ) = self.loss_function(link_poses_X, link_poses_Y)
        # discriminator loss
        outputs = self.discriminator(states_Y)
        dist = torch.sum((outputs - self.discriminator.model.c) ** 2, dim=1)
        loss_disc = torch.mean(relu(dist - self.discriminator.model.radius))

        loss_state_mapper_XY_disc = loss_state_mapper_XY + loss_disc

        return (
            loss_state_mapper_XY,
            loss_state_mapper_XY_disc,
            loss_state_mapper_XY_p,
            loss_state_mapper_XY_o,
        )

    def training_step(self, batch, batch_idx):
        (
            loss_state_mapper,
            loss_state_mapper_disc,
            loss_state_mapper_p,
            loss_state_mapper_o,
        ) = self.loss(batch["A"])
        self.log(
            "train_loss_LitStateMapper" + self.lit_config["log_suffix"],
            loss_state_mapper,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_loss_LitStateMapper_disc" + self.lit_config["log_suffix"],
            loss_state_mapper_disc,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_loss_LitStateMapper_p" + self.lit_config["log_suffix"],
            loss_state_mapper_p,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_loss_LitStateMapper_o" + self.lit_config["log_suffix"],
            loss_state_mapper_o,
            on_step=False,
            on_epoch=True,
        )
        return loss_state_mapper_disc

    def validation_step(self, batch, batch_idx):
        (
            loss_state_mapper,
            loss_state_mapper_disc,
            loss_state_mapper_p,
            loss_state_mapper_o,
        ) = self.loss(batch["A"])
        self.log(
            "validation_loss_LitStateMapper" + self.lit_config["log_suffix"],
            loss_state_mapper,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "validation_loss_LitStateMapper_disc" + self.lit_config["log_suffix"],
            loss_state_mapper_disc,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "validation_loss_LitStateMapper_p" + self.lit_config["log_suffix"],
            loss_state_mapper_p,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "validation_loss_LitStateMapper_o" + self.lit_config["log_suffix"],
            loss_state_mapper_o,
            on_step=False,
            on_epoch=True,
        )
        return loss_state_mapper_disc
