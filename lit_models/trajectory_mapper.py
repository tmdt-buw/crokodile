import os
import sys
from pathlib import Path
import torch

from pytorch_lightning.trainer.supporters import CombinedLoader
from lit_models.state_mapper import LitStateMapper
from lit_models.transition_model import LitTransitionModel
from lit_models.lit_model import LitModel
from models.trajectory_encoder import TrajectoryEncoder
from itertools import chain
from torch.nn import MSELoss

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nn import create_network, KinematicChainLoss
from utils.soft_dtw_cuda import SoftDTW

from config import data_folder


class LitTrajectoryMapper(LitModel):
    def __init__(self, config):
        super(LitTrajectoryMapper, self).__init__(config["TrajectoryMapper"])
        self.trajectory_encoder, self.policy = self.model
        self.transition_model = LitTransitionModel(config)
        self.state_mapper = LitStateMapper(config)
        self.loss_function = self.get_loss()

    def get_model(self):
        data_path_X = os.path.join(data_folder, self.lit_config["data"]["data_file_X"])
        data_X = torch.load(data_path_X)

        data_path_Y = os.path.join(data_folder, self.lit_config["data"]["data_file_Y"])
        data_Y = torch.load(data_path_Y)

        trajectory_encoder = TrajectoryEncoder(
            state_dim=data_X["trajectories_states_train"].shape[-1],
            action_dim=data_X["trajectories_actions_train"].shape[-1],
            behavior_dim=self.lit_config["model"]["behavior_dim"],
            max_len=data_X["trajectories_states_train"].shape[-1]
            + data_X["trajectories_actions_train"].shape[-1],
            **self.lit_config["model"]["encoder"],
        )

        policy = create_network(
            in_dim=data_Y["trajectories_states_train"].shape[-1]
            + self.lit_config["model"]["behavior_dim"],
            out_dim=data_Y["trajectories_actions_train"].shape[-1],
            **self.lit_config["model"]["decoder"],
        )

        return trajectory_encoder, policy

    def get_loss(self):
        data_path_X = os.path.join(data_folder, self.lit_config["data"]["data_file_X"])
        data_X = torch.load(data_path_X)

        data_path_Y = os.path.join(data_folder, self.lit_config["data"]["data_file_Y"])
        data_Y = torch.load(data_path_Y)

        link_positions_A = self.state_mapper.dht_model_A(
            torch.zeros((1, data_X["trajectories_states_train"].shape[-1]))
        )[0, :, :3, -1]
        link_positions_B = self.state_mapper.dht_model_B(
            torch.zeros((1, data_Y["trajectories_states_train"].shape[-1]))
        )[0, :, :3, -1]

        weight_matrix_AB_p, weight_matrix_AB_o = self.state_mapper.get_weight_matrices(
            link_positions_A,
            link_positions_B,
            self.lit_config["model"]["weight_matrix_exponent_p"],
        )

        loss_soft_dtw_AB = SoftDTW(
            use_cuda=True,
            dist_func=KinematicChainLoss(
                weight_matrix_AB_p, weight_matrix_AB_o, reduction=False
            ),
        )

        return loss_soft_dtw_AB

    def configure_optimizers(self):
        optimizer_action_mapper = torch.optim.AdamW(
            chain(self.trajectory_encoder.parameters(), self.policy.parameters()),
            lr=self.lit_config["train"].get("lr", 3e-4),
        )
        return optimizer_action_mapper

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

    def forward(self, batch):
        states_A, actions_A = batch
        with torch.no_grad():
            # bz
            behaviors = self.trajectory_encoder(states_A, actions_A)
            # bs
            states_B = self.state_mapper(states_A[:, 0])

            for _ in range(states_A.shape[1]):
                # b(s+z)
                states_B_behaviors = torch.concat((states_B, behaviors), dim=-1)
                # ba
                actions_B = self.policy(states_B_behaviors)

        return actions_B

    def loss(self, batch):
        # bls, bla
        trajectories_states_A, trajectories_actions_A = batch

        # bz
        behaviors = self.trajectory_encoder(
            trajectories_states_A, trajectories_actions_A
        )

        trajectories_states_B = []

        # bs
        states_B = self.state_mapper(trajectories_states_A[:, 0])
        trajectories_states_B.append(states_B)

        for _ in range(trajectories_actions_A.shape[1]):
            # b(s+z)
            states_B_behaviors = torch.concat((states_B, behaviors), dim=-1)

            # ba
            actions_B = self.policy(states_B_behaviors)
            # b(s+a)
            states_actions_B = torch.concat((states_B, actions_B), dim=-1)
            # bs
            states_B = self.transition_model(states_actions_B)

            trajectories_states_B.append(states_B)

        # bls
        trajectories_states_B = torch.stack(trajectories_states_B).swapdims(0, 1)

        # (b*l)s
        states_A = trajectories_states_A.reshape(-1, trajectories_states_A.shape[-1])
        states_B = trajectories_states_B.reshape(-1, trajectories_states_B.shape[-1])

        # (b*l)p44
        link_poses_A = self.state_mapper.dht_model_A(states_A)
        link_poses_B = self.state_mapper.dht_model_B(states_B)

        # blp44
        link_poses_A = link_poses_A.reshape(
            *trajectories_states_A.shape[:2], *link_poses_A.shape[1:]
        )
        link_poses_B = link_poses_B.reshape(
            *trajectories_states_B.shape[:2], *link_poses_B.shape[1:]
        )

        loss = self.loss_function(link_poses_A, link_poses_B).mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch["A"])
        self.log(
            "train_loss_LitTrajectoryMapper" + self.lit_config["log_suffix"],
            loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch["A"])
        self.log(
            "validation_loss_LitTrajectoryMapper" + self.lit_config["log_suffix"],
            loss,
            on_step=False,
            on_epoch=True,
        )
        return loss
