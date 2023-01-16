from copy import deepcopy

import numpy as np
import torch

from .mapper_state import StateMapper
from stage import LitStage
from world_models.transition import TransitionModel
from mapper import Mapper


class TrajectoryMapper(LitStage):
    transition_model = None
    state_mapper = None

    def __init__(self, config):
        self.model_cls = config["TrajectoryMapper"]["model_cls"]
        self.model_config = config["TrajectoryMapper"]
        super(TrajectoryMapper, self).__init__(config)

    def generate(self):
        super(TrajectoryMapper, self).generate()
        # load transition model
        self.transition_model = TransitionModel(self.config)
        self.model.transition_model = deepcopy(self.transition_model.model)
        del self.transition_model
        # load state mapper
        self.state_mapper = StateMapper(self.config)
        self.model.state_mapper = deepcopy(self.state_mapper.model)
        del self.state_mapper
        super(TrajectoryMapper, self).train()

    @classmethod
    def get_relevant_config(cls, config):
        return super(TrajectoryMapper, cls).get_relevant_config(config)


class WeaSCLMapper(Mapper):
    trajectory_mapper = None

    def __init__(self, config):
        super(WeaSCLMapper, self).__init__(config)

    def map_trajectories(self, trajectories):
        for trajectory in trajectories:
            yield self.map_trajectory(trajectory)

    def generate(self):
        self.trajectory_mapper = TrajectoryMapper(self.config)

    def load(self):
        self.trajectory_mapper = TrajectoryMapper(self.config)
        self.trajectory_mapper.load()

    def map_trajectory(self, trajectory):
        joint_positions_source = np.stack(
            [
                obs["state"]["robot"]["arm"]["joint_positions"]
                for obs in trajectory["obs"]
            ]
        )
        joint_positions_source = torch.from_numpy(joint_positions_source).float()

        actions_source = np.stack([action["arm"] for action in trajectory["actions"]])
        actions_source = torch.from_numpy(actions_source).float()

        predicted_states = self.trajectory_mapper.model.state_mapper(
            joint_positions_source
        )
        predicted_actions = self.trajectory_mapper.model(
            (joint_positions_source, actions_source)
        )

        trajectory = deepcopy(trajectory)

        for old_state, new_state in zip(trajectory["obs"], predicted_states):
            old_state["state"]["robot"]["arm"]["joint_positions"] = (
                new_state.cpu().detach().numpy()
            )

        for old_action, new_action in zip(trajectory["actions"], predicted_actions):
            old_action["arm"] = new_action.cpu().detach().tolist()

        return trajectory
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
import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
import math
from torch.nn import TransformerEncoderLayer, TransformerEncoder

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nn import NeuralNetwork


class TrajectoryEncoder(nn.Module):
    """
    The TrajectoryEncoder encodes a trajectory with elements of arbitrary continuous spaces.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        behavior_dim: int,
        max_len: int,
        d_model=64,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        **kwargs
    ):
        super(TrajectoryEncoder, self).__init__()

        self.max_len = max_len
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_state = nn.Conv1d(state_dim, d_model, 1)
        self.encoder_action = nn.Conv1d(action_dim, d_model, 1)

        encoder_layers = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.behavior_decoder = nn.Conv1d(d_model, behavior_dim, 1)

    def forward(self, states, actions):
        states_ = self.encoder_state(states.swapdims(1, 2))
        actions_ = self.encoder_action(actions.swapdims(1, 2))
        actions_ = torch.nn.functional.pad(actions_, (0, 1), value=torch.nan)

        states_actions = torch.stack((states_, actions_), dim=-1).view(
            *states_.shape[:2], -1
        )
        states_actions = states_actions[:, :, :-1]

        # padding = self.max_len - states_actions.shape[-1]
        # states_actions = torch.nn.functional.pad(actions_, (0, padding))
        states_actions.swapdims_(1, 2)

        out = self.transformer_encoder(states_actions)

        out.swapdims_(1, 2)

        behavior = self.behavior_decoder(out).mean(-1)

        return behavior


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5, dropout=0.0):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pos_div_term = position * div_term
        pe[:, 0::2] = torch.sin(pos_div_term[:, : (d_model + 2) // 2])
        pe[:, 1::2] = torch.cos(pos_div_term[:, : d_model // 2])
        pe = pe.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


states = torch.ones(2, 3, 3) * 2 - 1  # bld
actions = torch.zeros(2, 2, 2) * 2 - 1  # bmd

encoder = TrajectoryEncoder(
    state_dim=states.shape[-1],
    action_dim=actions.shape[-1],
    behavior_dim=32,
    max_len=states.shape[1] + actions.shape[1],
    d_model=2,
    nhead=2,
)

behavior = encoder(states, actions)
