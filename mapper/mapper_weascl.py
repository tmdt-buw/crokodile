import math
import sys
from copy import deepcopy
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from mapper import Mapper
from stage import LitStage
from utils.nn import KinematicChainLoss, create_network, get_weight_matrices
from utils.soft_dtw_cuda import SoftDTW
from world_models.transition import TransitionModelSource, TransitionModelTarget

from .mapper_state import EnvWrapper, StateMapper

sys.path.append(str(Path(__file__).resolve().parents[1]))

from functools import cached_property

from environments import get_env


class MapperWeaSCL(LitStage, Mapper):
    def __init__(self, config, **kwargs):
        super(MapperWeaSCL, self).__init__(config, **kwargs)

    def init_models(self, config, **kwargs):
        self.automatic_optimization = False

        env_config_source = config["EnvSource"]["env_config"]
        env_config_source["name"] = config["EnvSource"]["env"]
        self.env_source = EnvWrapper(get_env(env_config_source))

        env_config_target = config["EnvTarget"]["env_config"]
        env_config_target["name"] = config["EnvTarget"]["env"]
        self.env_target = EnvWrapper(get_env(env_config_target))

        self.state_mapper = StateMapper(config, **kwargs)

        self.action_mapper_source_target = self.get_action_mapper(
            self.env_source,
            self.env_target,
            config[self.__class__.__name__]["model"]["action_mapper"],
        )

        self.action_mapper_target_source = self.get_action_mapper(
            self.env_target,
            self.env_source,
            config[self.__class__.__name__]["model"]["action_mapper"],
        )

    def get_action_mapper(self, env_from, env_to, model_config):
        state_mapper = create_network(
            in_dim=env_from.state_space["robot"]["arm"]["joint_positions"].shape[-1]
            + env_from.action_space["arm"].shape[-1],
            out_dim=env_to.action_space["arm"].shape[-1],
            **model_config,
        )

        return state_mapper

    @cached_property
    def transition_model_source(self):
        transition_model = deepcopy(TransitionModelSource(self.config).transition_model)
        return transition_model

    @cached_property
    def transition_model_target(self):
        transition_model = deepcopy(TransitionModelTarget(self.config).transition_model)
        return transition_model

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

    def map_batch(self, batch, state_mapper_XY, action_mapper_XY, transition_model_Y):
        # batch x len x s_dim_X, batch x len x a_dim_X
        trajectories_states_X, trajectories_actions_X = batch

        # len x batch x s_dim_X
        trajectories_states_X = trajectories_states_X.swapdims(0, 1)
        # len x batch x a_dim_X
        trajectories_actions_X = trajectories_actions_X.swapdims(0, 1)

        trajectories_states_Y = []
        trajectories_actions_Y = []

        # batch, s_dim_Y
        states_Y = state_mapper_XY(trajectories_states_X[0])
        trajectories_states_Y.append(states_Y)

        for states_X, actions_X in zip(
            trajectories_states_X[1:], trajectories_actions_X
        ):
            # batch x (s_dim_Y + a_dim_Y)
            actions_Y = action_mapper_XY(states_X, actions_X)

            trajectories_actions_Y.append(actions_Y)

            # batch x (s_dim_T + a_dim_T)
            states_actions_Y = torch.concat(
                (trajectories_states_Y[-1], actions_Y), dim=-1
            )

            states_Y = transition_model_Y(states_actions_Y)

            trajectories_states_Y.append(states_Y)

        trajectories_states_Y = torch.stack(trajectories_states_Y).swapdims(0, 1)

        trajectories_actions_Y = torch.stack(trajectories_actions_Y).swapdims(0, 1)

        return trajectories_states_Y, trajectories_actions_Y

    def get_state_dict(self):
        state_dict = {
            "action_mapper_source_target": self.action_mapper_source_target.state_dict(),
            "action_mapper_target_source": self.action_mapper_target_source.state_dict(),
        }
        return state_dict

    def set_state_dict(self, state_dict):
        self.action_mapper_source_target.load_state_dict(
            state_dict["action_mapper_source_target"]
        )
        self.action_mapper_target_source.load_state_dict(
            state_dict["action_mapper_target_source"]
        )

    @cached_property
    def loss_function(self):
        dummy_state_source = torch.zeros(
            (
                1,
                self.env_source.state_space["robot"]["arm"]["joint_positions"].shape[
                    -1
                ],
            )
        )
        dummy_state_target = torch.zeros(
            (
                1,
                self.env_target.state_space["robot"]["arm"]["joint_positions"].shape[
                    -1
                ],
            )
        )

        self.env_source.dht_model.to(dummy_state_source)
        self.env_target.dht_model.to(dummy_state_target)

        link_positions_source = self.env_source.dht_model(dummy_state_source)[
            0, :, :3, -1
        ]
        link_positions_target = self.env_target.dht_model(dummy_state_target)[
            0, :, :3, -1
        ]

        weight_matrix_p, weight_matrix_o = get_weight_matrices(
            link_positions_source,
            link_positions_target,
            self.config[self.__class__.__name__]["model"]["weight_matrix_exponent_p"],
        )
        loss_fn_ = KinematicChainLoss(weight_matrix_p, weight_matrix_o)

        def loss_fn(trajectories_states_source, trajectories_states_target):
            # (batch * len) x s_dim_S
            states_source = trajectories_states_source.reshape(
                -1, trajectories_states_source.shape[-1]
            )
            # (batch * len) x s_dim_T
            states_target = trajectories_states_target.reshape(
                -1, trajectories_states_target.shape[-1]
            )

            # todo: check if .to() calls are necessary or can be moved to init_models()
            self.env_source.state2angle.to(states_source)
            self.env_target.state2angle.to(states_target)

            angles_source = self.env_source.state2angle(states_source)
            angles_target = self.env_target.state2angle(states_target)

            self.env_source.dht_model.to(angles_source)
            self.env_target.dht_model.to(angles_target)

            link_poses_source = self.env_source.dht_model(angles_source)
            link_poses_target = self.env_target.dht_model(angles_target)

            loss_fn_.to(link_poses_source)
            loss = loss_fn_(link_poses_source, link_poses_target)

            return loss

        return loss_fn

    def configure_optimizers(self):
        optimizer_action_mapper_source_target = torch.optim.AdamW(
            self.action_mapper_source_target.parameters(),
            lr=self.config[self.__class__.__name__]["train"].get("lr", 3e-4),
        )
        optimizer_action_mapper_target_source = torch.optim.AdamW(
            self.action_mapper_target_source.parameters(),
            lr=self.config[self.__class__.__name__]["train"].get("lr", 3e-4),
        )
        optimizers_state_mapper = self.state_mapper.configure_optimizers()

        return [
            optimizer_action_mapper_source_target,
            optimizer_action_mapper_target_source,
            *optimizers_state_mapper,
        ]

    def train_dataloader(self):
        # todo: replace data file with data stage
        dataloader_train_source = self.get_dataloader(
            self.config[self.__class__.__name__]["data"]["data_file_X"],
            "train",
        )
        dataloader_train_target = self.get_dataloader(
            self.config[self.__class__.__name__]["data"]["data_file_Y"],
            "train",
        )
        return CombinedLoader(
            {
                "source": dataloader_train_source,
                "target": dataloader_train_target,
            }
        )

    def val_dataloader(self):
        dataloader_validation_source = self.get_dataloader(
            self.config[self.__class__.__name__]["data"]["data_file_X"],
            "test",
            False,
        )
        dataloader_validation_target = self.get_dataloader(
            self.config[self.__class__.__name__]["data"]["data_file_Y"],
            "test",
            False,
        )
        return CombinedLoader(
            {
                "source": dataloader_validation_source,
                "target": dataloader_validation_target,
            }
        )

    def forward(self, batch):
        return self.map_batch(
            batch,
            self.state_mapper.state_mapper_source_target,
            self.action_mapper_source_target,
            self.transition_model_target,
        )

    def loss(self, batch, batch_idx):
        # batch x len x s_dim_S, batch x len x a_dim_S
        trajectories_states_source, trajectories_actions_source = batch["source"]
        trajectories_states_target, trajectories_actions_target = batch["target"]

        # batch x len x s_dim_T, batch x len x a_dim_T
        trajectories_states_target_, trajectories_actions_target_ = self.map_batch(
            batch["source"],
            self.state_mapper.state_mapper_source_target,
            self.action_mapper_source_target,
            self.transition_model_target,
        )
        loss_source_target = self.loss_function(
            trajectories_states_source, trajectories_states_target_
        )

        trajectories_states_source_, trajectories_actions_source_ = self.map_batch(
            batch["target"],
            self.state_mapper.state_mapper_target_source,
            self.action_mapper_target_source,
            self.transition_model_source,
        )
        loss_target_source = self.loss_function(
            trajectories_states_source_, trajectories_states_target
        )

        loss = loss_source_target + loss_target_source

        log_dict = {
            f"loss_source_target_{self.log_id}": loss_source_target,
            f"loss_target_source_{self.log_id}": loss_target_source,
            f"loss_{self.log_id}": loss,
        }

        return loss, log_dict

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        (
            optimizer_action_mapper_source_target,
            optimizer_action_mapper_target_source,
        ) = optimizers[:2]
        optimizers_state_mapper = optimizers[2:]

        loss_state_mapper, log_dict = self.state_mapper.training_step_(
            optimizers_state_mapper, batch, batch_idx
        )

        loss, log_dict_ = self.loss(batch, batch_idx)
        log_dict.update(log_dict_)

        optimizer_action_mapper_source_target.zero_grad()
        optimizer_action_mapper_target_source.zero_grad()
        self.manual_backward(loss)
        optimizer_action_mapper_target_source.step()
        optimizer_action_mapper_source_target.step()

        log_dict = {"train_" + k: v for k, v in log_dict.items()}
        self.log_dict(log_dict, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.loss(batch, batch_idx)

        log_dict = {"validation_" + k: v for k, v in log_dict.items()}
        self.log_dict(log_dict, on_step=False, on_epoch=True)

        return loss


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
        **kwargs,
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

        self.behavior_encoder = nn.Conv1d(d_model, behavior_dim, 1)

    def forward(self, states, actions):
        # batch x dim x len
        states_ = self.encoder_state(states.swapdims(1, 2))
        # batch x dim x len-1
        actions_ = self.encoder_action(actions.swapdims(1, 2))
        # batch x dim x len
        actions_ = torch.nn.functional.pad(actions_, (0, 1), value=torch.nan)

        # batch x dim x 2*len
        states_actions = torch.stack((states_, actions_), dim=-1).view(
            *states_.shape[:-1], -1
        )

        # batch x dim x 2*len-1
        states_actions = states_actions[:, :, :-1]

        # batch x len x dim
        states_actions.swapdims_(1, 2)

        states_actions_pe = self.positional_encoding(states_actions)

        te = self.transformer_encoder(states_actions_pe)

        te.swapdims_(1, 2)

        behavior = self.behavior_encoder(te).mean(-1)

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
