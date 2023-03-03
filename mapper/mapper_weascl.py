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
from world_models.transition import TransitionModel

from .mapper_state import EnvWrapper, StateMapper

sys.path.append(str(Path(__file__).resolve().parents[1]))

from functools import cached_property

from environments.environment_robot_task import EnvironmentRobotTask


class MapperWeaSCL(LitStage, Mapper):
    def __init__(self, config, **kwargs):
        super(MapperWeaSCL, self).__init__(config, **kwargs)

    def init_models(self, config, **kwargs):
        self.automatic_optimization = False

        self.state_mapper = StateMapper(config, **kwargs)

        env_source = EnvironmentRobotTask(config["EnvSource"]["env_config"])
        env_target = EnvironmentRobotTask(config["EnvTarget"]["env_config"])

        self.trajectory_encoder = TrajectoryEncoder(
            state_dim=env_source.state_space["robot"]["arm"]["joint_positions"].shape[
                -1
            ],
            action_dim=env_source.action_space["arm"].shape[-1],
            behavior_dim=config[self.__class__.__name__]["model"]["behavior_dim"],
            max_len=env_source.task.max_steps * 2 + 1,
            **config[self.__class__.__name__]["model"]["encoder"],
        )

        # todo: make policy own module
        self.policy = create_network(
            in_dim=config[self.__class__.__name__]["model"]["behavior_dim"]
            + env_target.state_space["robot"]["arm"]["joint_positions"].shape[-1],
            out_dim=env_target.action_space["arm"].shape[-1],
            **config[self.__class__.__name__]["model"]["decoder"],
        )

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

    @cached_property
    def transition_model(self):
        transition_model = deepcopy(TransitionModel(self.config).transition_model)
        transition_model.to(self.device)
        return transition_model

    def get_state_dict(self):
        state_dict = {
            "trajectory_encoder": self.trajectory_encoder.state_dict(),
            "policy": self.policy.state_dict(),
        }
        return state_dict

    def set_state_dict(self, state_dict):
        self.trajectory_encoder.load_state_dict(state_dict["trajectory_encoder"])
        self.policy.load_state_dict(state_dict["policy"])

    # todo: consolidate this into own file (see duplicate mapper_state.py)
    @cached_property
    def env_source(self):
        env = EnvironmentRobotTask(self.config["EnvSource"]["env_config"])
        env_wrapper = EnvWrapper(env)
        return env_wrapper

    # todo: consolidate this into own file (see duplicate mapper_state.py)
    @cached_property
    def env_target(self):
        env = EnvironmentRobotTask(self.config["EnvTarget"]["env_config"])
        env_wrapper = EnvWrapper(env)
        return env_wrapper

    @cached_property
    def loss_function(self):
        env_source = EnvironmentRobotTask(self.config["EnvSource"]["env_config"])
        env_target = EnvironmentRobotTask(self.config["EnvTarget"]["env_config"])

        link_positions_source = self.env_source.dht_model(
            torch.zeros(
                (1,) + env_source.state_space["robot"]["arm"]["joint_positions"].shape
            ).to(self.device)
        )[0, :, :3, -1]
        link_positions_target = self.env_target.dht_model(
            torch.zeros(
                (1,) + env_target.state_space["robot"]["arm"]["joint_positions"].shape
            ).to(self.device)
        )[0, :, :3, -1]

        weight_matrix_ST_p, weight_matrix_ST_o = get_weight_matrices(
            link_positions_source,
            link_positions_target,
            self.config[self.__class__.__name__]["model"]["weight_matrix_exponent_p"],
        )

        loss_soft_dtw = SoftDTW(
            use_cuda=True,
            dist_func=KinematicChainLoss(
                weight_matrix_ST_p, weight_matrix_ST_o, reduction=False
            ),
        )

        loss_soft_dtw.to(self.device)

        return loss_soft_dtw

    def configure_optimizers(self):
        optimizer_action_mapper = torch.optim.AdamW(
            chain(self.trajectory_encoder.parameters(), self.policy.parameters()),
            lr=self.config[self.__class__.__name__]["train"].get("lr", 3e-4),
        )
        optimizers_state_mapper = self.state_mapper.configure_optimizers()

        return [optimizer_action_mapper, *optimizers_state_mapper]

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
        states_source, actions_source = batch
        with torch.no_grad():
            # bz
            behaviors = self.trajectory_encoder(states_source, actions_source)
            # bs
            states_target = self.state_mapper(states_source[:, 0])

            for _ in range(states_source.shape[1]):
                # b(s+z)
                states_target_behaviors = torch.concat(
                    (states_target, behaviors), dim=-1
                )
                # ba
                actions_target = self.policy(states_target_behaviors)

        return actions_target

    def loss(self, batch):
        # batch x len x s_dim_S, batch x len x a_dim_S
        trajectories_states_source, trajectories_actions_source = batch

        # batch x z_dim
        behaviors = self.trajectory_encoder(
            trajectories_states_source, trajectories_actions_source
        )

        trajectories_states_target = []

        # batch, s_dim_T
        states_target = self.state_mapper(trajectories_states_source[:, 0])
        trajectories_states_target.append(states_target)

        for _ in range(trajectories_actions_source.shape[1]):
            # batch x (s_dim_S + dim_z)
            states_target_behaviors = torch.concat((states_target, behaviors), dim=-1)

            # batch x a_dim_T
            actions_target = self.policy(states_target_behaviors)
            # batch x (s_dim_T + a_dim_T)
            states_actions_target = torch.concat(
                (states_target, actions_target), dim=-1
            )
            # batch x s_dim_T
            states_target = self.transition_model(states_actions_target)

            trajectories_states_target.append(states_target)

        # bls
        trajectories_states_target = torch.stack(trajectories_states_target).swapdims(
            0, 1
        )

        # (b*l)s
        states_source = trajectories_states_source.reshape(
            -1, trajectories_states_source.shape[-1]
        )
        states_target = trajectories_states_target.reshape(
            -1, trajectories_states_target.shape[-1]
        )

        # (b*l)p44
        link_poses_source = self.env_source.dht_model(states_source)
        link_poses_target = self.env_target.dht_model(states_target)

        # blp44
        link_poses_source = link_poses_source.reshape(
            *trajectories_states_source.shape[:2], *link_poses_source.shape[1:]
        )
        link_poses_target = link_poses_target.reshape(
            *trajectories_states_target.shape[:2], *link_poses_target.shape[1:]
        )

        loss = self.loss_function(link_poses_source, link_poses_target).mean()
        log_dict = {
            f"loss_{self.log_id}": loss,
        }

        return loss, log_dict

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        optimizer_action_mapper = optimizers[0]
        optimizers_state_mapper = optimizers[1:]

        loss_state_mapper, log_dict = self.state_mapper.training_step_(
            optimizers_state_mapper, batch, batch_idx
        )

        loss, log_dict_ = self.loss(batch["source"])
        log_dict.update(log_dict_)

        optimizer_action_mapper.zero_grad()
        self.manual_backward(loss)
        optimizer_action_mapper.step()

        log_dict = {"train_" + k: v for k, v in log_dict.items()}
        self.log_dict(log_dict, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.loss(batch["source"])

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
