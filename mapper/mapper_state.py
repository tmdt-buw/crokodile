import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
from pytorch_lightning.trainer.supporters import CombinedLoader
from world_models.discriminator import DiscriminatorSource, DiscriminatorTarget
from models.dht import get_dht_model
import logging
import itertools

sys.path.append(str(Path(__file__).resolve().parents[1]))

from functools import cached_property

from config import data_folder
from stage import LitStage
from functools import cached_property
from environments.environment_robot_task import EnvironmentRobotTask
from utils.nn import KinematicChainLoss, create_network, get_weight_matrices


class StateMapper(LitStage):
    def __init__(self, config, **kwargs):
        super(StateMapper, self).__init__(config, **kwargs)

    @classmethod
    def get_relevant_config(cls, config):
        return {
            **super(StateMapper, cls).get_relevant_config(config),
            **Discriminator.get_relevant_config(config),
        }

    def get_state_mapper(self, env_from_config, env_to_config):
        env_from = EnvironmentRobotTask(env_from_config["env_config"])
        env_to = EnvironmentRobotTask(env_to_config["env_config"])

        state_mapper = create_network(
            in_dim=env_from.state_space["robot"]["arm"]["joint_positions"].shape[-1],
            out_dim=env_to.state_space["robot"]["arm"]["joint_positions"].shape[-1],
            **self.config[self.__class__.__name__]["model"],
        )

        return state_mapper

    @cached_property
    def state_mapper_source_target(self):
        return self.get_state_mapper(self.config["EnvSource"], self.config["EnvTarget"])

    @cached_property
    def state_mapper_target_source(self):
        return self.get_state_mapper(self.config["EnvTarget"], self.config["EnvSource"])

    def get_state_dict(self):
        state_dict = {
            "state_mapper_source_target": self.state_mapper_source_target.state_dict()
        }

        if self.config[self.__class__.__name__]["train"].get(
            "cycle_consistency_factor", 0.0
        ):
            state_dict[
                "state_mapper_target_source"
            ] = self.state_mapper_target_source.state_dict()

        return state_dict

    def set_state_dict(self, state_dict):
        self.state_mapper_source_target.load_state_dict(
            state_dict["state_mapper_source_target"]
        )

        if "state_mapper_target_source" in state_dict:
            self.state_mapper_target_source.load_state_dict(
                state_dict["state_mapper_target_source"]
            )
        elif self.config[self.__class__.__name__]["train"].get(
            "cycle_consistency_factor", 0.0
        ):
            logging.warning(
                "No state dict for state_mapper_target_source found, but cycle_consistency_factor != 0."
            )

    @cached_property
    def dht_models(self):
        env_source = EnvironmentRobotTask(self.config["EnvSource"]["env_config"])
        env_target = EnvironmentRobotTask(self.config["EnvTarget"]["env_config"])

        return deepcopy(env_source.robot.dht_model), deepcopy(
            env_target.robot.dht_model
        )

    @cached_property
    def discriminator_source(self):
        return deepcopy(DiscriminatorSource(self.config).discriminator)

    @cached_property
    def discriminator_target(self):
        return deepcopy(DiscriminatorTarget(self.config).discriminator)

    @cached_property
    def loss_function(self):
        env_source = EnvironmentRobotTask(self.config["EnvSource"]["env_config"])
        env_target = EnvironmentRobotTask(self.config["EnvTarget"]["env_config"])

        dummy_state_source = torch.zeros(
            (1, env_source.state_space["robot"]["arm"]["joint_positions"].shape[-1])
        )
        dummy_state_target = torch.zeros(
            (1, env_target.state_space["robot"]["arm"]["joint_positions"].shape[-1])
        )

        dht_model_source, dht_model_target = self.dht_models

        dht_model_source.to(dummy_state_source)
        dht_model_target.to(dummy_state_target)

        link_positions_source = dht_model_source(dummy_state_source)[0, :, :3, -1]
        link_positions_target = dht_model_target(dummy_state_target)[0, :, :3, -1]

        weight_matrix_p, weight_matrix_o = get_weight_matrices(
            link_positions_source,
            link_positions_target,
            self.config[self.__class__.__name__]["model"]["weight_matrix_exponent_p"],
        )
        loss_fn_ = KinematicChainLoss(weight_matrix_p, weight_matrix_o)

        def loss_fn(states_source, states_target):
            env_source.robot.state2angle.to(states_source)
            env_target.robot.state2angle.to(states_target)

            angles_source = env_source.robot.state2angle(states_source)
            angles_target = env_target.robot.state2angle(states_target)

            dht_model_source.to(angles_source)
            dht_model_target.to(angles_target)

            link_poses_source = dht_model_source(angles_source)
            link_poses_target = dht_model_target(angles_target)

            loss_fn_.to(link_poses_source)

            loss = loss_fn_(link_poses_source, link_poses_target)

            return loss

        return loss_fn

    def configure_optimizers(self):
        parameters = [self.state_mapper_source_target.parameters()]

        if self.config[self.__class__.__name__]["train"].get(
            "cycle_consistency_factor", 0.0
        ):
            parameters.append(self.state_mapper_target_source.parameters())

        optimizer_state_mapper = torch.optim.AdamW(
            itertools.chain(*parameters),
            lr=self.config[self.__class__.__name__]["train"].get("lr", 3e-4),
        )
        return optimizer_state_mapper

    def train_dataloader(self):
        dataloaders = {}

        dataloaders["source"] = self.get_dataloader(
            self.config[self.__class__.__name__]["data"]["data_file_X"], "train"
        )

        if self.config[self.__class__.__name__]["train"].get(
            "cycle_consistency_factor", 0.0
        ):
            dataloaders["target"] = self.get_dataloader(
                self.config[self.__class__.__name__]["data"]["data_file_Y"], "train"
            )
        return CombinedLoader(dataloaders)

    def val_dataloader(self):
        dataloaders = {}

        dataloaders["source"] = self.get_dataloader(
            self.config[self.__class__.__name__]["data"]["data_file_X"], "test", False
        )

        if self.config[self.__class__.__name__]["train"].get(
            "cycle_consistency_factor", 0.0
        ):
            dataloaders["target"] = self.get_dataloader(
                self.config[self.__class__.__name__]["data"]["data_file_Y"],
                "test",
                False,
            )
        return CombinedLoader(dataloaders)

    def forward(self, states_source):
        states_target = self.state_mapper(states_source)
        return states_target

    def loss(
        self,
        batch_X,
        state_mapper_XY,
        loss_function,
        discriminator_Y=None,
        state_mapper_YX=None,
    ):
        trajectories_states_X, _ = batch_X

        states_X = trajectories_states_X.reshape(-1, trajectories_states_X.shape[-1])

        state_mapper_XY.to(states_X)

        states_Y = state_mapper_XY(states_X)

        loss_state_mapper_XY = loss_function(states_X, states_Y)

        loss = loss_state_mapper_XY

        if discriminator_Y and self.config[self.__class__.__name__]["train"].get(
            "discriminator_factor", 0.0
        ):
            # discriminator loss
            discriminator_Y.to(states_Y)
            outputs = discriminator_Y(states_Y)
            dist = torch.sum((outputs - self.discriminator_Y.c) ** 2, dim=1)
            loss_discriminator = torch.mean(relu(dist - self.discriminator_Y.radius))
            loss += (
                self.config[self.__class__.__name__]["train"]["discriminator_factor"]
                * loss_discriminator
            )

        if state_mapper_YX and self.config[self.__class__.__name__]["train"].get(
            "cycle_consistency_factor", 0.0
        ):
            # cycle consistency loss
            states_X_ = state_mapper_YX(states_Y)
            loss_cycle_consistency = torch.nn.functional.mse_loss(states_X, states_X_)

            loss += (
                self.config[self.__class__.__name__]["train"][
                    "cycle_consistency_factor"
                ]
                * loss_cycle_consistency
            )

        return loss

    def step(self, batch, batch_idx, prefix=""):
        loss_source_target = self.loss(
            batch["source"],
            self.state_mapper_source_target,
            self.loss_function,
            self.discriminator_target,
            self.state_mapper_target_source,
        )

        self.log(
            f"{prefix}loss_source_target_{self.log_id}",
            loss_source_target,
            on_step=False,
            on_epoch=True,
        )

        loss_target_source = self.loss(
            batch["target"],
            self.state_mapper_target_source,
            lambda t, s: self.loss_function(s, t),
            self.discriminator_source,
            self.state_mapper_source_target,
        )

        self.log(
            f"{prefix}loss_target_source_{self.log_id}",
            loss_target_source,
            on_step=False,
            on_epoch=True,
        )

        loss = loss_source_target + loss_target_source

        self.log(
            f"{prefix}loss_{self.log_id}",
            loss,
            on_step=False,
            on_epoch=True,
        )

        return loss

    """
        Perform training step. Customized behavior to log different loss compoents.
        Refer to pytorch lightning docs.
    """

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, "train_")

        return loss

    """
        Perform validation step. Customized behavior to log different loss compoents.
        Refer to pytorch lightning docs.
    """

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, "val_")

        return loss

    @classmethod
    def get_relevant_config(cls, config):
        config_ = super().get_relevant_config(config)

        config_["EnvSource"] = config["EnvSource"]
        config_["EnvTarget"] = config["EnvTarget"]

        return config_
