import itertools
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.nn.functional import relu

from models.dht import get_dht_model
from world_models.discriminator import DiscriminatorSource, DiscriminatorTarget

sys.path.append(str(Path(__file__).resolve().parents[1]))

from functools import cached_property

from config import data_folder
from environments.environment_robot_task import EnvironmentRobotTask
from stage import LitStage
from utils.nn import KinematicChainLoss, create_network, get_weight_matrices


class EnvWrapper:
    def __init__(self, env):
        self.state_space = deepcopy(env.state_space)
        self.action_space = deepcopy(env.action_space)
        self.dht_model = deepcopy(env.robot.dht_model)
        self.state2angle = deepcopy(env.robot.state2angle)
        self.angle2state = deepcopy(env.robot.angle2state)


class StateMapper(LitStage):
    def __init__(self, config, **kwargs):
        super(StateMapper, self).__init__(config, **kwargs)

    def init_models(self, config, **kwargs):
        self.automatic_optimization = False

        self.state_mapper_source_target = self.get_state_mapper(
            config["EnvSource"],
            config["EnvTarget"],
            config[self.__class__.__name__]["model"],
        )
        self.state_mapper_target_source = self.get_state_mapper(
            config["EnvTarget"],
            config["EnvSource"],
            config[self.__class__.__name__]["model"],
        )

        self.discriminator_source = DiscriminatorSource(config, **kwargs)
        self.discriminator_target = DiscriminatorTarget(config, **kwargs)

    @classmethod
    def get_relevant_config(cls, config):
        return {
            **super(StateMapper, cls).get_relevant_config(config),
            **DiscriminatorSource.get_relevant_config(config),
            **DiscriminatorTarget.get_relevant_config(config),
        }

    def get_state_mapper(self, env_from_config, env_to_config, model_config):
        env_from = EnvironmentRobotTask(env_from_config["env_config"])
        env_to = EnvironmentRobotTask(env_to_config["env_config"])

        state_mapper = create_network(
            in_dim=env_from.state_space["robot"]["arm"]["joint_positions"].shape[-1],
            out_dim=env_to.state_space["robot"]["arm"]["joint_positions"].shape[-1],
            **model_config,
        )

        return state_mapper

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
    def env_source(self):
        env = EnvironmentRobotTask(self.config["EnvSource"]["env_config"])
        env_wrapper = EnvWrapper(env)
        return env_wrapper

    @cached_property
    def env_target(self):
        env = EnvironmentRobotTask(self.config["EnvTarget"]["env_config"])
        env_wrapper = EnvWrapper(env)
        return env_wrapper

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

        def loss_fn(states_source, states_target):
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
        optimizer_state_mapper_source_target = torch.optim.AdamW(
            self.state_mapper_source_target.parameters(),
            lr=self.config[self.__class__.__name__]["train"].get("lr", 3e-4),
        )
        optimizer_state_mapper_target_source = torch.optim.AdamW(
            self.state_mapper_target_source.parameters(),
            lr=self.config[self.__class__.__name__]["train"].get("lr", 3e-4),
        )

        optimizers = [
            optimizer_state_mapper_source_target,
            optimizer_state_mapper_target_source,
        ]

        optimizers.append(self.discriminator_source.configure_optimizers())
        optimizers.append(self.discriminator_target.configure_optimizers())

        return optimizers

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
        states_target = self.state_mapper_source_target(states_source)
        return states_target

    def loss_(
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
            output_discriminator_Y = discriminator_Y(states_Y)

            loss_discriminator = torch.nn.functional.binary_cross_entropy(
                output_discriminator_Y,
                torch.ones((states_Y.shape[0], 1), device=states_Y.device),
            )

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

    def loss(self, batch, batch_idx):
        loss_source_target = self.loss_(
            batch["source"],
            self.state_mapper_source_target,
            self.loss_function,
            self.discriminator_target,
            self.state_mapper_target_source,
        )

        loss_target_source = self.loss_(
            batch["target"],
            self.state_mapper_target_source,
            lambda t, s: self.loss_function(s, t),
            self.discriminator_source,
            self.state_mapper_source_target,
        )

        loss = loss_source_target + loss_target_source

        log_dict = {
            f"loss_source_target_{self.log_id}": loss_source_target,
            f"loss_target_source_{self.log_id}": loss_target_source,
            f"loss_{self.log_id}": loss,
        }

        return loss, log_dict

    """
        Perform training step. Customized behavior to log different loss compoents.
        Refer to pytorch lightning docs.
    """

    def training_step(self, batch, batch_idx):
        loss, log_dict = self.training_step_(self.optimizers(), batch, batch_idx)

        log_dict = {"train_" + k: v for k, v in log_dict.items()}

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def training_step_(self, optimizers, batch, batch_idx):
        (
            optimizer_state_mapper_source_target,
            optimizer_state_mapper_target_source,
            optimizer_discriminator_source,
            optimizer_discriminator_target,
        ) = optimizers

        loss, log_dict = self.loss(batch, batch_idx)

        optimizer_state_mapper_source_target.zero_grad()
        optimizer_state_mapper_target_source.zero_grad()
        self.manual_backward(loss)
        optimizer_state_mapper_source_target.step()
        optimizer_state_mapper_target_source.step()

        trajectories_states_source, _ = batch["source"]
        states_source = trajectories_states_source.reshape(
            -1, trajectories_states_source.shape[-1]
        )

        trajectories_states_target, _ = batch["target"]
        states_target = trajectories_states_target.reshape(
            -1, trajectories_states_target.shape[-1]
        )

        states_target_ = self.state_mapper_source_target(states_source)
        states_source_ = self.state_mapper_target_source(states_target)

        loss_discriminator_source, log_dict_ = self.discriminator_source.loss(
            {"true": states_source, "fake": states_source_}
        )
        log_dict.update(log_dict_)
        loss_discriminator_target, log_dict_ = self.discriminator_target.loss(
            {"true": states_target, "fake": states_target_}
        )
        log_dict.update(log_dict_)

        loss_discriminator = loss_discriminator_source + loss_discriminator_target

        optimizer_discriminator_source.zero_grad()
        optimizer_discriminator_target.zero_grad()
        self.manual_backward(loss_discriminator)
        optimizer_discriminator_source.step()
        optimizer_discriminator_target.step()

        return loss, log_dict

    """
        Perform validation step. Customized behavior to log different loss compoents.
        Refer to pytorch lightning docs.
    """

    def validation_step_(self, batch, batch_idx):
        return self.loss(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.validation_step_(batch, batch_idx)

        log_dict = {"validation_" + k: v for k, v in log_dict.items()}

        return loss

    @classmethod
    def get_relevant_config(cls, config):
        config_ = super().get_relevant_config(config)

        config_["EnvSource"] = config["EnvSource"]
        config_["EnvTarget"] = config["EnvTarget"]

        return config_
