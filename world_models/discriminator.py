import os
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import data_folder
from environments.environment_robot_task import EnvironmentRobotTask
from stage import LitStage
from utils.nn import create_network
from environments.environment_robot_task import EnvironmentRobotTask
from environments import get_env


class Discriminator(LitStage):
    def __init__(self, config, **kwargs):
        super(Discriminator, self).__init__(config, **kwargs)

    def init_models(self, config, **kwargs):
        # load environment to get correct state space dimensions
        env_config = config[self.env_type]["env_config"]
        env_config["name"] = config[self.env_type]["env"]
        env = get_env(env_config)

        self.discriminator = create_network(
            in_dim=env.state_space["robot"]["arm"]["joint_positions"].shape[-1],
            out_dim=1,
            **config[self.__class__.__name__]["model"],
        )

    def forward(self, x):
        return self.discriminator(x)

    def get_state_dict(self):
        return self.discriminator.state_dict()

    def set_state_dict(self, state_dict):
        self.discriminator.load_state_dict(state_dict)

    def configure_optimizers(self):
        optimizer_discriminator = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config[self.__class__.__name__]["train"].get("lr", 3e-4),
        )
        # scheduler_disc = {
        #     "scheduler": MultiStepLR(
        #         optimizer_discriminator,
        #         [
        #             self.config[self.__class__.__name__]["train"].get(
        #                 "scheduler_epoch", 150
        #             )
        #         ],
        #         self.config[self.__class__.__name__]["train"].get("lr_decrease", 0.1),
        #     ),
        #     "name": "scheduler_lr_disc",
        # }

        return optimizer_discriminator # , [scheduler_disc]

    def loss(self, batch):
        labels_true = self.discriminator(batch["true"])
        labels_fake = self.discriminator(batch["fake"])

        loss = torch.nn.functional.binary_cross_entropy(
            labels_true, torch.ones_like(labels_true)
        ) + torch.nn.functional.binary_cross_entropy(
            labels_fake, torch.zeros_like(labels_fake))

        return loss

    @classmethod
    def get_relevant_config(cls, config):
        config_ = super().get_relevant_config(config)

        config_[cls.env_type] = config[cls.env_type]

        return config_


class DiscriminatorSource(Discriminator):
    env_type = "EnvSource"


class DiscriminatorTarget(Discriminator):
    env_type = "EnvTarget"
