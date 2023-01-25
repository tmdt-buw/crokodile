import os
import sys
from functools import cached_property
from pathlib import Path
from typing import Dict

import torch
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import data_folder
from stage import LitStage
from utils.nn import create_network
from environments.environment_robot_task import EnvironmentRobotTask


class DeepSVDD(torch.nn.Module):
    """A class for the Deep SVDD method."""

    def __init__(self, config: Dict):
        """Inits DeepSVDD with the given parameters..

        Args:
            config (Dict): {
                "objective": "one-class" or "soft-boundary"
                "nu": value in [0, 1)
                "radius": radius of hypersphere,
                "in_dim": input dimension of neural network,
                "out_dim": output dimension of neural network,
                "network_width": number of neurons for linear layers,
                "network_depth": number of layers,
                "dropout": droput probability
            }
        """
        super(DeepSVDD, self).__init__()
        assert config["objective"] in (
            "one-class",
            "soft-boundary",
        ), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = config["objective"]
        # fraction of data which is are outliers
        assert (0 < config["nu"]) & (
            config["nu"] <= 1
        ), "Hyperparameter nu should be in (0, 1]."
        self.nu = config["nu"]
        # hypersphere radius R
        self.radius = torch.nn.Parameter(
            torch.tensor(config.get("radius", 0.0)), requires_grad=False
        )
        # hypersphere center c
        self.c = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.warmup_epochs = config.get("warmup_epochs", 10)
        # neural network \phi
        self.rep_dim = config["out_dim"]
        self.net = create_network(
            in_dim=config["in_dim"],
            out_dim=self.rep_dim,
            network_width=config["network_width"],
            network_depth=config["network_depth"],
            dropout=config["dropout"],
            bias=False,
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class DeepSVDDLoss(torch.nn.Module):
    def __init__(self, objective, warmup_epochs):
        super(DeepSVDDLoss, self).__init__()
        self.objective = objective
        self.warmup_epochs = warmup_epochs

    def forward(self, x, discriminator, current_epoch=-1):
        radius = None
        dist = torch.sum((x - discriminator.c) ** 2, dim=1)
        if self.objective == "soft-boundary":
            scores = dist - discriminator.radius**2
            loss = discriminator.radius**2 + (1 / discriminator.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores)
            )
            # Update hypersphere radius R on mini-batch distances
            if current_epoch > self.warmup_epochs:
                radius = torch.tensor(self.get_radius(dist, discriminator.nu))
        else:
            scores = dist
            loss = torch.mean(dist)

        return loss, scores, radius

    @staticmethod
    def get_radius(dist: torch.Tensor, nu: float):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return torch.quantile(torch.sqrt(dist), 1 - nu)


class Discriminator(LitStage):
    def __init__(self, config, **kwargs):
        super(Discriminator, self).__init__(config, **kwargs)

    @cached_property
    def discriminator(self):
        env = EnvironmentRobotTask(self.config["EnvTarget"]["env_config"])

        state_dim = env.state_space["robot"]["arm"]["joint_positions"].shape[-1]

        self.config[self.__class__.__name__]["model"]["in_dim"] = state_dim

        discriminator = DeepSVDD(self.config[self.__class__.__name__]["model"])
        # init center c with initial forward pass of some data
        data_c = torch.randn(
            self.config[self.__class__.__name__]["model"]["init_center_samples"],
            state_dim,
        )
        discriminator.eval()
        with torch.no_grad():
            output = discriminator(data_c)
            c = torch.mean(output, dim=0)
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        eps = self.config[self.__class__.__name__]["model"]["eps"]
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        discriminator.c.data = c

        return discriminator

    def get_state_dict(self):
        return self.discriminator.state_dict()

    def set_state_dict(self, state_dict):
        self.discriminator.load_state_dict(state_dict)

    @cached_property
    def loss_function(self):
        return DeepSVDDLoss(
            self.config[self.__class__.__name__]["model"]["objective"],
            self.config[self.__class__.__name__]["model"]["warmup_epochs"],
        )

    def configure_optimizers(self):
        optimizer_discriminator = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config[self.__class__.__name__]["train"].get("lr", 3e-4),
        )
        scheduler_disc = {
            "scheduler": MultiStepLR(
                optimizer_discriminator,
                [
                    self.config[self.__class__.__name__]["train"].get(
                        "scheduler_epoch", 150
                    )
                ],
                self.config[self.__class__.__name__]["train"].get("lr_decrease", 0.1),
            ),
            "name": "scheduler_lr_disc",
        }

        return [optimizer_discriminator], [scheduler_disc]

    def loss(self, batch):
        trajectories_states, _ = batch
        states_x = trajectories_states.reshape(-1, trajectories_states.shape[-1])
        self.discriminator.to(states_x)
        outputs = self.discriminator(states_x)
        loss, scores, radius = self.loss_function(
            outputs, self.discriminator, self.current_epoch
        )
        if radius:
            self.discriminator.radius.data = radius
        return loss

    @classmethod
    def get_relevant_config(cls, config):
        config_ = super().get_relevant_config(config)

        config_["EnvTarget"] = config["EnvTarget"]

        return config_
