from typing import Dict

import os
import sys
from pathlib import Path
from copy import deepcopy

import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nn import create_network
from lit_model import LitModel, LitTrainer
data_folder = "data"

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
            scores = dist - discriminator.radius ** 2
            loss = discriminator.radius ** 2 + (1 / discriminator.nu) * torch.mean(
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


class LitDiscriminator(LitModel):
    def __init__(
            self,
            config
    ):
        super(LitDiscriminator, self).__init__(config)
        self.loss_function = self.get_loss()

    def get_model(self):
        data_path = os.path.join(data_folder, self.lit_config["data"])
        data = torch.load(f=data_path)
        self.lit_config["model"]["in_dim"] = data["trajectories_states_train"].shape[-1]
        discriminator_model = DeepSVDD(self.lit_config["model"])
        # init center c with initial forward pass of some data
        data_c = data["trajectories_states_train"][:, :-1].reshape(
            -1, data["trajectories_states_train"].shape[-1]
        )[: self.lit_config["model"]["init_center_samples"], :]
        discriminator_model.eval()
        with torch.no_grad():
            output = discriminator_model(data_c)
            c = torch.mean(output, dim=0)
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        eps = self.lit_config["model"]["eps"]
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        discriminator_model.c.data = c

        return discriminator_model

    def get_loss(self):
        discriminator_loss = DeepSVDDLoss(
            self.lit_config["model"]["objective"],
            self.lit_config["model"]["warmup_epochs"],
        )
        return discriminator_loss

    def configure_optimizers(self):
        optimizer_discriminator = torch.optim.AdamW(
            self.model.parameters(), lr=self.lit_config["train"].get("lr", 3e-4)
        )
        scheduler_disc = {
            "scheduler": MultiStepLR(
                optimizer_discriminator,
                [self.lit_config["train"].get("scheduler_epoch", 150)],
                self.lit_config["train"].get("lr_decrease", 0.1),
            ),
            "name": "scheduler_lr_disc",
        }

        return [optimizer_discriminator], [scheduler_disc]

    def train_dataloader(self):
        return self.get_dataloader(self.lit_config["data"], "train")

    def val_dataloader(self):
        return self.get_dataloader(self.lit_config["data"], "test", False)

    def forward(self, x):
        with torch.no_grad():
            embedded_state = self.discriminator(x)
        return embedded_state

    def loss(self, batch):
        trajectories_states, _ = batch
        states_x = trajectories_states.reshape(-1, trajectories_states.shape[-1])
        outputs = self.discriminator(states_x)
        loss, scores, radius = self.discriminator_loss(
            outputs, self.discriminator, self.current_epoch
        )
        if radius:
            self.discriminator.R.data = radius
        return loss, scores

    def training_step(self, batch, batch_idx):
        loss, _ = self.loss(batch)
        self.log(
            "train_loss_discriminator" + self.lit_config["log_suffix"],
            loss.item(),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.loss(batch)
        self.log(
            "validation_loss_discriminator" + self.lit_config["log_suffix"],
            loss.item(),
            on_step=False,
            on_epoch=True,
        )
        return loss



class Discriminator(LitTrainer):
    def __init__(self, config):
        self.model_cls = config["Discriminator"]["model_cls"]
        self.model_config = config["Discriminator"]

        super(Discriminator, self).__init__(config)

    def generate(self):
        super(Discriminator, self).generate()
        self.train()

    @classmethod
    def get_relevant_config(cls, config):
        return super(Discriminator, cls).get_relevant_config(config)




def main():
    data_file_A = "panda_5_200_40.pt"
    data_file_B = "ur5_5_200_40.pt"

    config_A = {
        "Discriminator": {
            "model_cls": "discriminator",
            "data": data_file_A,
            "log_suffix": "_A",
            "model": {
                "objective": "soft-boundary",
                "eps": 0.01,
                "init_center_samples": 100,
                "nu": 1e-3,
                "out_dim": 4,
                "network_width": 256,
                "network_depth": 4,
                "dropout": 0.2,
                "warmup_epochs": 10,
            },
            "train": {
                "batch_size": 512,
                "lr": 1e-3,
                "scheduler_epoch": 150,
                "lr_decrease": 0.1,
            },
            "callbacks": {
            }
        }
    }

    config_B = deepcopy(config_A)
    config_B["data"] = data_file_B
    config_B["log_suffix"] = "_B"

    model_str = f"test"

    callbacks_A = [
        ModelCheckpoint(monitor="validation_loss_discriminator_A", mode="min", filename=f"{model_str}_A"),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="validation_loss_discriminator_A",
            min_delta=1e-5,
            mode="min",
            patience=250,
        ),
    ]

    discriminator_model_A = LitDiscriminator(
        config=config_A
    )
    trainer = pl.Trainer(
        max_epochs=1000,
        max_time="00:07:55:00",
        accelerator="gpu",
        default_root_dir="models/model_split/lightning_checkpoint/discriminator/",
        callbacks=callbacks_A,
        fast_dev_run=True
    )
    trainer.fit(discriminator_model_A)


if __name__ == "__main__":
    main()
