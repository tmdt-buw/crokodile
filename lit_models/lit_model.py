import os
import sys
from pathlib import Path


import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import data_folder


class LitModel(pl.LightningModule):
    def __init__(self, config):
        super(LitModel, self).__init__()
        self.lit_config = config
        self.model = self.get_model()

    def get_model(self):
        raise NotImplementedError(
            f"get_model() not implemented for {self.__class__.__name__}"
        )

    def configure_optimizers(self):
        raise NotImplementedError(
            f"configure_optimizers() not implemented for {self.__class__.__name__}"
        )

    def get_dataloader(self, data_file, dataset_type="train", shuffle=True):
        data_path = os.path.join(data_folder, data_file)
        data = torch.load(data_path)

        trajectories_states = data[f"trajectories_states_{dataset_type}"].float()
        trajectories_actions = data[f"trajectories_actions_{dataset_type}"].float()

        dataloader = DataLoader(
            TensorDataset(trajectories_states, trajectories_actions),
            batch_size=self.lit_config["train"]["batch_size"],
            num_workers=os.cpu_count() - 1,
            shuffle=shuffle,
            pin_memory=True,
        )

        return dataloader

    def train_dataloader(self):
        raise NotImplementedError(
            f"train_dataloader() not implemented for {self.__class__.__name__}"
        )

    def val_dataloader(self):
        raise NotImplementedError(
            f"val_dataloader() not implemented for {self.__class__.__name__}"
        )

    def forward(self, x):
        raise NotImplementedError(
            f"forward() not implemented for {self.__class__.__name__}"
        )

    def loss(self, batch):
        raise NotImplementedError(
            f"loss() not implemented for {self.__class__.__name__}"
        )

    def training_step(self, batch, batch_idx):
        raise NotImplementedError(
            f"training_step() not implemented for {self.__class__.__name__}"
        )

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError(
            f"validation_step() not implemented for {self.__class__.__name__}"
        )
