"""
Train state mappings with dht models.
"""

import os
import sys
from pathlib import Path

from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from multiprocessing import cpu_count, set_start_method

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from config import data_folder, wandb_config
from models.dht import get_dht_model
from models.transformer import Seq2SeqTransformer
from utils.nn import (
    KinematicChainLoss,
    NeuralNetwork,
    create_network,
    init_xavier_uniform,
)


class LitStateMapper(pl.LightningModule):
    def __init__(
        self,
        data_file,
        state_mapper_config={},
        batch_size=32,
        num_workers=1,
        **kwargs,
    ):
        super(LitStateMapper, self).__init__()
        self.save_hyperparameters()

        data_path = os.path.join(data_folder, data_file)
        data = torch.load(data_path)

        data_A = data["A"]
        data_B = data["B"]

        self.state_mapper_AB = create_network(
            in_dim=data_A["states_train"].shape[1],
            out_dim=data_B["states_train"].shape[1],
            **state_mapper_config,
        )

        self.transition_model_B = create_network(
            in_dim=data_A["states_train"].shape[1],
            out_dim=data_B["states_train"].shape[1],
            **state_mapper_config,
        )

        # self.state_mapper_BA = torch.nn.Sequential(
        #     torch.nn.Transformer(d_model=1, nhead=1),
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(self.max_dof, self.max_dof)
        # )

        self.state_mapper_AB.apply(init_xavier_uniform)
        # self.state_mapper_BA.apply(init_xavier_uniform)

        self.dht_model_A = get_dht_model(data_A["dht_params"], data_A["joint_limits"])
        self.dht_model_B = get_dht_model(data_B["dht_params"], data_B["joint_limits"])

        self.loss_fn = MSELoss()

        # manual optimization in training_step
        self.automatic_optimization = False

        """
            Maps states A -> B
            Required by super class LightningModule

            Args:
                states_A
            Returns:
                Mapped states_B
        """

    def forward(self, states_A):
        return self.state_mapper_AB(states_A)

    """
        Generate dataloader used for training.
        Refer to pytorch lightning docs.
    """

    def train_dataloader(self):
        data_path = os.path.join(data_folder, self.hparams.data_file)
        data = torch.load(data_path)

        states_train_A = data["A"]["states_train"]
        states_train_B = data["B"]["states_train"]

        dataloader_train = DataLoader(
            TensorDataset(states_train_A, states_train_B),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
        )

        return dataloader_train

    """
        Generate dataloader used for validation.
        Refer to pytorch lightning docs.
    """

    def val_dataloader(self):
        data_path = os.path.join(data_folder, self.hparams.data_file)
        data = torch.load(data_path)

        states_validation_A = data["A"]["states_test"]
        states_validation_B = data["B"]["states_test"]

        dataloader_validation = DataLoader(
            TensorDataset(states_validation_A, states_validation_B),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

        return dataloader_validation

    """
        Perform training step. Customized behavior to enable accumulation of all losses into one variable.
        Refer to pytorch lightning docs.
    """

    def training_step(self, batch, batch_idx):
        optimizer_state_mapper_AB = self.optimizers()
        optimizer_state_mapper_AB.zero_grad()
        # optimizer_state_mapper_BA.zero_grad()

        loss_AB = self.step(batch, batch_idx)

        self.manual_backward(loss_AB)
        # self.manual_backward(loss_BA)

        optimizer_state_mapper_AB.step()
        # optimizer_state_mapper_BA.step()

        self.log("train_loss_AB", loss_AB, on_step=False, on_epoch=True)
        # self.log("train_loss_BA", loss_BA, on_step=False, on_epoch=True)

    """
        Perform validation step. Customized behavior to enable accumulation of all losses into one variable.
        Refer to pytorch lightning docs.
    """

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss_AB = self.step(batch, batch_idx)

        loss = loss_AB

        self.log("val_loss_AB", loss_AB.item(), on_step=False, on_epoch=True)
        # self.log("val_loss_BA", loss_BA.item(), on_step=False, on_epoch=True)
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True)

    def step(self, batch, batch_idx=None):
        states_A, states_B = batch

        states_B_ = torch.nn.functional.pad(
            states_B[:, :-1], (1, 0), mode="constant", value=torch.nan
        )

        states_B_pred = self.state_mapper_AB(states_A, states_B_)
        # states_A_ = self.state_mapper_BA(states_B)

        loss_AB = self.loss_fn(
            self.dht_model_B(states_B_pred)[:, -1, :3, -1],
            self.dht_model_B(states_B)[:, -1, :3, -1],
        )
        # loss_BA = self.loss_fn(states_A_, states_A)

        # loss = loss_AB + loss_BA

        return loss_AB

    def training_epoch_end(self, outputs):
        super(LitStateMapper, self).training_epoch_end(outputs)

        scheduler_state_mapper_AB = self.lr_schedulers()
        # scheduler_state_mapper_AB.step(self.trainer.callback_metrics["val_loss_AB"])

    """
        Helper function to generate all optimizers.
        Refer to pytorch lightning docs.
    """

    def configure_optimizers(self):
        optimizer_state_mapper_AB = torch.optim.Adam(
            self.state_mapper_AB.parameters(),
            lr=self.hparams.state_mapper_config.get("lr", 3e-4),
        )
        # optimizer_state_mapper_BA = torch.optim.Adam(self.state_mapper_BA.parameters(),
        #                                              lr=self.hparams.state_mapper_lr)

        scheduler_state_mapper_AB = {
            "scheduler": ReduceLROnPlateau(
                optimizer_state_mapper_AB, factor=0.9, patience=100
            ),
            "monitor": "validation_loss_state_mapper_AB",
            "name": "scheduler_optimizer_state_mapper_AB",
        }

        # scheduler_state_mapper_BA = {"scheduler": ReduceLROnPlateau(optimizer_state_mapper_BA),
        #                              "monitor": "validation_loss_state_mapper_BA",
        #                              "name": "scheduler_optimizer_state_mapper_BA"
        #                              }

        return [
            optimizer_state_mapper_AB,
            # optimizer_state_mapper_BA
        ], [
            scheduler_state_mapper_AB,
            # scheduler_state_mapper_BA
        ]


if __name__ == "__main__":
    wandb_config.update(
        {
            "group": "state_mapper",
            "tags": ["transformer", "tcp loss"],
        }
    )

    config = {
        "data_file": "panda-ur5_100000_1000.pt",
        "d_model": 8,
        "nhead": 1,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "dim_feedforward": 32,
        "dropout": 0.1,
        "lr": 3e-4,
        "batch_size": 64,
        "max_epochs": 10_000,
    }

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        devices = -1
        config["num_workers"] = cpu_count() // torch.cuda.device_count()
    else:
        devices = None
        config["num_workers"] = cpu_count()

    state_mapper = LitStateMapper(
        **config,
        state_mapper_dropout=config["dropout"],
        state_mapper_lr=config["lr"],
    )

    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min"),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val_loss", mode="min", patience=500),
    ]

    logger = WandbLogger(**wandb_config, log_model="all")
    # logger = TensorBoardLogger("results")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        # trainer = pl.Trainer(strategy=DDPStrategy(), accelerator="gpu", devices=devices,
        logger=logger,
        max_epochs=config["max_epochs"],
        callbacks=callbacks,
    )
    trainer.fit(state_mapper)
