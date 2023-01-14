import os
import sys
from pathlib import Path
import wandb
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import data_folder
from stage import Stage


class LitModel(LightningModule, Stage):
    def __init__(self, config):
        LightningModule.__init__(self)
        Stage.__init__(self, config)

    def save(self, path=None):
        if path is None and self.config["cache"]["mode"] == "wandb":
            with wandb.init(
                id=self.run_id, **self.config["wandb_config"], resume="must"
            ) as run:
                self.save(self.tmpdir)

                artifact = wandb.Artifact(name=self.hash, type=self.__class__.__name__)
                artifact.add_dir(self.tmpdir)
                run.log_artifact(artifact)
        elif os.path.exists(path):
            logging.info("Pytorch Lightning models are saved while training.")
        else:
            raise ValueError(f"Invalid path: {path}")

    def generate(self):
        self.model = self.get_model()

        with wandb.init(
            config=self.get_relevant_config(self.config),
            **self.config.get("wandb_config", {"mode": "disabled"}),
            group=self.__class__.__name__,
            tags=[self.hash],
        ) as run:
            self.run_id = run.id
            wandb_logger = WandbLogger(save_dir=self.tmpdir)
            callbacks = [
                # todo: adjust wandb tag to be based on env config once static data files are removed
                ModelCheckpoint(
                    monitor=f"validation_loss_{self.__class__.__name__}",
                    mode="min",
                    filename=f"checkpoint_{self.hash}",
                )
            ]

            self.trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                max_time="00:07:55:00",
                max_epochs=self.config[self.__class__.__name__]["train"]["max_epochs"],
                logger=wandb_logger,
                callbacks=callbacks,
                # fast_dev_run=False,
            )
            self.trainer.fit(self)

    def load(self, path=None):
        if path is None and self.config["cache"]["mode"] == "wandb":
            wandb_config = self.config["wandb_config"]
            wandb_checkpoint_path = (
                f"{wandb_config['entity']}/"
                f"{wandb_config['project']}/"
                f"{self.hash}:latest"
            )
            logging.info(f"wandb artifact: {wandb_checkpoint_path}")

            wandb.Api().artifact(wandb_checkpoint_path).download(self.tmpdir)

            checkpoint_folders = glob.glob(self.tmpdir + "**/**/*.ckpt", recursive=True)

            assert checkpoint_folders, f"No checkpoints found in {self.tmpdir}"

            if len(checkpoint_folders) > 1:
                logging.warning(
                    f"More than one checkpoint folder found: " f"{checkpoint_folders}"
                )

            checkpoint_path = checkpoint_folders[0]

            self.load(checkpoint_path)
        elif os.path.exists(path):
            # self.model_config.update({"num_workers": 1, "num_gpus": 0})
            self.model = self.model_cls.load_from_checkpoint(
                checkpoint_path=path,
                map_location=torch.device("cpu"),
                config=self.config,
            )
        else:
            raise ValueError(f"Invalid path: {path}")

    @classmethod
    def get_relevant_config(cls, config):
        return {
            "data": config[cls.__name__]["data"],
            # "log_suffix": config[cls.__name__]["log_suffix"],
            "model": config[cls.__name__]["model"],
            "train": config[cls.__name__]["train"],
            "callbacks": config[cls.__name__]["callbacks"],
        }

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
            batch_size=self.config[self.__class__.__name__]["train"]["batch_size"],
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
