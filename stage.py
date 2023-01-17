import glob
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

import wandb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import data_folder


class Stage:
    def __init__(self, config):
        self.config = config
        self.hash = self.get_config_hash(config)
        self.tmpdir = tempfile.mkdtemp()

        self.log_prefix = config[self.__class__.__name__].get("log_prefix", "")
        self.log_suffix = config[self.__class__.__name__].get("log_suffix", "")
        self.log_id = self.log_prefix + self.__class__.__name__ + self.log_suffix

        if "cache" not in config or config["cache"]["mode"] == "disabled":
            load = False
            save = False
        else:
            load = config["cache"].get("load", False)
            if type(load) is str:
                load = self.__class__.__name__ == load
            elif type(load) is list:
                load = self.__class__.__name__ in load
            elif type(load) is dict:
                load = load.get(self.__class__.__name__, False)
                if type(load) is str:
                    self.hash = load
                    load = True

            save = config["cache"].get("save", False)
            if type(save) is str:
                save = self.__class__.__name__ == save
            elif type(save) is list:
                save = self.__class__.__name__ in save
            elif type(save) is dict:
                save = save.get(self.__class__.__name__, False)

            assert (
                type(load) is bool and type(save) is bool
            ), f"Invalid cache config: {config['cache']}"

        logging.info(
            f"Stage {self.__class__.__name__}:"
            f"\n\thash: {self.hash}"
            f"\n\ttmpdir: {self.tmpdir}"
            f"\n\tload: {load}"
            f"\n\tsave: {save}"
        )

        if load:
            try:
                logging.info(f"Loading cache for {self.__class__.__name__}.")

                self.load()
                # Don't save again if we loaded
                save = False
            except Exception as e:
                logging.warning(
                    f"Loading cache not possible " f"for {self.__class__.__name__}. "
                )
                logging.exception(e)
                self.generate()
        else:
            logging.info(f"Generating {self.__class__.__name__}.")

            self.generate()

        if save:
            logging.info(f"Saving {self.__class__.__name__}.")

            self.save()

    def __del__(self):
        try:
            shutil.rmtree(self.tmpdir)
        except AttributeError:
            pass

    def generate(self):
        raise NotImplementedError(
            f"generate() not implemented for {self.__class__.__name__}"
        )

    def load(self):
        raise NotImplementedError(
            f"load() not implemented for {self.__class__.__name__}"
        )

    def save(self):
        raise NotImplementedError(
            f"save() not implemented for {self.__class__.__name__}"
        )

    @classmethod
    def get_relevant_config(cls, config):
        raise NotImplementedError(
            f"get_relevant_config() not implemented for {cls.__name__}"
        )

    @classmethod
    def get_config_hash(cls, config):
        return hashlib.sha256(
            json.dumps(
                cls.get_relevant_config(config),
                default=lambda o: "<not serializable>",
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:6]


class LitStage(LightningModule, Stage):
    def __init__(self, config):
        LightningModule.__init__(self)
        Stage.__init__(self, config)

    """Stage methods"""

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
                    monitor=f"validation_loss_{self.log_id}",
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
            # todo: call wandb.finish()?

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
            # todo: must be custom for each model, as some stages (e.g. MapperWeaSCL) have multiple models
            raise NotImplementedError()
            self.model = self.model_cls.load_from_checkpoint(
                checkpoint_path=path,
                map_location=torch.device("cpu"),
                config=self.config,
            )
        else:
            raise ValueError(f"Invalid path: {path}")

    @classmethod
    def get_relevant_config(cls, config):
        return config[cls.__name__]

    """LightningModule methods"""

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
        return self.get_dataloader(
            self.config[self.__class__.__name__]["data"], "train"
        )

    def val_dataloader(self):
        return self.get_dataloader(
            self.config[self.__class__.__name__]["data"], "test", False
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, batch):
        raise NotImplementedError(
            f"loss() not implemented for {self.__class__.__name__}"
        )

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log(
            f"train_loss_{self.__class__.__name__}",
            loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log(
            f"validation_loss_{self.log_id}",
            loss,
            on_step=False,
            on_epoch=True,
        )
        return loss
