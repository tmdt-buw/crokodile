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
import wandb
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import data_folder


class Stage:
    def __init__(self, config, run=True, **kwargs):
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

        if run:
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
                        f"Loading cache not possible "
                        f"for {self.__class__.__name__}. "
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
        except FileNotFoundError:
            logging.info(f"{self.tmpdir} already deleted.")

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
    def __init__(self, config, load_checkpoint=False, **kwargs):
        LightningModule.__init__(self)
        Stage.__init__(self, config, run=not load_checkpoint)

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
            state_dict = self.get_state_dict()
            torch.save(state_dict, os.path.join(path, f"state_dict_{self.hash}.pt"))
        else:
            raise ValueError(f"Invalid path: {path}")

    def generate(self):
        with wandb.init(
            config=self.get_relevant_config(self.config),
            **self.config.get("wandb_config", {"mode": "disabled"}),
            group=self.__class__.__name__,
            tags=[self.hash],
        ) as run, tempfile.TemporaryDirectory() as tmpdir:
            self.run_id = run.id
            wandb_logger = WandbLogger(log_model=False, save_dir=tmpdir)
            checkpoint_callback = ModelCheckpoint(
                monitor=f"validation_loss_{self.log_id}",
                mode="min",
                filename=f"checkpoint_{self.hash}",
            )

            trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                max_time="00:07:55:00",
                max_epochs=self.config[self.__class__.__name__]["train"]["max_epochs"],
                logger=wandb_logger,
                callbacks=[checkpoint_callback],
            )
            trainer.fit(self)

            self.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                config=self.config,
                load_checkpoint=True,
            )

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

            checkpoint_path = os.path.join(self.tmpdir, f"state_dict_{self.hash}.pt")

            assert os.path.isfile(
                checkpoint_path
            ), f"Checkpoint file not found {checkpoint_path}"

            self.load(checkpoint_path)
        elif os.path.exists(path):
            state_dict = torch.load(path)
            self.set_state_dict(state_dict)
        else:
            raise ValueError(f"Invalid path: {path}")

    def get_state_dict(self):
        raise NotImplementedError()

    def set_state_dict(self, state_dict):
        raise NotImplementedError()

    @classmethod
    def get_relevant_config(cls, config):
        return {
            cls.__name__: config[cls.__name__],
        }

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
        # todo: remove dependency on data file
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
