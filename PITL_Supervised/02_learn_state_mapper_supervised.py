"""
Train state mappings with dht models.
"""

import os
import sys
from pathlib import Path
import tempfile

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import wandb
from torch.multiprocessing import Process

from multiprocessing import cpu_count, set_start_method

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import wandb_config
from models.state_mapper_supervised import LitStateMapper

try:
    set_start_method('spawn')
except RuntimeError:
    pass


if __name__ == '__main__':
    CPU_COUNT = cpu_count()
    GPU_COUNT = torch.cuda.device_count()

    wandb_config.update(
        {
            "group": "state_mapper",
            "tags": ["supervised"],
        }
    )

    config = {
        "data_file": "panda-ur5_10000_1000.pt",
        "state_mapper_config": {
            "network_width": 256,
            "network_depth": 8,
            "dropout": .1,
            "out_activation": "tanh",
            "lr": 3e-3,
        },
        "batch_size": 32,
        "max_epochs": 3_000,
    }
    # pl.utilities.rank_zero.rank_zero_only(wandb.init)(config=config, **wandb_config)
    # wandb.init(config=config, **wandb_config)

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        devices = -1
        config["num_workers"] = cpu_count() // torch.cuda.device_count()
    else:
        devices = None
        config["num_workers"] = cpu_count()

    state_mapper = LitStateMapper(
        **config
    )

    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min"),
        LearningRateMonitor(logging_interval='step'),
        # EarlyStopping(monitor="validation_loss", mode="min", patience=1000)
    ]

    logger = WandbLogger(**wandb_config, log_model="all")
    # logger = TensorBoardLogger("results")

    try:
        trainer = pl.Trainer(strategy=DDPStrategy(), accelerator="gpu", devices=devices,
                             logger=logger, max_epochs=config["max_epochs"], callbacks=callbacks)
        trainer.fit(state_mapper)
    except:
        trainer = pl.Trainer(accelerator="gpu", devices=devices,
                             logger=logger, max_epochs=config["max_epochs"], callbacks=callbacks)
        trainer.fit(state_mapper)

