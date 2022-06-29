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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import wandb
from torch.multiprocessing import Process

from config import wandb_config

sys.path.append(str(Path(__file__).resolve().parents[1]))

from multiprocessing import cpu_count, set_start_method
from models.domain_mapper import LitDomainMapper

try:
    set_start_method('spawn')
except RuntimeError:
    pass

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")


def run_training(config, wandb_config={}):
    if "warm_start" in config:

        with tempfile.TemporaryDirectory() as dir:
            api = wandb.Api()
            artifact = api.artifact(config["warm_start"])
            artifact_dir = artifact.download(dir)

            domain_mapper = LitDomainMapper.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))
    else:
        domain_mapper = LitDomainMapper(
            **config,
            dynamics_model_network_width=config["network_width"],
            dynamics_model_network_depth=config["network_depth"],
            dynamics_model_dropout=config["dropout"],
            dynamics_model_lr=config["lr"],
            state_mapper_network_width=config["network_width"],
            state_mapper_network_depth=config["network_depth"],
            state_mapper_dropout=config["dropout"],
            state_mapper_lr=config["lr"],
            action_mapper_network_width=config["network_width"],
            action_mapper_network_depth=config["network_depth"],
            action_mapper_dropout=config["dropout"],
            action_mapper_lr=config["lr"],
        )

    # trainer = pl.Trainer(strategy="ddp", accelerator="gpu", logger=wandb_logger)
    # trainer = pl.Trainer(strategy=DDPSpawnStrategy(), accelerator="gpu", logger=wandb_logger)
    callbacks = [
        ModelCheckpoint(monitor="validation_loss", mode="min"),
        EarlyStopping(monitor="validation_loss", mode="min", patience=1000)
    ]

    wandb_logger = WandbLogger(**wandb_config, log_model="all")

    try:
        trainer = pl.Trainer(accelerator="gpu", devices=devices,
                             logger=wandb_logger, max_epochs=config["max_epochs"], callbacks=callbacks)
        trainer.fit(domain_mapper)
    except:

        trainer = pl.Trainer(strategy=DDPStrategy(), accelerator="gpu", devices=devices,
                             logger=wandb_logger, max_epochs=config["max_epochs"], callbacks=callbacks)
        trainer.fit(domain_mapper)


def launch_agent(sweep_id, devices_ids, count):
    global devices
    devices = devices_ids
    wandb.agent(sweep_id, function=run_training, count=count)


if __name__ == '__main__':
    CPU_COUNT = cpu_count()
    GPU_COUNT = torch.cuda.device_count()

    data_file_A = "panda_10000_1000.pt"
    data_file_B = "ur5_10000_1000.pt"

    wandb_config.update(
        {
            "group": "domain_mapper",
        }
    )

    sweep = False

    if sweep:

        sweep_config = {
            "method": "bayes",
            'metric': {
                'name': 'validation_loss',
                'goal': 'minimize'
            },
            "parameters": {
                "data_file_A": {
                    "value": data_file_A
                },
                "data_file_B": {
                    "value": data_file_B
                },
                "network_width": {
                    "min": 8,
                    "max": 256
                },
                "network_depth": {
                    "min": 1,
                    "max": 5
                },
                # "weight_matrix_exponent": {
                #     "value": 10
                # },
                "dropout": {
                    "min": 0.,
                    "max": .15
                },
                "lr": {
                    "min": 0.0001,
                    "max": 0.005
                },
                "batch_size": {
                    "values": [32, 64, 128, 256, 512]
                },
                "num_workers": {
                    "value": CPU_COUNT // GPU_COUNT
                },
                "max_epochs": {
                    "value": 100
                },
            }
        }

        runs_per_agent = 5

        sweep_id = wandb.sweep(sweep_config, **wandb_config)
        # sweep_id = "bitter/robot2robot_state_mapper/ola3rf5f"

        processes = []

        for device_id in range(GPU_COUNT):
            process = Process(target=launch_agent, args=(sweep_id, [device_id], runs_per_agent))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

    else:
        config = {
            "data_file_A": data_file_A,
            "data_file_B": data_file_B,
            "network_width": 64,
            "network_depth": 4,
            "weight_matrix_exponent": 1e3,
            "dropout": 0.1,
            "lr": 3e-4,
            "batch_size": 64,
            "num_workers": 10,
            "max_epochs": 10_000,
            # "warm_start": "robot2robot/PITL/model-6o48z2nx:best"
        }

        # pl.utilities.rank_zero.rank_zero_only(wandb.init)(config=config, **wandb_config)
        # wandb.init(config=config, **wandb_config)

        torch.cuda.empty_cache()
        devices = -1 if torch.cuda.is_available() else None
        # devices = list(range(1,8))
        run_training(config, wandb_config)
