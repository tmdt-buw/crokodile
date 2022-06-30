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
from models.domain_mapper import LitDomainMapper

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def run_training(config, wandb_config={}):
    if "warm_start" in config:

        with tempfile.TemporaryDirectory() as dir:
            api = wandb.Api()
            artifact = api.artifact(config["warm_start"])
            artifact_dir = artifact.download(dir)

            domain_mapper = LitDomainMapper.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))
    else:
        domain_mapper = LitDomainMapper(
            **config
        )

    callbacks = [
        ModelCheckpoint(monitor="validation_loss", mode="min"),
        LearningRateMonitor(logging_interval='epoch'),
        # EarlyStopping(monitor="validation_loss", mode="min", patience=1000)
    ]

    wandb_logger = WandbLogger(
        **wandb_config,
        log_model=True
    )

    # try:
    trainer = pl.Trainer(max_epochs=config["max_epochs"], max_time="00:07:55:00",
                         strategy=DDPStrategy(), accelerator="gpu", devices=devices,
                         logger=wandb_logger, callbacks=callbacks)
    trainer.fit(domain_mapper)
    # except:
    #     trainer = pl.Trainer(max_epochs=config["max_epochs"], max_time="00:07:55:00",
    #                          accelerator="gpu", devices=devices,
    #                          logger=wandb_logger, callbacks=callbacks)
    #     trainer.fit(domain_mapper)


def launch_agent(sweep_id, devices_ids, count):
    global devices
    devices = devices_ids
    wandb.agent(sweep_id, function=run_training, count=count)


if __name__ == '__main__':
    CPU_COUNT = cpu_count()
    GPU_COUNT = torch.cuda.device_count()

    data_file_A = "panda_5_10000_1000.pt"
    data_file_B = "ur5_5_10000_1000.pt"

    wandb_config.update(
        {
            "group": "domain_mapper",
            "tags": ["behavior encoder"],
            # "mode": "disabled"
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
            "transition_model_config": {
                "network_width": 256,
                "network_depth": 8,
                "dropout": .1,
                "out_activation": "tanh",
                "lr": 3e-3,
            },
            "state_mapper_config": {
                "network_width": 256,
                "network_depth": 8,
                "dropout": .1,
                "out_activation": "tanh",
                "lr": 3e-3,
            },
            "action_mapper_config": {
                "behavior_dim": 64,
                "encoder": {
                    "lr": 3e-3,
                    "d_model": 16,
                    "nhead": 4,
                    "num_layers": 2,
                    "num_decoder_layers": 2,
                    "dim_feedforward": 64,
                },
                "decoder": {
                    "network_width": 256,
                    "network_depth": 8,
                    "dropout": .1,
                    "out_activation": "tanh",
                },
            },

            "batch_size": 32,
            "max_epochs": 10_000,
            "num_workers": cpu_count(),
        }

        torch.cuda.empty_cache()

        devices = -1

        run_training(config, wandb_config)
