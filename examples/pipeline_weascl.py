import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from world_models.discriminator import Discriminator
from world_models.transition import TransitionModel

data_file_A = "panda_5_20000_4000.pt"
data_file_B = "ur5_5_20000_4000.pt"


def transition_model_main():
    config = {
        "TransitionModel": {
            "model_cls": "transition_model",
            "data": data_file_B,
            # "log_suffix": "_B",
            "model": {
                "network_width": 256,
                "network_depth": 4,
                "dropout": 0.0,
                "out_activation": "tanh",
            },
            "train": {
                "max_epochs": 1,
                "batch_size": 2048,
                "lr": 1e-3,
            },
        },
    }

    TransitionModel(config)


def discriminator_main():
    config = {
        "Discriminator": {
            "data": data_file_B,
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
                "max_epochs": 1,
                "batch_size": 512,
                "lr": 1e-3,
                "scheduler_epoch": 150,
                "lr_decrease": 0.1,
            },
        },
    }

    Discriminator(config)


def state_mapper_main():
    from mapper.mapper_state import StateMapper

    config = {
        "StateMapper": {
            "model_cls": "state_mapper",
            "data": {"data_file_X": data_file_A, "data_file_Y": data_file_B},
            "log_suffix": "_AB",
            "model": {
                "network_width": 1024,
                "network_depth": 4,
                "dropout": 0.1,
                "out_activation": "tanh",
                "weight_matrix_exponent_p": np.inf,
            },
            "train": {
                "max_epochs": 1,
                "batch_size": 512,
                "lr": 1e-3,
            },
        },
        "Discriminator": {
            "data": data_file_B,
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
                "max_epochs": 1,
                "batch_size": 512,
                "lr": 1e-3,
                "scheduler_epoch": 150,
                "lr_decrease": 0.1,
            },
        },
    }
    model = StateMapper(config)


def trajectory_mapper_main():
    config_AB = {
        "wandb_config": {
            "project": "robot2robot",
            "entity": "robot2robot",
        },
        "cache": {
            "mode": "wandb",
            "load": {"StateMapper": "913e59", "TransitionModel": "f8c6ad"},
            "save": True,
        },
        "StateMapper": {
            "model_cls": "state_mapper",
            "data": {"data_file_X": data_file_A, "data_file_Y": data_file_B},
            "log_suffix": "_AB",
            "model": {
                "network_width": 1024,
                "network_depth": 4,
                "dropout": 0.1,
                "out_activation": "tanh",
                "weight_matrix_exponent_p": np.inf,
            },
            "train": {
                "max_epochs": 50,
                "batch_size": 512,
                "lr": 1e-3,
            },
        },
        "Discriminator": {
            "model_cls": "discriminator",
            "data": data_file_B,
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
                "max_epochs": 50,
                "batch_size": 512,
                "lr": 1e-3,
                "scheduler_epoch": 150,
                "lr_decrease": 0.1,
            },
        },
        "TransitionModel": {
            "model_cls": "transition_model",
            "data": data_file_B,
            "log_suffix": "_B",
            "model": {
                "network_width": 256,
                "network_depth": 4,
                "dropout": 0.0,
                "out_activation": "tanh",
            },
            "train": {
                "max_epochs": 50,
                "batch_size": 2048,
                "lr": 1e-3,
            },
        },
        "TrajectoryMapper": {
            "model_cls": "trajectory_mapper",
            "data": {"data_file_X": data_file_A, "data_file_Y": data_file_B},
            # "log_suffix": "_AB",
            "model": {
                "weight_matrix_exponent_p": np.inf,
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
                    "dropout": 0.1,
                    "out_activation": "tanh",
                },
            },
            "train": {
                "max_epochs": 50,
                "batch_size": 512,
                "lr": 1e-3,
            },
        },
    }
    model = TrajectoryMapper(config_AB)


if __name__ == "__main__":
    # transition_model_main()
    # discriminator_main()
    state_mapper_main()
    # trajectory_mapper_main()
