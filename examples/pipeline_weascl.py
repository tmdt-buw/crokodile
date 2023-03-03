import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging

import numpy as np

from mapper import Mapper
from world_models.discriminator import DiscriminatorSource
from world_models.transition import TransitionModel

logging.getLogger().setLevel(logging.INFO)

data_file_A = "panda_5_10_10.pt"
data_file_B = "ur5_5_10_10.pt"

config_task = {
    "name": "reach",
    "max_steps": 25,
    "accuracy": 0.03,
}


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
        "EnvSource": {
            "env": "robot-task",
            "env_config": {
                "robot_config": {
                    "name": "panda",
                    "sim_time": 0.1,
                    "scale": 0.1,
                },
                "task_config": config_task,
                "disable_env_checking": True,
            },
        },
        "DiscriminatorSource": {
            "data": data_file_A,
            "model": {
                "network_width": 1024,
                "network_depth": 4,
                "dropout": 0.1,
                "out_activation": "sigmoid",
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

    DiscriminatorSource(config)


def state_mapper_main():
    from mapper.mapper_state import StateMapper

    config = {
        "EnvSource": {
            "env": "robot-task",
            "env_config": {
                "robot_config": {
                    "name": "panda",
                    "sim_time": 0.1,
                    "scale": 0.1,
                },
                "task_config": config_task,
                "disable_env_checking": True,
            },
        },
        "EnvTarget": {
            "env": "robot-task",
            "env_config": {
                "robot_config": {
                    "name": "ur5",
                    "sim_time": 0.1,
                    "scale": 1.0,
                },
                "task_config": config_task,
                "disable_env_checking": True,
            },
        },
        "StateMapper": {
            "data": {"data_file_X": data_file_A, "data_file_Y": data_file_B},
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
                "cycle_consistency_factor": 0.1,
                "discriminator_factor": 0.25,
            },
        },
        "DiscriminatorSource": {
            "data": data_file_A,
            "model": {
                "network_width": 1024,
                "network_depth": 4,
                "dropout": 0.1,
                "out_activation": "sigmoid",
            },
            "train": {
                "lr": 1e-3,
            },
        },
        "DiscriminatorTarget": {
            "data": data_file_B,
            "model": {
                "network_width": 1024,
                "network_depth": 4,
                "dropout": 0.1,
                "out_activation": "sigmoid",
            },
            "train": {
                "lr": 1e-3,
            },
        },
    }
    model = StateMapper(config)


def trajectory_mapper_main():
    config_task = {
        "name": "reach",
        "max_steps": 25,
        "accuracy": 0.03,
    }

    config = {
        "EnvSource": {
            "env": "robot-task",
            "env_config": {
                "robot_config": {
                    "name": "panda",
                    "sim_time": 0.1,
                    "scale": 0.1,
                },
                "task_config": config_task,
                "disable_env_checking": True,
            },
        },
        "EnvTarget": {
            "env": "robot-task",
            "env_config": {
                "robot_config": {
                    "name": "ur5",
                    "sim_time": 0.1,
                    "scale": 1.0,
                },
                "task_config": config_task,
                "disable_env_checking": True,
            },
        },
        "TransitionModelSource": {
            "model_cls": "transition_model",
            "data": data_file_A,
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
        "TransitionModelTarget": {
            "model_cls": "transition_model",
            "data": data_file_B,
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
        "StateMapper": {
            "data": {"data_file_X": data_file_A, "data_file_Y": data_file_B},
            "model": {
                "network_width": 1024,
                "network_depth": 4,
                "dropout": 0.1,
                "out_activation": "tanh",
                "weight_matrix_exponent_p": np.inf,
            },
            "train": {
                # "batch_size": 512,
                "lr": 1e-3,
                "cycle_consistency_factor": 0.1,
                "discriminator_factor": 0.25,
            },
        },
        "DiscriminatorSource": {
            "data": data_file_A,
            "model": {
                "network_width": 1024,
                "network_depth": 4,
                "dropout": 0.1,
                "out_activation": "sigmoid",
            },
            "train": {
                "lr": 1e-3,
            },
        },
        "DiscriminatorTarget": {
            "data": data_file_B,
            "model": {
                "network_width": 1024,
                "network_depth": 4,
                "dropout": 0.1,
                "out_activation": "sigmoid",
            },
            "train": {
                "lr": 1e-3,
            },
        },
        "Mapper": {
            "type": "weascl",
            "data": {"data_file_X": data_file_A, "data_file_Y": data_file_B},
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
                "action_mapper": {
                    "network_width": 256,
                    "network_depth": 8,
                    "dropout": 0.1,
                    "out_activation": "tanh",
                },
            },
            "train": {
                "max_epochs": 1,
                "batch_size": 512,
                "lr": 1e-3,
            },
        },
    }
    model = Mapper(config)


if __name__ == "__main__":
    # transition_model_main()
    # discriminator_main()
    # state_mapper_main()
    trajectory_mapper_main()
