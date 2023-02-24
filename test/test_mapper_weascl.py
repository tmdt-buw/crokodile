import numpy as np

from mapper.mapper_weascl import Mapper


def test_mapper_weascl_runnable():
    data_file_A = "panda_5_10_2.pt"
    data_file_B = "ur5_5_10_2.pt"

    config_task = {
        "name": "reach",
        "max_steps": 25,
        "accuracy": 0.03,
    }

    config = {
        "EnvSource": {
            "env": "robot_task",
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
            "env": "robot_task",
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
        "TransitionModel": {
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
                "decoder": {
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
    Mapper(config)
