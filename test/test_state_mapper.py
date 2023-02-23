import numpy as np

from mapper.mapper_state import StateMapper


def test_state_mapper_runnable():
    data_file_A = "panda_5_10_2.pt"
    data_file_B = "ur5_5_10_2.pt"
    config = {
        "EnvSource": {
            "env": "robot_task",
            "env_config": {
                "robot_config": {
                    "name": "panda",
                    "sim_time": 0.1,
                    "scale": 0.1,
                },
                "task_config": {
                    "name": "reach",
                    "max_steps": 25,
                    "accuracy": 0.03,
                },
                "disable_env_checking": True,
            },
        },
        "EnvTarget": {
            "env": "robot_task",
            "env_config": {
                "robot_config": {
                    "name": "ur5",
                    "sim_time": 0.1,
                    "scale": 0.1,
                },
                "task_config": {
                    "name": "reach",
                    "max_steps": 25,
                    "accuracy": 0.03,
                },
                "disable_env_checking": True,
            },
        },
        "StateMapper": {
            "data": {"data_file_X": data_file_A, "data_file_Y": data_file_B},
            "model": {
                "network_width": 16,
                "network_depth": 2,
                "dropout": 0.1,
                "out_activation": "tanh",
                "weight_matrix_exponent_p": np.inf,
            },
            "train": {
                "max_epochs": 1,
                "batch_size": 1,
                "lr": 1e-3,
            },
        },
        "Discriminator": {
            "data": data_file_B,
            "model": {
                "objective": "soft-boundary",
                "eps": 0.01,
                "init_center_samples": 5,
                "nu": 1e-3,
                "out_dim": 4,
                "network_width": 16,
                "network_depth": 2,
                "dropout": 0.2,
                "warmup_epochs": 1,
            },
            "train": {
                "max_epochs": 1,
                "batch_size": 1,
                "lr": 1e-3,
                "scheduler_epoch": 1,
                "lr_decrease": 0.1,
            },
        },
    }
    StateMapper(config)
