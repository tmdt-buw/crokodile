from lit_models.lit_trainer import TransitionModel, Discriminator, StateMapper, TrajectoryMapper
import numpy as np

data_file_A = "panda_5_20000_4000.pt"
data_file_B = "ur5_5_20000_4000.pt"


def transition_model_main():
    config_A = {
        "wandb_config": {
            "project": "robot2robot",
            "entity": "robot2robot",
        },
        "cache": {
            "mode": "wandb",
            "load": True,
            "save": True,
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
            "callbacks": {},
        },
    }

    model = TransitionModel(config_A)


def discriminator_main():
    config_A = {
        "wandb_config": {
            "project": "robot2robot",
            "entity": "robot2robot",
        },
        "cache": {
            "mode": "wandb",
            "load": False,
            "save": True,
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
            "callbacks": {},
        },
    }

    Discriminator(config_A)


def state_mapper_main():
    config_AB = {
        "wandb_config": {
            "project": "robot2robot",
            "entity": "robot2robot",
        },
        "cache": {
            "mode": "wandb",
            "load": {"Discriminator": "5444fd"},
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
            "callbacks": {},
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
            "callbacks": {},
        },
    }
    model = StateMapper(config_AB)


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
            "callbacks": {},
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
            "callbacks": {},
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
            "callbacks": {},
        },
        "TrajectoryMapper": {
            "model_cls": "trajectory_mapper",
            "data": {"data_file_X": data_file_A, "data_file_Y": data_file_B},
            "log_suffix": "_AB",
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
            "callbacks": {},
        },
    }
    model = TrajectoryMapper(config_AB)


if __name__ == "__main__":
    #transition_model_main()
    #discriminator_main()
    #state_mapper_main()
    trajectory_mapper_main()
