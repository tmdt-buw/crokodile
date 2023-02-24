from world_models.discriminator import Discriminator


def test_discriminator_runnable():
    # data_file_A = "panda_5_10_2.pt"
    data_file_B = "ur5_5_10_2.pt"
    config = {
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
    Discriminator(config)
