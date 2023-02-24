from world_models.transition import TransitionModel


def test_transition_model_runnable():
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
        "TransitionModel": {
            "model_cls": "transition_model",
            "data": data_file_B,
            # "log_suffix": "_B",
            "model": {
                "network_width": 16,
                "network_depth": 2,
                "dropout": 0.0,
                "out_activation": "tanh",
            },
            "train": {
                "max_epochs": 1,
                "batch_size": 1,
                "lr": 1e-3,
            },
        },
    }
    TransitionModel(config)
