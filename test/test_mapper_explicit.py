from environments.environment_robot_task import Callbacks
from trainer.apprentice import Apprentice


def test_mapper_explicit_runnable():
    config_task = {
        "name": "reach",
        "max_steps": 25,
        "accuracy": 0.03,
    }

    config = {
        "wandb_config": {
            "project": "robot2robot",
            "entity": "robot2robot",
            "mode": "disabled",
        },
        "cache": {
            "mode": "wandb",
        },
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
        "DemonstrationsTarget": {},
        "Expert": {
            "model_cls": "PPO",
            "model": {
                "framework": "torch",
                "callbacks": Callbacks,
                "model": {
                    "vf_share_layers": False,
                },
                "disable_env_checking": True,
            },
            "train": {
                "max_epochs": 1,
                "success_threshold": 0.9,
            },
        },
        "EnvironmentSampler": {
            "num_demonstrations": 10,
            "max_trials": 10,
            "discard_unsuccessful": False,
            "env": "EnvSource",
            "policy": "Random",
        },
        "Mapper": {
            "type": "explicit",
            "weight_matrix_exponent": 1,
            "weight_kcl": 1,
        },
        "Pretrainer": {
            "model_cls": "MARWIL",
            "model": {
                "framework": "torch",
                "model": {
                    "vf_share_layers": False,
                },
                "actions_in_input_normalized": True,
                "callbacks": Callbacks,
                "lr": 3e-4,
                "train_batch_size": 1,
                "num_workers": 1,
                "num_gpus": 0,
                "off_policy_estimation_methods": {},
                "evaluation_num_workers": 1,
                "evaluation_interval": 1,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "evaluation_config": {
                    "input": "sampler",
                    "callbacks": Callbacks,
                    "off_policy_estimation_methods": {},
                    "explore": False,
                },
            },
            "train": {
                "max_epochs": 1,
                "success_threshold": 0.9,
            },
        },
        "Apprentice": {
            "model_cls": "PPO",
            "model": {
                "framework": "torch",
                "callbacks": Callbacks,
                "model": {
                    "vf_share_layers": False,
                },
                "disable_env_checking": True,
            },
            "train": {
                "max_epochs": 1,
                "success_threshold": 0.9,
            },
        },
    }
    Apprentice(config=config)
