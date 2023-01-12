from copy import deepcopy

from environments.environment_robot_task import Callbacks
from trainer.apprentice import Apprentice

if __name__ == "__main__":

    config_model = {
        "fcnet_hiddens": [180] * 5,
        "fcnet_activation": "relu",
        "vf_share_layers": False,
    }

    config_algorithm = {
        "framework": "torch",
        "callbacks": Callbacks,
        "model": config_model,
        "num_sgd_iter": 3,
        "lr": 3e-4,
        "train_batch_size": 2496,
        "sgd_minibatch_size": 256,
        "disable_env_checking": True,
        "num_workers": 1,
        "num_gpus": 0,
    }

    config_robot_source = {
        "name": "panda",
        "sim_time": 0.1,
        "scale": 0.1,
    }

    config_robot_target = {
        "name": "ur5",
        "sim_time": 0.1,
        "scale": 0.2,
    }

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
                "robot_config": config_robot_source,
                "task_config": config_task,
            },
        },
        "EnvTarget": {
            "env": "robot_task",
            "env_config": {
                "robot_config": config_robot_target,
                "task_config": config_task,
            },
        },
        "Expert": {
            "model_cls": "PPO",
            "model": deepcopy(config_algorithm),
            "train": {
                "max_epochs": 1,
                "success_threshold": 0.9,
            },
        },
        "DemonstrationsSource": {
            "num_demonstrations": 100,
            "max_trials": 100,
            "discard_unsuccessful": False,
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
                "model": config_model,
                "actions_in_input_normalized": True,
                "callbacks": Callbacks,
                "lr": 3e-4,
                "train_batch_size": 256,
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
            "model": deepcopy(config_algorithm),
            "train": {
                "max_epochs": 1,
                "success_threshold": 0.9,
            },
        },
    }

    Apprentice(config=config)
