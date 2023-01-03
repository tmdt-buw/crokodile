import logging
from copy import deepcopy
from multiprocessing import cpu_count

import torch

from environments.environment_robot_task import Callbacks
from main import Apprentice

if __name__ == "__main__":

    config_model = {
        "fcnet_hiddens": [180] * 5,
        "fcnet_activation": "relu",
        "vf_share_layers": False,
    }

    config_evaluation = {
        # "evaluation_parallel_to_training": True,
        "evaluation_interval": 1000,
        "evaluation_duration": 100,
        "evaluation_duration_unit": "episodes",
        "evaluation_config": {
            "input": "sampler",
            "callbacks": Callbacks,
            # "off_policy_estimation_methods": {"simulation": {"type": "simulation"}},
            "off_policy_estimation_methods": {},
            "explore": False,
        },
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
        # Parallelize environment rollouts.
        "num_workers": cpu_count(),
        "num_gpus": torch.cuda.device_count(),
        # "evaluation_num_workers": 1,
        # **config_evaluation
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

    # config_robot_target = deepcopy(config_robot_source)

    if config_robot_target == config_robot_source:
        logging.warning(
            "Same source and target robot. If you are not debugging, this is probably a mistake."
        )

    config_task = {
        "name": "reach",
        "max_steps": 25,
        "accuracy": 0.03,
    }

    config = {
        "wandb_config": {
            "project": "robot2robot",
            "entity": "robot2robot",
        },
        "cache": {
            "mode": "wandb",
            # "load": False,
            "load": {
                "Expert": "a2036c",
                # "DemonstrationsSource": "535493", # "23c77d", # 1_000_000 trajectories
                # "DemonstrationsTarget": "f2108b", # "bc9ed5",
            },
            "save": True,
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
                "max_epochs": 10_000,
                "success_threshold": 0.9,
            },
        },
        "DemonstrationsSource": {
            "num_demonstrations": 1_000_000,
            "max_trials": 2_000_000,
        },
        "Mapper": {
            "type": "explicit",
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
                "num_workers": cpu_count() - 1,
                "num_gpus": 1,
                "evaluation_num_workers": 1,
                **config_evaluation,
            },
            "train": {
                "max_epochs": 100_000,
                "success_threshold": 0.9,
            },
        },
        "Apprentice": {
            "model_cls": "PPO",
            "model": deepcopy(config_algorithm),
            "train": {
                "max_epochs": 5_000,
                "success_threshold": 0.9,
            },
        },
    }

    # Pretrainer(config=config)
    # DemonstrationsSource(config=config)
    Apprentice(config=config)
    # Apprentice(config=config)
