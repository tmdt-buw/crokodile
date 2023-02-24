from demonstrations import EnvironmentSampler
from environments.environment_robot_task import Callbacks


def test_environment_sampler_runnable():
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
        "EnvironmentSampler": {
            "num_demonstrations": 10,
            "max_trials": 10,
            "discard_unsuccessful": False,
            "env": "EnvSource",
            "policy": "Random",
        },
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
                "success_threshold": 1.0,
            },
        },
    }

    EnvironmentSampler(config)
