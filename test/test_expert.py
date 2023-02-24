from environments.environment_robot_task import Callbacks
from trainer.expert import Expert

# def test_expert_runnable():
#    config = {
#        "EnvSource": {
#            "env": "robot_task",
#            "env_config": {
#                "robot_config": {
#                    "name": "panda",
#                    "sim_time": 0.1,
#                    "scale": 0.1,
#                },
#                "task_config": {
#                    "name": "reach",
#                    "max_steps": 25,
#                    "accuracy": 0.03,
#                },
#                "disable_env_checking": True,
#            },
#        },
#        "Expert": {
#            "model_cls": "PPO",
#            "model": {
#                "framework": "torch",
#                "callbacks": Callbacks,
#                "model": {
#                    "vf_share_layers": False,
#                },
#                "disable_env_checking": True,
#            },
#            "train": {
#                "max_epochs": 1,
#                "success_threshold": 1.0,
#            },
#        },
#    }
#
#    Expert(config)
#
