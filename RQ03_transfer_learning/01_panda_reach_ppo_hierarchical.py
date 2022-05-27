import datetime
import os
import sys
from multiprocessing import cpu_count
from pathlib import Path
from copy import deepcopy

import wandb
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.rllib.env import MultiAgentEnv

sys.path.append(str(Path(__file__).resolve().parents[1]))
from environments.environment_robot_task import EnvironmentRobotTask, Callbacks
from environments.environment_imitation_learning import EnvironmentImitationLearning

register_env("robot_task", lambda config: EnvironmentRobotTask(config))
register_env("imitation", lambda config: EnvironmentImitationLearning(config))
register_env("hierarchical", lambda config: EnvironmentImitationLearning(config))

wandb_mode = "online"
# wandb_mode = "disabled"
project = "ray_ppo_panda_reach"

config_env_source = {
    "name": "robot-task",
    "robot_config": {
        "name": "panda",
        "sim_time": .1,
        "scale": .1,
    },
    "task_config": {
        "name": "reach",
        "max_steps": 25,
        "accuracy": .03,
    },
}

config = {
    "framework": "torch",

    "callbacks": Callbacks,

    "model": {
        "fcnet_hiddens": [180] * 5,
        "fcnet_activation": "relu",
    },

    "num_sgd_iter": 3,
    "lr": 3e-4,

    "train_batch_size": 2496,
    "sgd_minibatch_size": 256,

    # Parallelize environment rollouts.
    "num_workers": cpu_count(),
    # "num_workers": 1,
    # "num_gpus": torch.cuda.device_count(),
    "num_gpus": 1,
}

max_epochs = 1_000_000

# Train for n iterations and report results (mean episode rewards).
with wandb.init(config=config, project=project, group="panda_reach_ppo", entity="robot2robot", mode=wandb_mode):
    config["env"] = "robot_task"
    config["env_config"] = deepcopy(config_env_source)

    agent = PPOTrainer(config)

    for epoch in range(max_epochs):
        results = agent.train()
        print(f"Epoch: {epoch} | avg. reward={results['episode_reward_mean']:.3f} | "
              f"success ratio={results['custom_metrics']['success_mean']:.3f}")

        wandb.log({
            'episode_reward_mean': results['episode_reward_mean'],
        }, step=epoch)

        wandb.log({
            'episode_success_mean': results['custom_metrics']["success_mean"],
        }, step=epoch)

        if results['custom_metrics']["success_mean"] > .97:
            break

    checkpoint = agent.save(wandb.run.dir)