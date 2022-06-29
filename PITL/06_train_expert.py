import sys
from multiprocessing import cpu_count
from pathlib import Path
import tempfile

import wandb
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from environments.environment_robot_task import EnvironmentRobotTask, Callbacks
from config import wandb_config

register_env("robot_task", lambda config: EnvironmentRobotTask(config))

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

    "env": "robot_task",
    "env_config": {
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
    },

    # Parallelize environment rollouts.
    "num_workers": cpu_count(),
    # "num_workers": 1,
    "num_gpus": torch.cuda.device_count(),
    # "num_gpus": 1,
}

max_epochs = 10_000

wandb_config.update({
    "group": "panda_reach_ppo",
})

# Train for n iterations and report results (mean episode rewards).
with wandb.init(config=config, **wandb_config):
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

    with tempfile.TemporaryDirectory() as dir:
        checkpoint = agent.save(dir)

        artifact = wandb.Artifact('agent_panda_reach_ppo', type='rllib checkpoint')
        artifact.add_dir(dir)
        wandb.log_artifact(artifact)
