import datetime
import os
import sys
from multiprocessing import cpu_count
from pathlib import Path
from copy import deepcopy
import tempfile
import re
import wandb
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

sys.path.append(str(Path(__file__).resolve().parents[1]))
from environments.environment_robot_task import EnvironmentRobotTask, Callbacks
from environments.environment_policy_extension import EnvironmentPolicyExtension
from models.domain_mapper import LitDomainMapper

from config import *

register_env("policy_extension", lambda config: EnvironmentPolicyExtension(config))

# wandb_mode = "online"
wandb_mode = "disabled"

checkpoint_reference = "robot2robot/PITL/model-1w2koi3e:best"

# download checkpoint locally (if not already cached)

with tempfile.TemporaryDirectory() as dir:
    api = wandb.Api()
    artifact = api.artifact(checkpoint_reference)
    artifact_dir = artifact.download(dir)
    # load checkpoint
    domain_mapper = LitDomainMapper.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))


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

    "env": "policy_extension",
    "env_config": {
        "state_mapper": domain_mapper.state_mapper_BA,
        "action_mapper": domain_mapper.action_mapper_AB,

        "name": "robot-task",
        "robot_config": {
            "name": "ur5",
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
    "num_workers": cpu_count() // 2,
    # "num_workers": 0,
    # "num_gpus": torch.cuda.device_count(),
    "num_gpus": 1,
}

agent_source_path = "robot2robot/PITL/agent:best"  # todo: switch to agent_panda_reach once artifact is updated

agent = PPOTrainer(config)

with tempfile.TemporaryDirectory() as dir:
    api = wandb.Api()
    artifact = api.artifact(agent_source_path)
    artifact.download(dir)
    # load checkpoint
    checkpoint_folder = [f for f in os.listdir(dir) if re.match("checkpoint_\d+$", f)][0]
    checkpoint_folder = os.path.join(dir, checkpoint_folder)
    checkpoint_file = [f for f in os.listdir(checkpoint_folder) if re.match("checkpoint-\d+$", f)][0]
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
    agent.restore(checkpoint_path)

max_epochs = 5_000

# Train for n iterations and report results (mean episode rewards).
with wandb.init(config=config, **wandb_config, group="panda2ur5_reach_ppo_polexp"):
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
