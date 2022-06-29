import datetime
import os
import sys
from multiprocessing import cpu_count
from pathlib import Path

import wandb
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

sys.path.append(str(Path(__file__).resolve().parents[1]))
from environments.environment_robot_task import EnvironmentRobotTask, Callbacks
from environments.environment_imitation_learning import EnvironmentImitationLearning

register_env("robot_task", lambda config: EnvironmentRobotTask(config))
register_env("imitation", lambda config: EnvironmentImitationLearning(config))

wandb_mode = "online"
# wandb_mode = "disabled"
project = "ray_ppo_panda_reach"

config = {
    "env": "imitation",
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

    # Config dict to be passed to our custom env's constructor.
    "env_config": {
        "name": "imitation",

        "expert_trajectory_state_weight": 1.,
        "sparse_reward": True,

        "env_source_config": {
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
        },
        "env_target_config": {
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
    },

    # Parallelize environment rollouts.
    "num_workers": cpu_count(),
    # "num_gpus": torch.cuda.device_count(),
    # "num_gpus": 1,

    # "evaluation_num_workers": 2,
    # Enable evaluation, once per training iteration.
    # "evaluation_interval": 50,
    # Run 10 episodes each time evaluation runs.
    # "evaluation_duration": 10,
}

# Create an RLlib Trainer instance to learn how to act in the above
# environment.
trainer = PPOTrainer(config)

# init_dir = "/mnt/first/karolos/results/ppo_panda_reach_ray/20220514-124711/checkpoint_000945/checkpoint-945"
# trainer.load_checkpoint(init_dir)

results_dir = "./results/ray_ppo_panda_reach"
experiment_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_dir = os.path.join(results_dir, experiment_name)

os.makedirs(experiment_dir, exist_ok=True)

epochs = 1_000_000
checkpoint_epochs = 1_000

# Train for n iterations and report results (mean episode rewards).
with wandb.init(config=config, project=project, group="imitation_blank", entity="bitter", mode=wandb_mode):
    for epoch in range(epochs):
        results = trainer.train()
        print(f"Epoch: {epoch} | avg. reward={results['episode_reward_mean']:.3f} | "
              f"success ratio={results['custom_metrics']['success_mean']:.3f}")

        wandb.log({
            'episode_reward_mean': results['episode_reward_mean'],
        }, step=epoch)

        wandb.log({
            'episode_success_mean': results['custom_metrics']["success_mean"],
        }, step=epoch)

        if epochs % checkpoint_epochs == 0:
            trainer.save(experiment_dir)

        if results['custom_metrics']["success_mean"] == 1.:
            break

    trainer.save(experiment_dir)
