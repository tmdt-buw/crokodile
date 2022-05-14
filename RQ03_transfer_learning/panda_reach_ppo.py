import sys
from pathlib import Path
import datetime
import torch.cuda
from ray.rllib.agents.ppo import PPOTrainer

import os


sys.path.append(str(Path(__file__).parents[1].resolve()))

from karolos.environments.environment_robot_task_gym import EnvironmentRobotTaskGym, Callbacks
from ray.tune.registry import register_env
from multiprocessing import cpu_count
import wandb

register_env("robot_task", lambda config: EnvironmentRobotTaskGym(config))

wandb_mode = "online"
# wandb_mode = "disabled"
project = "ray_ppo_panda_reach"

config = {
    "env": "robot_task",
    "framework": "torch",

    # "Q_model": {
    #     "fcnet_hiddens": [128, 64, 32, 16, 8, 4],
    #     "fcnet_activation": "relu",
    # },
    # # Model options for the policy function (see `Q_model` above for details).
    # # The difference to `Q_model` above is that no action concat'ing is
    # # performed before the post_fcnet stack.
    # "policy_model": {
    #     "fcnet_hiddens": [128, 128, 64, 32, 16],
    #     "fcnet_activation": "relu",
    # },
    # "prioritized_replay": True,

    "callbacks": Callbacks,

    # "model": {
    #     "fcnet_hiddens": [64] * 5,
    #     "fcnet_activation": "relu",
    # },

    "num_sgd_iter": 4,

    # "kl_coeff": 0.0,
    "train_batch_size": 2496,
    "sgd_minibatch_size": 64,

    # Config dict to be passed to our custom env's constructor.
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
    # "num_gpus": torch.cuda.device_count(),

    # "evaluation_num_workers": 2,
    # Enable evaluation, once per training iteration.
    # "evaluation_interval": 50,
    # Run 10 episodes each time evaluation runs.
    # "evaluation_duration": 10,
}

# Create an RLlib Trainer instance to learn how to act in the above
# environment.
# trainer = SACTrainer(config)
trainer = PPOTrainer(config)

init_dir = "/mnt/first/karolos/results/ppo_panda_reach_ray/20220514-124711/checkpoint_000945/checkpoint-945"

trainer.load_checkpoint(init_dir)

results_dir = "./results/ppo_panda_reach_ray"
experiment_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_dir = os.path.join(results_dir, experiment_name)

os.makedirs(experiment_dir, exist_ok=True)

epochs = 1_000_000
checkpoint_epochs = 1_000

# Train for n iterations and report results (mean episode rewards).
with wandb.init(config=config, project=project, entity="bitter", mode=wandb_mode):
    for epoch in range(epochs):
        results = trainer.train()
        print(f"Epoch: {epoch}; avg. reward={results['episode_reward_mean']}")

        wandb.log({
            'episode_reward_mean': results['episode_reward_mean'],
        }, step=epoch)

        wandb.log({
            'episode_success_mean': results['custom_metrics']["success_mean"],
        }, step=epoch)

        if epochs % checkpoint_epochs == 0:
            trainer.save(experiment_dir)

    trainer.save(experiment_dir)