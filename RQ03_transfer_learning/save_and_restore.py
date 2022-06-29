import datetime
import os
import sys
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np

import wandb
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

sys.path.append(str(Path(__file__).resolve().parents[1]))
from environments.environment_robot_task import Callbacks

from gym import Env, spaces


class SimpleEnv(Env):
    def __init__(self, config=None):
        super(SimpleEnv, self).__init__()

        if config is None:
            config = {}

        self.observation_space = spaces.Box(-1, 1, (6,))
        self.action_space = spaces.Box(-1, 1, (3,))

        self.state = None
        self.step_counter = 0
        self.max_episode_steps = config.get("max_episode_steps", 10)
        self.action_scale = config.get("action_scale", .1)
        self.accuracy = config.get("accuracy", .01)

    def reset(self):
        self.state = self.observation_space.sample()

        self.step_counter = 0

        return self.state

    def step(self, action: np.ndarray):
        self.step_counter += 1

        self.state[:3] += action * self.action_scale
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)

        distance = np.linalg.norm(self.state[:3] - self.state[3:])

        success = distance < self.accuracy

        done = self.step_counter >= self.max_episode_steps
        done |= success

        if done:
            if success:
                reward = 1
            else:
                reward = -1
        else:
            reward = - distance / self.max_episode_steps

        # print(reward, self.state, done, success)

        info = {"success": success}

        return self.state, reward, done, info


class SimpleEnv2(Env):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.observation_space = spaces.Box(-1, 1, (6,))
        self.action_space = spaces.Box(-1, 1, (3,))

        self.state = None
        self.step_counter = 0
        self.max_episode_steps = config.get("max_episode_steps", 10)
        self.action_scale = config.get("action_scale", .1)
        self.accuracy = config.get("accuracy", .01)

    def reset(self):
        self.state = self.observation_space.sample()

        self.step_counter = 0

        return self.state

    def step(self, action: np.ndarray):
        self.step_counter += 1

        self.state[:3] -= action * self.action_scale
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)

        distance = np.linalg.norm(self.state[:3] - self.state[3:])

        success = distance < self.accuracy

        done = self.step_counter >= self.max_episode_steps
        done |= success

        if done:
            if success:
                reward = 1
            else:
                reward = -1
        else:
            reward = - distance / self.max_episode_steps

        # print(reward, self.state, done, success)

        info = {"success": success}

        return self.state, reward, done, info


from ray.rllib.utils import check_env

check_env(SimpleEnv())

register_env("simple", lambda config: SimpleEnv(config))
register_env("simple2", lambda config: SimpleEnv2(config))

wandb_mode = "online"
# wandb_mode = "disabled"
project = "ray_ppo_simple"

config = {
    "env": "simple",
    "framework": "torch",

    "callbacks": Callbacks,

    # Config dict to be passed to our custom env's constructor.
    "env_config": {
        "max_episode_steps": 10,
        "action_scale": .1,
        "accuracy": .05,
    },

    # Parallelize environment rollouts.
    "num_workers": cpu_count(),
    # "num_workers": 1,
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

results_dir = "./results/ray_ppo_simple"
experiment_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_dir = os.path.join(results_dir, experiment_name)

os.makedirs(experiment_dir, exist_ok=True)

epochs = 10
checkpoint_epochs = 1_000

# Train for n iterations and report results (mean episode rewards).
with wandb.init(config=config, project=project, entity="bitter", mode=wandb_mode):
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

        if results['custom_metrics']["success_mean"] == 1.:
            break

    checkpoint = trainer.save(experiment_dir)
    trainer.cleanup()

    config["env"] = "simple2"
    config["env_config"] = {
        "max_steps": 5,
        "action_scale": .1,
        "accuracy": .05,
    }

    # reinitialize trainer
    trainer = PPOTrainer(config)
    trainer.restore(checkpoint)

    for epoch in range(epochs, 2 * epochs):
        results = trainer.train()
        print(f"Epoch: {epoch} | avg. reward={results['episode_reward_mean']:.3f} | "
              f"success ratio={results['custom_metrics']['success_mean']:.3f}")

        wandb.log({
            'episode_reward_mean': results['episode_reward_mean'],
        }, step=epoch)

        wandb.log({
            'episode_success_mean': results['custom_metrics']["success_mean"],
        }, step=epoch)

        if results['custom_metrics']["success_mean"] == 1.:
            break
