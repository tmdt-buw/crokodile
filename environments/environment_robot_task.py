import random
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import os

import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
from gym import Env, spaces
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.evaluation import Episode

sys.path.append(str(Path(__file__).resolve().parents[0]))

from environments_robot_task.robots import get_robot
from environments_robot_task.tasks import get_task


class EnvironmentRobotTask(Env):
    """
    A class to combine a robot instance with a task instance
    to define a RL environment
    """

    def __init__(self, config):
        robot_config = config["robot_config"]
        task_config = config["task_config"]
        render = config.get("render", False)
        bullet_client = config.get("bullet_client")

        self.render = render

        self.task_config = task_config
        self.robot_config = robot_config

        if bullet_client is None:
            connection_mode = p.GUI if render else p.DIRECT

            bullet_client = bc.BulletClient(connection_mode)

            bullet_client.setAdditionalSearchPath(pd.getDataPath())

            time_step = 1. / 300.
            bullet_client.setTimeStep(time_step)
            bullet_client.setRealTimeSimulation(0)

        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        self.bullet_client = bullet_client

        self.task = get_task(task_config, self.bullet_client)

        self.robot = get_robot(robot_config, self.bullet_client)

        # self.action_space = spaces.flatten_space(self.robot.action_space)
        self.action_space = self.robot.action_space

        self.state_space = spaces.Dict({
            'robot': self.robot.state_space,
            'task': self.task.state_space,
        })

        self.goal_space = self.task.goal_space["desired"]

        self.max_steps = self.task.max_steps

        self.reward_function = self.task.reward_function
        self.success_criterion = self.task.success_criterion

    def __del__(self):
        del self.robot
        del self.task
        del self.bullet_client

    @property
    def observation_space(self):
        observation_space = spaces.Dict({
            "state": self.state_space,
            "goal": self.goal_space
        })

        return observation_space

    def reset(self, desired_observation=None):
        """
        Reset the environment and return new state
        """

        if desired_observation is None:
            desired_observation = {}

        desired_state = desired_observation.get("state", {})
        desired_goal = desired_observation.get("goal", {})

        state_robot = self.robot.reset(desired_state.get("robot"))
        state_task, goal, info = self.task.reset(desired_state.get("task"), desired_goal, self.robot, state_robot)

        observation = {
            "state": {
                'robot': state_robot,
                'task': state_task
            },
            "goal": goal["desired"]
        }

        return observation

    def step(self, action):

        # action = spaces.unflatten(self.robot.action_space, action)

        state_robot = self.robot.step(action)
        state_task, goal, done, info = self.task.step(state_robot, self.robot)

        observation = {
            "state": {
                'robot': state_robot,
                'task': state_task
            },
            "goal": goal["desired"]
        }

        success = self.success_criterion(goal)
        done |= success

        reward = self.reward_function(goal, done)

        info["success"] = success

        return observation, reward, done, info

    def seed(self, seed=None):
        random.seed(seed)


class Callbacks(DefaultCallbacks):
    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[-1]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called after episode is done!"
            )

        episode.custom_metrics["success"] = episode.last_info_for()["success"]
