import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data as pd
from gym import Env

sys.path.append(str(Path(__file__).resolve().parents[0]))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from mappers import get_mapper
from experts import get_expert
from utils.utils import unwind_dict_values

try:
    from environments import get_env
except:
    from . import get_env


class EnvironmentImitationLearning(Env):
    """
    A class to combine a robot instance with a task instance
    to define a RL environment
    """

    def __init__(self, config):
        env_source_config = config["env_source_config"]
        env_target_config = config["env_target_config"]

        self.env_source = get_env(env_source_config)
        self.env_target = get_env(env_target_config)

        self.observation_space = self.env_source.state_space

        self.state_mapper, self.action_mapper = get_mapper(self.env_source, self.env_target)
        self.target_expert = get_expert(self.env_target)

        self.action_space = self.env_source.action_space
        self.state_space = self.env_source.state_space
        self.goal_space = self.env_source.goal_space

        self.reward_function = self.env_source.reward_function
        self.success_criterion = self.env_source.success_criterion

        self.expert_trajectory_state_weight = config.get("expert_trajectory_state_weight", 0.)

        self.sparse_reward = config.get("sparse_reward", False)

        self.expert_trajectory = None

        self.step_counter = 0

    def __del__(self):
        del self.env_source
        del self.env_target

    def reset(self, desired_observation=None):
        """
        Reset the environment and return new state
        """

        while True:
            state = self.env_source.reset(desired_observation)

            if self.expert_trajectory_state_weight:
                mapped_target_state = self.state_mapper(state)
                observation_target = self.env_target.reset(mapped_target_state)

                # todo: check that env_target reset was successful (and not random init)
                # todo: map states back to source
                # get expert trajectory
                expert_trajectory = [observation_target]

                done = False

                while not done:
                    expert_action = self.target_expert.predict(observation_target["state"],
                                                               observation_target["goal"])
                    if expert_action:
                        # expert_trajectory.append(expert_action)
                        observation_target, goal_target, done, _ = self.env_target.step(expert_action)
                        expert_trajectory.append(observation_target)
                    else:
                        done = True

                    done |= self.env_target.success_criterion(observation_target["goal"])

                self.expert_trajectory = expert_trajectory

                if self.env_target.success_criterion(observation_target["goal"]):
                    break
                else:
                    desired_observation = self.env_source.state_space.sample()
            else:
                break

        self.step_counter = 0

        return state

    def step(self, action):
        state, reward, done, info = self.env_source.step(action)

        if self.sparse_reward and not done:
            reward = 0.

        if self.expert_trajectory_state_weight:
            mapped_target_state = self.state_mapper(state)

            state_flattened = unwind_dict_values(mapped_target_state["state"])

            state_expert = self.expert_trajectory[min(self.step_counter, len(self.expert_trajectory) - 1)]
            expert_state_flattened = unwind_dict_values(state_expert["state"])

            reward_expert = np.exp(-10 * np.linalg.norm(state_flattened - expert_state_flattened)) * 2 - 1
            # reward_expert /= self.env_source.max_steps

            reward += self.expert_trajectory_state_weight * reward_expert

        self.step_counter += 1

        return state, reward, done, info


if __name__ == '__main__':
    robot_source = "panda"
    robot_target = "panda"
    task = "reach"

    # p.connect(p.GUI)
    p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pd.getDataPath())

    config = {
        "expert_trajectory_state_weight": 1.,
        "sparse_reward": True,

        "env_source_config": {
            "name": "robot-task",
            "task_config": {
                "name": task,
                "max_steps": 25
            },
            "robot_config": {
                "name": robot_source,
            },
            "bullet_client": p
        },
        "env_target_config": {
            "name": "robot-task",
            "task_config": {
                "name": task,
                "max_steps": 25,
                "offset": (1, 1, 0),
            },
            "robot_config": {
                "name": robot_target,
                "offset": (1, 1, 0),
            },
            "bullet_client": p
        }
    }

    env = EnvironmentImitationLearning(config)
    state = env.reset()

    # print(info["expert_trajectory"])

    done = False

    while not done:
        try:
            expert_action = info["expert_trajectory"][1]
        except:
            expert_action = env.action_space.sample()
        state, reward, done, info = env.step(expert_action)
        done |= env.success_criterion(state["goal"])

    print(env.success_criterion(state["goal"]))
