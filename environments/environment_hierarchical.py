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

from ray.rllib.env import MultiAgentEnv

try:
    from environments import get_env
except:
    from . import get_env


class HierarchicalEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()

        env_config = config["env_config"]
        self.env = get_env(env_config)

        self.high_level_steps = config["high_level_steps"]

    def __del__(self):
        del self.env

    def reset(self):
        self.cur_obs = self.env.reset()
        self.current_goal = None
        self.steps_remaining_at_level = None
        self.num_high_level_steps = 0
        # current low level agent id. This must be unique for each high level
        # step since agent ids cannot be reused.
        self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
        return {
            "high_level_agent": self.cur_obs,
        }

    def step(self, action_dict):
        assert len(action_dict) == 1, action_dict
        if "high_level_agent" in action_dict:
            return self._high_level_step(action_dict["high_level_agent"])
        else:
            return self._low_level_step(list(action_dict.values())[0])

    def _high_level_step(self, action):
        self.current_goal = action
        self.steps_remaining_at_level = self.high_level_steps
        self.num_high_level_steps += 1
        self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
        obs = {self.low_level_agent_id: [self.cur_obs, self.current_goal]}
        rew = {self.low_level_agent_id: 0}
        done = {"__all__": False}
        return obs, rew, done, {}

    def _low_level_step(self, action):
        self.steps_remaining_at_level -= 1

        # Step in the actual env
        f_obs, f_rew, f_done, _ = self.env.step(action)
        new_pos = tuple(f_obs[0])
        self.cur_obs = f_obs

        # Calculate low-level agent observation and reward
        obs = {self.low_level_agent_id: [f_obs, self.current_goal]}
        rew = {self.low_level_agent_id: 0}

        # Handle env termination & transitions back to higher level
        done = {"__all__": False}
        if f_done:
            done["__all__"] = True
            rew["high_level_agent"] = f_rew
            obs["high_level_agent"] = f_obs
        elif self.steps_remaining_at_level == 0:
            done[self.low_level_agent_id] = True
            rew["high_level_agent"] = 0
            obs["high_level_agent"] = f_obs

        return obs, rew, done, {}


if __name__ == '__main__':
    robot_source = "panda"
    robot_target = "panda"
    task = "reach"

    # p.connect(p.GUI)
    p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pd.getDataPath())

    config = {
        "env_config": {
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
        "high_level_steps": 5,
    }

    env = HierarchicalEnv(config)
    state = env.reset()

    print(state)

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
