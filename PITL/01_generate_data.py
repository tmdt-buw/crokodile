import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from environments.environments_robot_task.robots import get_robot

from config import *

def generate_samples(robot, n):
    states = []
    actions = []
    next_states = []

    for _ in tqdm(range(n)):
        state = robot.reset()
        action = robot.action_space.sample()
        next_state = robot.step(action)

        state_joints_arm = state["arm"]["joint_positions"]
        state_joints_arm = torch.FloatTensor(state_joints_arm)

        action_arm = action["arm"]
        action_arm = torch.FloatTensor(action_arm)

        next_state_joints_arm = next_state["arm"]["joint_positions"]
        next_state_joints_arm = torch.FloatTensor(next_state_joints_arm)

        states.append(state_joints_arm)
        actions.append(action_arm)
        next_states.append(next_state_joints_arm)

    states = torch.stack(states)
    actions = torch.stack(actions)
    next_states = torch.stack(next_states)

    return states, actions, next_states


if __name__ == '__main__':
    os.makedirs(data_folder, exist_ok=True)

    for robot_name in ["panda", "ur5"]:
        print(f"Generate data: {robot_name}")

        robot_config = {
            "sim_time": .1,
            "scale": .1
        }

        n_samples_train = 10_000
        n_samples_test = 1_000

        import pybullet as p

        # p.connect(p.GUI)
        p.connect(p.DIRECT)

        robot = get_robot({
            "name": robot_name,
            **robot_config
        }, bullet_client=p)

        states_train, actions_train, next_states_train = generate_samples(robot, n_samples_train)
        states_test, actions_test, next_states_test = generate_samples(robot, n_samples_test)

        joint_limits = torch.tensor([joint.limits for joint in robot.joints])

        torch.save(
            {
                "states_train": states_train,
                "actions_train": actions_train,
                "next_states_train": next_states_train,
                "states_test": states_test,
                "actions_test": actions_test,
                "next_states_test": next_states_test,
                "dht_params": robot.dht_params,
                "joint_limits": joint_limits,
                "robot_config": robot_config,
                "state_space": robot.state_space,
                "action_space": robot.action_space
            },
            os.path.join(data_folder, f"{robot_name}_{n_samples_train}_{n_samples_test}.pt")
        )
