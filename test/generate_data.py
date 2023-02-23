import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
import pybullet as p

from config import *
from environments.environments_robot_task.robots import get_robot

p.connect(p.DIRECT)


def generate_samples(robot, samples, trajectory_length):
    trajectories_states = []
    trajectories_actions = []

    for _ in tqdm(range(samples)):
        trajectory_states = []
        trajectory_actions = []

        state = robot.reset()
        state_ = state["arm"]["joint_positions"]
        trajectory_states.append(state_)

        action = robot.action_space.sample()

        for _ in range(trajectory_length):
            action_ = action["arm"]
            trajectory_actions.append(action_)

            state = robot.step(action)
            state_ = state["arm"]["joint_positions"]
            trajectory_states.append(state_)

        trajectory_states = torch.tensor(trajectory_states)
        trajectory_actions = torch.tensor(trajectory_actions)

        trajectories_states.append(trajectory_states)
        trajectories_actions.append(trajectory_actions)

    trajectories_states = torch.stack(trajectories_states).float()
    trajectories_actions = torch.stack(trajectories_actions).float()

    return trajectories_states, trajectories_actions


os.makedirs(data_folder, exist_ok=True)
trajectory_length = 5
samples_train = 10
samples_test = 2

for robot_name, robot_config in [
    ("panda", {"name": "panda", "sim_time": 0.1, "scale": 0.1}),
    ("ur5", {"name": "ur5", "sim_time": 0.1, "scale": 0.1}),
]:
    print(f"Generate data: {robot_name}")
    robot = get_robot({"name": robot_name, **robot_config}, bullet_client=p)
    if hasattr(robot, "joint_limits"):
        joint_limits = robot.joint_limits
    else:
        joint_limits = torch.tensor([joint.limits for joint in robot.joints])
    trajectories_states_train, trajectories_actions_train = generate_samples(
        robot, samples_train, trajectory_length
    )
    trajectories_states_test, trajectories_actions_test = generate_samples(
        robot, samples_test, trajectory_length
    )
    torch.save(
        {
            "trajectories_states_train": trajectories_states_train,
            "trajectories_actions_train": trajectories_actions_train,
            "trajectories_states_test": trajectories_states_test,
            "trajectories_actions_test": trajectories_actions_test,
            "dht_params": robot.dht_params,
            "joint_limits": joint_limits,
            "robot_config": robot_config,
            "state_space": robot.state_space,
            "action_space": robot.action_space,
        },
        os.path.join(
            data_folder,
            f"{robot_name}_{trajectory_length}_{samples_train}_{samples_test}.pt",
        ),
    )
