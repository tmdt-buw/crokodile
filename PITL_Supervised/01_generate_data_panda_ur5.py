import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from environments.environments_robot_task.robots import get_robot

from config import *

import pybullet as p


def generate_samples(robotA, robotB, n):
    states_A = []
    actions_A = []
    next_states_A = []
    states_B = []

    pbar = tqdm(total=n, desc="Progress")

    while len(states_A) < n:
        state_A = robotA.reset()
        tcp_pose_A = robotA.get_tcp_pose()

        joint_positions_B = robotB.calculate_inverse_kinematics(tcp_pose_A[0], tcp_pose_A[1][[3, 0, 1, 2]])

        if joint_positions_B is None:
            continue

        state_joints_arm_B = robotB.normalize_joints(joint_positions_B)
        state_joints_arm_B = torch.FloatTensor(state_joints_arm_B)

        state_joints_arm_A = state_A["arm"]["joint_positions"]
        state_joints_arm_A = torch.FloatTensor(state_joints_arm_A)

        action_A = robotA.action_space.sample()
        next_state_A = robotA.step(action_A)

        action_arm_A = action_A["arm"]
        action_arm_A = torch.FloatTensor(action_arm_A)

        next_state_joints_arm_A = next_state_A["arm"]["joint_positions"]
        next_state_joints_arm_A = torch.FloatTensor(next_state_joints_arm_A)

        states_A.append(state_joints_arm_A)
        actions_A.append(action_arm_A)
        next_states_A.append(next_state_joints_arm_A)

        states_B.append(state_joints_arm_B)

        pbar.update(1)

    states_A = torch.stack(states_A)
    actions_A = torch.stack(actions_A)
    next_states_A = torch.stack(next_states_A)
    states_B = torch.stack(states_B)

    return states_A, actions_A, next_states_A, states_B


if __name__ == '__main__':
    os.makedirs(data_folder, exist_ok=True)

    p.connect(p.DIRECT)

    n_samples_train = 10_000
    n_samples_test = 1000

    robot_A_name, robot_A_config = ("panda", {"name": "panda", "sim_time": .1, "scale": .1})
    robot_B_name, robot_B_config = ("ur5", {"name": "ur5", "sim_time": .1, "scale": .1})

    print(f"Generate data: {robot_A_name} {robot_B_name}")

    robot_A = get_robot({
        "name": robot_A_name,
        **robot_A_config
    }, bullet_client=p)

    robot_B = get_robot({
        "name": robot_B_name,
        **robot_B_config
    }, bullet_client=p)

    for link_A in range(p.getNumJoints(robot_A.model_id)):
        p.setCollisionFilterGroupMask(robot_A.model_id, link_A, 0, 0)

    for link_B in range(p.getNumJoints(robot_B.model_id)):
        p.setCollisionFilterGroupMask(robot_B.model_id, link_B, 0, 0)

    joint_limits_A = torch.tensor([joint.limits for joint in robot_A.joints])
    joint_limits_B = torch.tensor([joint.limits for joint in robot_B.joints])

    states_train_A, actions_train_A, next_states_train_A, states_train_B = generate_samples(robot_A, robot_B, n_samples_train)
    states_test_A, actions_test_A, next_states_test_A, states_test_B = generate_samples(robot_A, robot_B, n_samples_test)

    torch.save(
        {
            "A": {
                "states_train": states_train_A,
                "states_test": states_test_A,
                "actions_train": actions_train_A,
                "actions_test": actions_test_A,
                "next_states_train": next_states_train_A,
                "next_states_test": next_states_test_A,
                "dht_params": robot_A.dht_params,
                "joint_limits": joint_limits_A,
                "robot_config": robot_A_config,
                "state_space": robot_A.state_space,
                "action_space": robot_A.action_space
            },
            "B": {
                "states_train": states_train_B,
                "states_test": states_test_B,
                "dht_params": robot_B.dht_params,
                "joint_limits": joint_limits_B,
                "robot_config": robot_B_config,
                "state_space": robot_B.state_space,
                "action_space": robot_B.action_space
            },
        },
        os.path.join(data_folder, f"{robot_A_name}-{robot_B_name}_{n_samples_train}_{n_samples_test}.pt")
    )
