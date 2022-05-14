import numpy as np
import os
import torch
from tqdm import tqdm

from robots import get_robot

def generate_samples(robot, n):
    states = []

    for _ in tqdm(range(n)):
        state = robot.reset()

        state_joints_arm = state["arm"]["joint_positions"]
        state_joints_arm = torch.FloatTensor(state_joints_arm)

        states.append(state_joints_arm)

    states = torch.stack(states)

    return states


if __name__ == '__main__':
    os.makedirs("data", exist_ok=True)

    robot_name = "ur5"
    n_samples_train = 10_000
    n_samples_test = 1_000

    import pybullet as p
    p.connect(p.GUI)

    robot = get_robot({
        "name": robot_name,
    }, bullet_client=p)

    exit()

    states_train = generate_samples(robot, n_samples_train)
    states_test = generate_samples(robot, n_samples_test)

    joint_limits = torch.tensor([joint.limits for joint in robot.joints])

    torch.save({
        "states_train": states_train,
        "states_test": states_test,
        "dht_params": robot.dht_params,
        "joint_limits": joint_limits,
    }, f"data/{robot_name}_{n_samples_train}_{n_samples_test}.pt")
