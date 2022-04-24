"""
Verify differentiability of dht module by performing inverse kinematics.
"""

import os
import sys
from pathlib import Path

import time
import numpy as np
import torch
import wandb
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

# wandb_mode = "online"
wandb_mode = "disabled"

from gradient_based_inverse_kinematics import create_network
from karolos.environments.environments_robot_task.robots import get_robot
import pybullet as p

p.connect(p.GUI)

if __name__ == '__main__':
    # validate_loss_fn()

    run_path = "bitter/robot2robot_state_mapper_v2/runs/wv1seduy"

    wandb.login()

    file_config = wandb.restore("config.yaml", run_path, replace=True)

    with open(file_config.name, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)

    file_checkpoint = wandb.restore("model.pt", run_path, replace=True)

    checkpoint = torch.load(file_checkpoint.name, map_location="cpu")

    data_file_A, data_file_B = config["data_files"]["value"]

    data_A = torch.load(data_file_A)
    data_B = torch.load(data_file_B)

    network = create_network(
        len(data_A["joint_limits"]),
        len(data_B["joint_limits"]),
        config["network_width"]["value"],
        config["network_depth"]["value"],
        config["dropout"]["value"],
        data_B["dht_params"],
        data_B["joint_limits"],
    )

    network.load_state_dict(checkpoint["model_state_dict"], strict=False)
    state_mapper = network[0]
    rescaler = network[1][0]

    # print(checkpoint)

    robot_A = get_robot({
        "name": os.path.basename(data_file_A).split("_")[0],
        "scale": .1,
        "sim_time": .1
    }, p)

    robot_B = get_robot({
        "name": os.path.basename(data_file_B).split("_")[0],
        "scale": .1,
        "sim_time": .1,
        "offset": (1, 1, 0)
    }, p)

    state_B = robot_B.state_space.sample()

    while True:
        state_A = robot_A.reset()
        state_joints_arm_A = state_A["arm"]["joint_positions"]
        state_joints_arm_A = torch.tensor(state_joints_arm_A, dtype=torch.float32).unsqueeze(0)


        state_joints_arm_B = state_mapper(state_joints_arm_A)[0]
        state_joints_arm_B = state_joints_arm_B.detach().numpy()
        state_B["arm"]["joint_positions"] = state_joints_arm_B

        state_B_ = robot_B.reset(state_B)

        time.sleep(1.)