"""
Verify differentiability of dht module by performing inverse kinematics.
"""

import os
import sys
from pathlib import Path

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

# wandb_mode = "online"
wandb_mode = "disabled"

from gradient_based_inverse_kinematics import create_network
from environments.environments_robot_task.robots import get_robot
import pybullet as p

p.connect(p.DIRECT)

if __name__ == '__main__':
    robot_A = get_robot({
        "name": "panda",
        "scale": .1,
        "sim_time": .1
    }, p)

    robot_B = get_robot({
        "name": "ur5",
        "scale": .1,
        "sim_time": .1,
        # "offset": (1, 1, 0)
    }, p)

    view_matrix_1 = p.computeViewMatrix(
        [2,0,0],
        [0 ,0,0],
        [0,0,1]
    )

    view_matrix_2 = p.computeViewMatrix(
        [0,2,0],
        [0 ,0,0],
        [0,0,1]
    )

    view_matrix_3 = p.computeViewMatrix(
        [0,0,2],
        [0 ,0,0],
        [-1,0,0]
    )

    RENDER_WIDTH = 300
    RENDER_HEIGHT = 300

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
        nearVal=0.1, farVal=3.0)

    def get_image(view_matrix):
        (_, _, px, _, _) = p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]

        return rgb_array


    while True:
        robot_A.reset()
        robot_B.reset()

        fig, axes = plt.subplots(1,3)

        axes[0].imshow(get_image(view_matrix_1))
        axes[1].imshow(get_image(view_matrix_2))
        axes[2].imshow(get_image(view_matrix_3))

        plt.tight_layout()
        plt.show()

        time.sleep(3)