import sys
from pathlib import Path

import pybullet as p
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))


from copy import deepcopy

from environments import get_env
from environments.environments_robot_task.robots import get_robot
from environments.experts import get_expert
from utils.nn import KinematicChainLoss, Sawtooth, get_weight_matrices

p.connect(p.GUI)
# p.connect(p.DIRECT)
p.setRealTimeSimulation(True)

robot = get_robot(
    {
        "name": "ur5",
        "scale": 0.1,
        "sim_time": 0.1,
        # "offset": (1, 0, 0),
    },
    p,
)

state_goal = [0, 0, 0, -1.0, -0.5, 0.5]


tcp_pose = robot.get_tcp_pose()
print(tcp_pose)

tcp_pose = torch.eye(4)
tcp_pose[:3, :3] = torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
tcp_pose[:3, -1] = torch.tensor([0.5, 0, 0.2])

angles = robot.inverse_kinematics(tcp_pose)
states = robot.angle2state(angles)
state = states[0, 0]

robot.reset({"arm": {"joint_positions": state}}, force=True)

while True:
    p.stepSimulation()

while True:
    j = int(input("joint"))
    v = float(input("value"))

    state_goal[j] = v
    robot.reset(
        {"arm": {"joint_positions": state_goal}},
    )

# [0, -1, 0, -1.0, -0.5, 0.5]
