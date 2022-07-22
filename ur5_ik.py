import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import pybullet as p
from models.dht import get_dht_model
from environments.environments_robot_task.robots.robot_ur5 import RobotUR5

sys.path.append(str(Path(__file__).resolve().parents[1]))
from environments.environments_robot_task.robots import get_robot


def angles_equal(angle1, angle2):
    angle_diff = (angle1 - angle2).abs() % (2 * torch.pi)
    return torch.isclose(angle_diff, torch.tensor(0.), atol=1e-3).any() or \
           torch.isclose(angle_diff, torch.tensor(2 * torch.pi), atol=1e-3).any()


def plot_pose(pose, length=.1):
    P = pose[:3, -1]
    orientation = pose[:3, :3]

    lines = []

    for lineid in range(3):
        color = [0] * 3
        color[lineid] = 1
        line = p.addUserDebugLine(P, P + length * orientation[:, lineid], color)
        lines.append(line)

    return lines



joint_limits = torch.tensor([
    (-6.2831, 6.2831), (-2.3561, 2.3561), (-3.1415, 3.1415), (-2.3561, 2.3561), (-6.2831, 6.2831), (-6.2831, 6.2831)
])

dht_params = [
    {"d": .089159, "a": 0., "alpha": np.pi / 2.},
    {"d": 0., "a": -.425, "alpha": 0.},
    {"d": 0., "a": -.39225, "alpha": 0.},
    {"d": .10915, "a": 0., "alpha": np.pi / 2.},
    {"d": .09465, "a": 0., "alpha": -np.pi / 2.},
    {"d": .0823, "a": 0., "alpha": 0},
    {"theta": np.pi / 2, "d": .15, "a": 0., "alpha": np.pi},
]

dht_model = get_dht_model(dht_params, joint_limits)



def calculate_ur5_ik(T_0E):
    # T_6E is constant and known, so T_06 = T_0E * T_E6 can be derived
    T_06 = torch.einsum("xy,byz->xz", T_0E, dht_model[1].transformations[6](torch.empty(1, 1)).inverse())
    assert torch.isclose(T_06, poses[5]).all()

    # theta 5 is still unknown, but the translation_5 is independent, so we can use an arbitrary angle
    # IMPORTANT: T_05_ is not the actual T_05, but only used to extract the P_5
    T_05_ = torch.einsum("xy,byz->xz", T_06, dht_model[1].transformations[5](torch.empty(1, 1)).inverse())
    P_5 = T_05_[:3, -1]
    assert torch.isclose(P_5, poses[4][:3, -1]).all()

    # Theta 1
    theta_1_a = torch.atan2(P_5[1], P_5[0]) + .5 * torch.pi
    theta_1_b = torch.acos(dht_params[3]["d"] / (P_5[0] ** 2 + P_5[1] ** 2) ** .5)

    # 2 solutions
    theta_1 = torch.stack((theta_1_a + theta_1_b, theta_1_a - theta_1_b))

    assert angles_equal(theta_1, angles[:, 0]), f"{theta_1}, {angles[:, 0]}"

    # Theta 5
    # 4 solutions
    theta_5_ = torch.acos(
        (T_06[0, -1] * theta_1.sin() - T_06[1, -1] * theta_1.cos() - dht_params[3]["d"]) /
        dht_params[5]["d"])

    theta_5 = torch.concat((theta_5_, -theta_5_))
    theta_5 = torch.nan_to_num(theta_5)

    assert angles_equal(theta_5, angles[:, 4]), f"{theta_5}, {angles[:, 4]}"

    # Theta 6
    # 4 solutions

    theta_6 = torch.atan2((-T_06[0, 1] * theta_1.sin() + T_06[1, 1] * theta_1.cos()).repeat(2) / theta_5.sin(),
                          (T_06[0, 0] * theta_1.sin() - T_06[1, 0] * theta_1.cos()).repeat(2) / theta_5.sin())

    theta_6 = torch.nan_to_num(
        theta_6) * theta_5.sin().bool()  # if sin(theta_5) == 0, set to 0 (arbitrary value possible)

    assert angles_equal(theta_6, angles[:, 5]), f"{theta_6}, {angles[:, 5]}"

    # Theta 3

    # one pose for each possibility
    T_05 = torch.einsum("xy,pyz->pxz", T_06, dht_model[1].transformations[5](theta_6.unsqueeze(-1)).inverse())
    T_04 = torch.einsum("pxy,pyz->pxz", T_05, dht_model[1].transformations[4](theta_5.unsqueeze(-1)).inverse())

    assert torch.isclose(T_05, poses[4], atol=1e-3).all(-1).all(-1).any()
    assert torch.isclose(T_04, poses[3], atol=1e-3).all(-1).all(-1).any()

    T_01 = dht_model[1].transformations[0](theta_1.unsqueeze(-1))
    assert torch.isclose(T_01, poses[0], atol=1e-3).all(-1).all(-1).any()

    T_03_ = torch.einsum("pxy,pyz->pxz", T_04, dht_model[1].transformations[3](torch.empty(1, 1)).inverse())
    T_13_ = torch.einsum("pxy,pyz->pxz", T_01.repeat(2, 1, 1).inverse(), T_03_)

    P_13 = T_13_[:, :3, -1]

    P_13_xy_length = torch.linalg.norm(P_13[:, :2], axis=-1)

    theta_3_ = torch.acos((P_13_xy_length ** 2 - dht_params[1]["a"] ** 2 - dht_params[2]["a"] ** 2) /
                          (2 * dht_params[1]["a"] * dht_params[2]["a"]))

    theta_3 = torch.concat((theta_3_, -theta_3_))

    theta_3 = torch.nan_to_num(theta_3)

    assert angles_equal(theta_3, angles[:, 2]), f"{theta_3}, {angles[:, 2]}"

    # Theta 2

    theta_2 = -torch.atan2(P_13[:, 1], -P_13[:, 0]).repeat(2) + torch.asin(
        dht_params[2]["a"] * theta_3.sin() / P_13_xy_length.repeat(2))

    assert angles_equal(theta_2, angles[:, 1]), f"{theta_2}, {angles[:, 1]}"

    # Theta 4
    T_12 = dht_model[1].transformations[1](theta_2.unsqueeze(-1))
    T_23 = dht_model[1].transformations[2](theta_3.unsqueeze(-1))

    T_03 = torch.einsum("pij,pjk,pkl->pil", T_01.repeat(4, 1, 1), T_12, T_23)

    T_34 = torch.einsum("pxy,pyz->pxz", T_03.inverse(), T_04.repeat(2, 1, 1))

    theta_4 = torch.atan2(T_34[:, 1, 0], T_34[:, 0, 0])

    assert angles_equal(theta_4, angles[:, 3]), f"{theta_4}, {angles[:, 3]}"

    solutions = torch.stack((
        theta_1.repeat(4), theta_2, theta_3, theta_4, theta_5.repeat(2), theta_6.repeat(2)
    ), dim=-1)

    assert angles_equal(solutions, angles)

    return solutions


if __name__ == '__main__':

    p.connect(p.DIRECT)

    robot = RobotUR5(p)

    state = torch.rand(1, len(joint_limits))
    # state = torch.zeros(1, len(joint_limits))

    # state[0, 1] = .1

    robot.reset({"arm": {"joint_Ps": state[0]}}, force=True)
    robot.visualize_tcp()
    print(state)

    angles = dht_model[0](state)

    print(angles)

    poses = dht_model(state)[0]

    for pose in poses:
        plot_pose(pose, .3)

    # input()

    print(poses[6])
    print(poses[5])

    T_0E = poses[6]

    print(angles)
    print(calculate_ur5_ik(T_0E))
