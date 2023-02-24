import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pylab as p
import torch

sys.path.append(str(Path(__file__).resolve().parent))

from robot import Link, Robot

# todo implement domain randomization


class RobotUR5(Robot):
    def __init__(
        self,
        bullet_client=None,
        offset=(0, 0, 0),
        sim_time=0.0,
        scale=1.0,
        parameter_distributions=None,
        **kwargs
    ):
        if parameter_distributions is not None:
            logging.warning("Domain randomization not implemented for UR5")
            raise NotImplementedError()

        # load robot in simulation
        urdf_file = os.path.join(str(Path(__file__).absolute().parent), "UR5/ur5.urdf")

        joints_arm = {
            "shoulder_pan_joint": (0, (-np.pi, np.pi), 3.15, 300),
            "shoulder_lift_joint": (0, (-np.pi, np.pi), 3.15, 150),
            "elbow_joint": (0, (-np.pi, np.pi), 3.15, 150),
            "wrist_1_joint": (0, (-np.pi, np.pi), 3.2, 28),
            "wrist_2_joint": (0, (-np.pi, np.pi), 3.2, 28),
            "wrist_3_joint": (0, (-np.pi, np.pi), 3.2, 28),
        }

        joints_hand = {
            # hand
            "left_inner_finger_joint": (0.3, (0.0, 0.0425), 2.0, 20),
            "right_inner_finger_joint": (0.3, (0.0, 0.0425), 2.0, 20),
        }

        links = {
            "shoulder_link": Link(3.7, 0.01),
            "upper_arm_link": Link(8.393, 0.01),
            "forearm_link": Link(2.275, 0.01),
            "wrist_1_link": Link(1.219, 0.01),
            "wrist_2_link": Link(1.219, 0.01),
            "wrist_3_link": Link(0.1879, 0.01),
            "robotiq_85_base_link": Link(0.1879, 0.01),
            "left_inner_finger": Link(0.1879, 0.01),
            "right_inner_finger": Link(0.1879, 0.01),
            "tcp": Link(0.0, 0.01),
        }

        dht_params = [
            {"d": 0.089159, "a": 0.0, "alpha": np.pi / 2.0},
            {"d": 0.0, "a": -0.425, "alpha": 0.0},
            {"d": 0.0, "a": -0.39225, "alpha": 0.0},
            {"d": 0.10915, "a": 0.0, "alpha": np.pi / 2.0},
            {"d": 0.09465, "a": 0.0, "alpha": -np.pi / 2.0},
            {"d": 0.0823, "a": 0.0, "alpha": 0},
            {"theta": np.pi / 2, "d": 0.15, "a": 0.0, "alpha": np.pi},
            # {"theta": 0., "d": -.105, "a": 0., "alpha": np.pi},
        ]

        super(RobotUR5, self).__init__(
            bullet_client=bullet_client,
            urdf_file=urdf_file,
            joints_arm=joints_arm,
            joints_hand=joints_hand,
            links=links,
            dht_params=dht_params,
            offset=offset,
            sim_time=sim_time,
            scale=scale,
            parameter_distributions=parameter_distributions,
            **kwargs
        )

        # todo introduce friction

    def inverse_kinematics(self, T_0E):
        if T_0E.dim() == 2:
            T_0E = T_0E.unsqueeze(0)

        # T_6E is constant and known, so T_06 = T_0E * T_E6 can be derived
        T_06 = torch.einsum(
            "axy,oyz->axz",
            T_0E,
            self.dht_model.transformations[6](torch.empty(1, 1)).inverse(),
        )

        # theta 5 is still unknown, but the translation_5 is independent, so we can use an arbitrary angle
        # IMPORTANT: T_05_ is not the actual T_05, but only used to extract the P_5
        T_05_ = torch.einsum(
            "axy,oyz->axz",
            T_06,
            self.dht_model.transformations[5](torch.empty(1, 1)).inverse(),
        )
        P_5 = T_05_[:, :3, -1]

        # Theta 1
        theta_1_a = torch.atan2(P_5[:, 1:2], P_5[:, 0:1]) + 0.5 * torch.pi
        theta_1_b = torch.acos(
            self.dht_params[3]["d"] / (P_5[:, 0:1] ** 2 + P_5[:, 1:2] ** 2) ** 0.5
        )

        # 2 solutions
        theta_1 = torch.concat((theta_1_a + theta_1_b, theta_1_a - theta_1_b), axis=-1)

        # Theta 5
        # 4 solutions
        theta_5_ = torch.acos(
            (
                torch.einsum("a,ax->ax", T_06[:, 0, -1], theta_1.sin())
                - torch.einsum("a,ax->ax", T_06[:, 1, -1], theta_1.cos())
                - self.dht_params[3]["d"]
            )
            / self.dht_params[5]["d"]
        )

        theta_5 = torch.concat((theta_5_, -theta_5_), axis=-1)
        # theta_5 = torch.nan_to_num(theta_5)

        # Theta 6
        # 4 solutions

        theta_6 = torch.atan2(
            (
                torch.einsum("a,ax->ax", -T_06[:, 0, 1], theta_1.sin())
                + torch.einsum("a,ax->ax", T_06[:, 1, 1], theta_1.cos())
            ).repeat(1, 2)
            / theta_5.sin(),
            (
                torch.einsum("a,ax->ax", T_06[:, 0, 0], theta_1.sin())
                - torch.einsum("a,ax->ax", T_06[:, 1, 0], theta_1.cos())
            ).repeat(1, 2)
            / theta_5.sin(),
        )

        theta_6 = torch.where(
            theta_5.sin().bool(), theta_6, torch.zeros(1)
        )  # if sin(theta_5) == 0, set to 0 (arbitrary value possible)

        # Theta 3

        # one pose for each possibility
        T_05 = torch.einsum(
            "axy,apyz->apxz",
            T_06,
            self.dht_model.transformations[5](theta_6.reshape(-1, 1))
            .reshape(*theta_6.shape, 4, 4)
            .inverse(),
        )
        T_04 = torch.einsum(
            "apxy,apyz->apxz",
            T_05,
            self.dht_model.transformations[4](theta_5.reshape(-1, 1))
            .reshape(*theta_5.shape, 4, 4)
            .inverse(),
        )

        T_01 = self.dht_model.transformations[0](theta_1.reshape(-1, 1)).reshape(
            *theta_1.shape, 4, 4
        )

        T_03_ = torch.einsum(
            "apxy,ayz->apxz",
            T_04,
            self.dht_model.transformations[3](torch.empty(1, 1)).inverse(),
        )
        T_13_ = torch.einsum(
            "apxy,apyz->apxz", T_01.repeat(1, 2, 1, 1).inverse(), T_03_
        )

        P_13 = T_13_[:, :, :3, -1]

        P_13_xy_length = torch.linalg.norm(P_13[:, :, :2], axis=-1)

        theta_3_ = torch.acos(
            (
                P_13_xy_length**2
                - self.dht_params[1]["a"] ** 2
                - self.dht_params[2]["a"] ** 2
            )
            / (2 * self.dht_params[1]["a"] * self.dht_params[2]["a"])
        )

        theta_3 = torch.concat((theta_3_, -theta_3_), axis=-1)

        # theta_3 = torch.nan_to_num(theta_3)

        # Theta 2

        theta_2 = -torch.atan2(P_13[:, :, 1], -P_13[:, :, 0]).repeat(1, 2) + torch.asin(
            self.dht_params[2]["a"] * theta_3.sin() / P_13_xy_length.repeat(1, 2)
        )

        # Theta 4
        T_12 = self.dht_model.transformations[1](theta_2.reshape(-1, 1)).reshape(
            *theta_2.shape, 4, 4
        )
        T_23 = self.dht_model.transformations[2](theta_3.reshape(-1, 1)).reshape(
            *theta_3.shape, 4, 4
        )

        T_03 = torch.einsum("apij,apjk,apkl->apil", T_01.repeat(1, 4, 1, 1), T_12, T_23)

        T_34 = torch.einsum("apxy,apyz->apxz", T_03.inverse(), T_04.repeat(1, 2, 1, 1))

        theta_4 = torch.atan2(T_34[:, :, 1, 0], T_34[:, :, 0, 0])

        solutions = torch.stack(
            (
                theta_1.repeat(1, 4),
                theta_2,
                theta_3,
                theta_4,
                theta_5.repeat(1, 2),
                theta_6.repeat(1, 2),
            ),
            dim=-1,
        )

        return solutions


if __name__ == "__main__":
    import pybullet as p

    from utils.nn import Sawtooth

    p.connect(p.GUI)
    robot = RobotUR5(p)

    state = robot.reset()

    print(state)

    angles = robot.state2angle(torch.Tensor([state["arm"]["joint_positions"]]))
    print(angles)

    poses = robot.forward_kinematics(angles)
    ik_solutions = robot.inverse_kinematics(poses[0, -1:])

    angle2pi = Sawtooth(-torch.pi, torch.pi, -torch.pi, torch.pi)

    angles = angle2pi(ik_solutions)

    states = robot.angle2state(angles)

    for configuration in states[~states.isnan().any(-1)]:
        robot.reset({"arm": {"joint_positions": configuration}}, force=True)
        robot.visualize_tcp(0.3)

    robot.reset({"arm": {"joint_positions": state[0, 0]}}, force=True)

    object = p.createMultiBody(
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5] * 3),
        baseCollisionShapeIndex=p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.5] * 3
        ),
        baseMass=0.0,
    )

    p.resetBasePositionAndOrientation(object, [0, 0, 0], [0, 0, 0, 1])

    while True:
        p.stepSimulation()
        # robot.step(robot.action_space.sample())
        #
        # time.sleep(.3)
