import gym.spaces
import klampt
import numpy as np
from klampt.math import so3
from klampt.model import ik


class Expert:
    def __init__(self, env):
        # initialize ik
        self.robot = env.robot
        self.limits = env.task.limits

        self.ik_world = klampt.WorldModel()
        self.ik_world.loadElement(self.robot.urdf_file)
        self.ik_model = self.ik_world.robot(0)
        self.ik_dof_joint_ids = [
            jj for jj in range(self.ik_model.numLinks()) if self.ik_model.getJointType(jj) == "normal"
        ]
        self.ik_dof_joint_ids = self.ik_dof_joint_ids[: len(self.robot.joints)]

        assert len(self.ik_dof_joint_ids) == len(
            self.robot.joints
        ), "Mismatch between specified DOF and DOF found by Klampt!"

    def predict(self, state, goal):
        position_target = goal["desired"]["position"]
        position_target = np.array(
            [np.interp(value, [-1, 1], limits) for value, limits in zip(position_target, self.limits)]
        )

        # current_positions_arm = [np.interp(position, [-1, 1], joint.limits) for joint, position in
        #                          zip(self.robot.joints, state["robot"]["arm"]['joint_positions'])]

        action = self.robot.action_space_.sample()

        ik.setRandomSeed(0)
        ik_conf = np.zeros_like(self.ik_model.getConfig())

        # for ik_dof, pose in zip(self.ik_dof_joint_ids, current_positions_arm):
        #     ik_conf[ik_dof] = pose

        self.ik_model.setConfig(ik_conf)

        obj = ik.objective(
            self.ik_model.link(self.ik_model.numLinks() - 1),
            t=list(position_target),
            R=so3.from_quaternion([1, 0, 0, 0]),
        )

        res = ik.solve_global(obj, activeDofs=self.ik_dof_joint_ids)

        if not res:
            return None

        desired_state_arm = np.array([self.ik_model.getDOFPosition(jj) for jj in self.ik_dof_joint_ids])

        desired_state_arm_normed = np.array(
            [
                np.interp(delta_pose, joint.limits, [-1, 1])
                for joint, delta_pose in zip(self.robot.joints, desired_state_arm)
            ]
        )

        delta_poses_arm = desired_state_arm_normed - state["robot"]["arm"]["joint_positions"]

        action["arm"] = delta_poses_arm / self.robot.scale

        action["arm"] = np.clip(action["arm"], -1.0, 1.0).astype("float32")

        action = gym.spaces.flatten(self.robot.action_space_, action)

        return action
