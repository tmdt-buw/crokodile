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
        position_object = goal["achieved"]["object_position"]
        position_object = np.array(
            [np.interp(value, [-1, 1], limits) for value, limits in zip(position_object, self.limits)]
        )

        position_object_desired = goal["desired"]["object_position"]
        position_object_desired = np.array(
            [np.interp(value, [-1, 1], limits) for value, limits in zip(position_object_desired, self.limits)]
        )

        tcp_position = goal["achieved"]["tcp_position"]
        tcp_position = np.array([np.interp(value, [-1, 1], limits) for value, limits in zip(tcp_position, self.limits)])

        current_positions_arm = [
            np.interp(position, [-1, 1], joint.limits)
            for joint, position in zip(self.robot.joints, state["robot"]["arm"]["joint_positions"])
        ]

        action = self.robot.action_space_.sample()

        orientation_object = [0, 1, 0, 0]

        if np.linalg.norm(position_object - tcp_position) > 0.01:
            # move to object with open gripper
            action["hand"] = np.ones(1, dtype=np.float64)

            goal_position = position_object.copy()
            goal_orientation = orientation_object

            if np.linalg.norm(position_object[:2] - tcp_position[:2]) > 0.03:
                # align over object
                goal_position[-1] += 0.05
        else:
            # grip object
            action["hand"] = -np.ones(1, dtype=np.float64)

            if state["task"]["object_gripped"] > 0:
                goal_position = position_object_desired.copy()
                if np.linalg.norm(position_object[:2] - position_object_desired[:2]) > 0.02:
                    goal_position[-1] += 0.05
                    goal_orientation = orientation_object
                else:
                    goal_orientation = None
            else:
                goal_position = position_object.copy()
                goal_orientation = orientation_object

        # print(goal_position)

        if goal_position is not None:
            ik.setRandomSeed(0)
            ik_conf = np.zeros_like(self.ik_model.getConfig())

            for ik_dof, pose in zip(self.ik_dof_joint_ids, current_positions_arm):
                ik_conf[ik_dof] = pose

            self.ik_model.setConfig(ik_conf)

            # obj = IKObjective()

            # obj.setFixedPoint(self.ik_model.numLinks() - 1, [0,0,0], list(goal_position))

            if goal_orientation is None:
                obj = ik.objective(
                    self.ik_model.link(self.ik_model.numLinks() - 1), local=[0, 0, 0], world=list(goal_position)
                )
            else:
                obj = ik.objective(
                    self.ik_model.link(self.ik_model.numLinks() - 1),
                    t=list(goal_position),
                    R=so3.from_quaternion(goal_orientation),
                )

            res = ik.solve_global(obj, activeDofs=self.ik_dof_joint_ids)

            if not res:
                return None

            desired_state_arm = np.array([self.ik_model.getDOFPosition(jj) for jj in self.ik_dof_joint_ids])

            # print(desired_state_arm)

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
