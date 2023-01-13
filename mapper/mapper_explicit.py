from copy import deepcopy

import networkx as nx
import numpy as np
import torch

from environments.environments_robot_task.robots import get_robot
from utils.nn import KinematicChainLoss, get_weight_matrices

from . import Mapper


class MapperExplicit(Mapper):
    def __init__(self, config):
        super(MapperExplicit, self).__init__(config)

    def generate(self):
        self.robot_source = get_robot(
            self.config["EnvSource"]["env_config"]["robot_config"]
        )
        self.robot_target = get_robot(
            self.config["EnvTarget"]["env_config"]["robot_config"]
        )

        angles_source = torch.zeros(
            (1,) + self.robot_source.state_space["arm"]["joint_positions"].shape
        )
        link_poses_source = self.robot_source.forward_kinematics(angles_source)
        link_positions_source = link_poses_source[0, :, :3, 3]

        angles_target = torch.zeros(
            (1,) + self.robot_target.state_space["arm"]["joint_positions"].shape
        )
        link_poses_target = self.robot_target.forward_kinematics(angles_target)
        link_positions_target = link_poses_target[0, :, :3, 3]

        weight_matrices = get_weight_matrices(
            link_positions_source,
            link_positions_target,
            self.config["Mapper"]["weight_matrix_exponent"],
        )

        self.kcl = KinematicChainLoss(*weight_matrices, reduction=False)

    def load(self):
        # For the explicit mapper, there is nothing to load
        self.generate()

    @classmethod
    def get_relevant_config(cls, config):
        return {
            "EnvSource": {
                "env_config": {
                    "robot_config": config["EnvSource"]["env_config"]["robot_config"]
                }
            },
            "EnvTarget": {
                "env_config": {
                    "robot_config": config["EnvTarget"]["env_config"]["robot_config"]
                }
            },
            "Mapper": {
                "weight_kcl": config["Mapper"]["weight_kcl"],
                "weight_matrix_exponent": config["Mapper"]["weight_matrix_exponent"],
            },
        }

    def map_trajectories(self, trajectories):
        for trajectory in trajectories:
            try:
                yield self.map_trajectory(trajectory)
            except (AssertionError, nx.exception.NetworkXNoPath) as e:
                pass

    def map_trajectory(self, trajectory):

        joint_positions_source = np.stack(
            [
                obs["state"]["robot"]["arm"]["joint_positions"]
                for obs in trajectory["obs"]
            ]
        )
        joint_positions_source = torch.from_numpy(joint_positions_source).float()

        joint_angles_source = self.robot_source.state2angle(joint_positions_source)
        poses_source = self.robot_source.forward_kinematics(joint_angles_source)
        poses_tcp = poses_source[:, -1]
        joint_angles_target = self.robot_target.inverse_kinematics(poses_tcp)
        poses_target = self.robot_target.forward_kinematics(
            joint_angles_target.flatten(0, -2)
        )

        poses_source_ = poses_source.repeat_interleave(joint_angles_target.shape[1], 0)

        kcl = (
            self.kcl(poses_source_, poses_target)
            .squeeze()
            .reshape(joint_angles_target.shape[:2])
        )

        # .reshape(
        # *joint_angles_target.shape[:2], -1, 4, 4).shape

        # map joint angles inside joint limits (if possible)
        while (
            mask := joint_angles_target < self.robot_target.joint_limits[:, 0]
        ).any():
            joint_angles_target[mask] += 2 * np.pi

        while (
            mask := joint_angles_target > self.robot_target.joint_limits[:, 1]
        ).any():
            joint_angles_target[mask] -= 2 * np.pi

        mask = (joint_angles_target < self.robot_target.joint_limits[:, 0]) & (
            joint_angles_target > self.robot_target.joint_limits[:, 1]
        )

        # invalidate states which are outside of joint limits
        joint_angles_target[mask] = torch.nan

        if joint_angles_target.isnan().any(-1).all(-1).any():
            # todo: replace with custom exception
            raise AssertionError(
                "At least one state from trajectory could not be mapped."
            )

        joint_positions_target = self.robot_target.angle2state(joint_angles_target)

        G = nx.DiGraph()
        G.add_node("start")
        G.add_node("end")

        # add edges from start to first states
        for nn, jp in enumerate(joint_positions_target[0]):
            G.add_edge("s", f"0/{nn}", attr={"from": -1, "to": None, "weight": 0.0})

        # add edges from last states to end

        for nn, (jp, kcl_) in enumerate(zip(joint_positions_target[-1], kcl[-1])):
            if torch.isfinite(kcl_):
                G.add_edge(
                    f"{len(joint_positions_target) - 1}/{nn}",
                    "e",
                    attr={
                        "from": (len(joint_positions_target) - 1, nn),
                        "to": None,
                        "weight": self.config["Mapper"]["weight_kcl"] * kcl_,
                    },
                )

        for nn, (jp, kcl_, jp_next) in enumerate(
            zip(joint_positions_target[:-1], kcl[:-1], joint_positions_target[1:])
        ):
            actions = (jp_next.unsqueeze(0) - jp.unsqueeze(1)) / self.robot_target.scale

            # select only valid edges
            actions_max = torch.nan_to_num(actions, torch.inf).abs().max(-1)[0]
            # idx_valid = torch.where(actions_max.isfinite())
            idx_valid = torch.where(actions_max < 1.0)

            assert idx_valid[0].shape[0] > 0, "no valid actions found"
            assert kcl_[idx_valid[0]].isfinite().all(), "kcl is not finite"

            for xx, yy in zip(*idx_valid):
                G.add_edge(
                    f"{nn}/{xx}",
                    f"{nn + 1}/{yy}",
                    attr={
                        "from": (nn, xx),
                        "to": (nn + 1, yy),
                        "weight": actions_max[xx, yy]
                        + self.config["Mapper"]["weight_kcl"] * kcl_[xx],
                    },
                )

        path = nx.dijkstra_path(G, "s", "e")[1:-1]

        idx = [int(node.split("/")[1]) for node in path]
        best_states = joint_positions_target[range(len(joint_positions_target)), idx]
        best_actions = (best_states[1:] - best_states[:-1]) / self.robot_target.scale

        trajectory = deepcopy(trajectory)

        for old_state, new_state in zip(trajectory["obs"], best_states):
            old_state["state"]["robot"]["arm"]["joint_positions"] = (
                new_state.cpu().detach().numpy()
            )

        for old_action, new_action in zip(trajectory["actions"], best_actions):
            old_action["arm"] = new_action.cpu().detach().tolist()

        return trajectory
