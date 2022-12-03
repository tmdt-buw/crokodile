import logging
import sys
import os
import re
from multiprocessing import cpu_count
from pathlib import Path
import tempfile
import networkx as nx

import torch
import numpy as np
from ray.rllib.offline.json_reader import JsonReader

import wandb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from environments.environment_robot_task import EnvironmentRobotTask
from environments.environments_robot_task.robots import get_robot

from config import wandb_config
from expert import Expert


class Bridge:
    def __init__(self, config):
        self.robotA = get_robot(config["robot_config_A"])
        self.robotB = get_robot(config["robot_config_B"])

    def map_trajectories(self, trajectories):
        for trajectory in trajectories:
            yield self.map_trajectory(trajectory)

    def map_trajectory(self, trajectory):
        print("#" * 80)

        joint_positions_A = np.stack([obs["state"]["robot"]["arm"]["joint_positions"] for obs in trajectory["obs"]])
        joint_positions_A = torch.from_numpy(joint_positions_A).float()

        joint_angles_A = self.robotA.state2angle(joint_positions_A)
        tcp_poses = self.robotA.forward_kinematics(joint_angles_A)[:, -1]
        joint_angles_B = self.robotB.inverse_kinematics(tcp_poses)

        if joint_angles_B.isnan().any(-1).all(-1).any():
            # at least one state from trajectory could not be mapped
            return None

        joint_positions_B = self.robotB.angle2state(joint_angles_B)

        G = nx.DiGraph()
        G.add_node("start")
        G.add_node("end")

        # add edges from start to first states
        for nn, jp in enumerate(joint_positions_B[0]):
            G.add_edge("s", f"0/{nn}", attr={"from": -1, "to": None, "weight": 0.})

        # add edges from last states to end
        for nn, jp in enumerate(joint_positions_B[-1]):
            G.add_edge(f"{len(joint_positions_B) - 1}/{nn}", "e", attr={"from": (len(joint_positions_B) - 1,nn), "to": None, "weight": 0.})

        for nn, (jp, jp_next) in enumerate(zip(joint_positions_B[:-1], joint_positions_B[1:])):
            actions = (jp_next.unsqueeze(0) - jp.unsqueeze(1)) / self.robotB.scale

            # todo integrate kinematic chain similarity

            # select only valid edges
            actions_max = torch.nan_to_num(actions, torch.inf).abs().max(-1)[0]
            # idx_valid = torch.where(actions_max.isfinite())
            idx_valid = torch.where(actions_max < 1.)

            assert idx_valid[0].shape[0] > 0, "no valid actions found"

            for xx, yy in zip(*idx_valid):
                G.add_edge(f"{nn}/{xx}", f"{nn+1}/{yy}", attr={"from": (nn,xx), "to": (nn+1,yy), "weight": actions_max[xx,yy]})

        path = nx.dijkstra_path(G, "s", "e")[1:-1]

        idx = [int(node.split("/")[1]) for node in path]
        best_states = joint_positions_B[range(len(joint_positions_B)), idx]

        print(best_states)

        ...
        # np.stack([obs["state"]["robot"]["arm"]["joint_positions"] for obs in trajectory["obs"]])


        return None


if __name__ == "__main__":
    config = {
        "robot_config_A": {
            "name": "panda",
            "sim_time": .1,
            "scale": .1,
        },
        "robot_config_B": {
            "name": "ur5",
            "sim_time": .1,
            "scale": .2,
        },
        "task_config": {
            "name": "reach",
            "max_steps": 25,
            "accuracy": .03,
        },
    }

    with wandb.init(config=config, **wandb_config):
        bridge = Bridge(config)

        trajectories = Expert.load_demonstrations_from_wandb()

        list(bridge.map_trajectories(trajectories))
