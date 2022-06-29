"""
Class for controlling and plotting an arm with an arbitrary number of links.

Author: Daniel Ingram
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from matplotlib.path import Path
from models.dht import get_dht_model
from gym import Env


class RobotSimple(Env):
    def __init__(self, dht_params, joint_limits, scales=np.pi, max_steps=20):
        self.dht_params = dht_params
        self.joint_limits = joint_limits
        self.dht_model = get_dht_model(dht_params, joint_limits)

        # self.observation_space = spaces.Box(-1., 1., (len(link_lengths) + 4,))
        self.state_space = spaces.Dict({
            "arm": spaces.Dict({
                "joint_positions": spaces.Box(-1., 1., (len(dht_params),))
            }),
            "hand": spaces.Box(-1., 1., shape=(1,), dtype=np.float64)
        })

        self.goal_space = spaces.Box(-1., 1., (3,))

        self.action_space = spaces.Dict({
            "arm": spaces.Box(-1., 1., shape=(len(dht_params),), dtype=np.float64),
            "hand": spaces.Box(-1., 1., shape=(1,), dtype=np.float64)
        })

        self.max_steps = max_steps
        self.step_counter = 0

        if scales is None:
            self.scales = np.ones_like(dht_params)
        elif type(scales) == list:
            assert len(scales) == len(dht_params)
            self.scales = scales
        elif type(scales) in [int, float]:
            self.scales = np.ones_like(dht_params) * scales

        self.reset()

    def success_criterion(self, state, goal):
        poses = self.dht_model(
            torch.tensor(self.state_space["arm"]["joint_positions"].sample(), dtype=torch.float).unsqueeze(0))
        tcp_position = poses[0, -1, :3, -1].detach().cpu().numpy()

        distance = np.linalg.norm(tcp_position - goal)

        return distance < .01

    # reward-Funktion von mir ergaenzt
    def reward_function(self, state, goal, done):
        poses = self.dht_model(
            torch.tensor(self.state_space["arm"]["joint_positions"].sample(), dtype=torch.float).unsqueeze(0))
        tcp_position = poses[0, -1, :3, -1].detach().cpu().numpy()

        distance = np.linalg.norm(tcp_position - goal)

        if self.success_criterion(state, goal):
            reward = 1
        elif done:
            reward = -1
        else:
            reward = np.exp(-distance) - 1

        return reward

    def reset(self, state=None, goal=None):
        if state is None:
            self.state = self.state_space.sample()
        else:
            assert state in self.state_space
            self.state = state

        if goal is None:
            poses = self.dht_model(
                torch.tensor(self.state_space["arm"]["joint_positions"].sample(), dtype=torch.float).unsqueeze(0))
            self.goal = poses[0, -1, :3, -1].cpu().detach().numpy()
        else:
            assert goal in self.goal_space
            self.goal = goal

        return self.state

    def step(self, action):
        assert action in self.action_space

        self.step_counter += 1

        action = action["arm"] * self.scales

        self.state["arm"]["joint_positions"] = self.state["arm"]["joint_positions"] + action

        self.state["arm"]["joint_positions"] = self.state["arm"]["joint_positions"].clip(
            self.state_space["arm"]["joint_positions"].low,
            self.state_space["arm"]["joint_positions"].high).astype(float)

        done = self.success_criterion(self.state, self.goal)
        done |= self.step_counter >= self.max_steps

        # info = {}

        # reward = self.reward_function(self.state, self.goal, done)

        return self.state

    def plot(self, ax=None, color="C0", alpha=1., plot_goal=False):  # pragma: no cover
        # plt.cla()
        # for stopping simulation with the esc key.
        # plt.gcf().canvas.mpl_connect('key_release_event',
        #         lambda event: [exit(0) if event.key == 'escape' else None])

        poses = self.dht_model(torch.tensor(self.state["arm"]["joint_positions"], dtype=torch.float).unsqueeze(0))
        positions = torch.cat((torch.zeros(1, 3), poses[0, :, :3, -1])).detach().cpu().numpy()

        if ax is None:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(projection='3d')

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)

        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])

        ax.scatter(self.goal[0], self.goal[1], self.goal[2], color="k", marker="x")


if __name__ == "__main__":
    configA = {
        "dht_params": [
            {"d": 0., "a": .333, "alpha": 0.},
            {"d": 0., "a": .333, "alpha": 0.},
            {"d": 0., "a": .333, "alpha": 0.},
        ],
        "joint_limits": torch.tensor([[-1, 1], [-1, 1], [-1, 1]]) * np.pi,
        "scales": .3 * np.pi
    }
    configB = {
        "dht_params": [
            {"d": 0., "a": .5, "alpha": 0.},
            {"d": 0., "a": .5, "alpha": 0.},
        ],
        "joint_limits": torch.tensor([[-1, 1], [-1, 1]]) * 2 * np.pi,
        "scales": .3 * np.pi
    }

    arm_A = DHTArm(**configB)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    steps = 10

    state = arm_A.state_space.sample()
    state["arm"]["joint_positions"] = np.zeros_like(state["arm"]["joint_positions"])

    state = arm_A.reset(state)

    arm_A.plot(ax, plot_goal=True)

    states, actions, next_states = [], [], []

    for step in range(steps):
        action = arm_A.action_space.sample()
        action["arm"] = np.zeros_like(action["arm"])
        action["arm"][0] = .1

        next_state = arm_A.step(action)
        arm_A.plot(ax=ax, plot_goal=True)

    plt.show()
