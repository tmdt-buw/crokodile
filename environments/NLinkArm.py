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


class NLinkArm(object):
    def __init__(self, link_lengths, scales=np.pi):

        self.observation_space = spaces.Box(-1., 1., (len(link_lengths) + 4,))
        self.action_space = spaces.Box(-1., 1., (len(link_lengths),))

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.zeros_like(link_lengths)

        self.position_base = np.zeros(2)
        self.max_length = sum(self.link_lengths)

        if scales is None:
            self.scales = np.ones_like(self.joint_angles)
        elif type(scales) == list:
            assert len(scales) == len(self.joint_angles)
            self.scales = scales
        elif type(scales) in [int, float]:
            self.scales = np.ones_like(self.joint_angles) * scales

        self.goal = np.zeros(2)

        self.reset()

    def reset(self, angles=None, goal=None):
        if angles is None:
            self.joint_angles = np.random.uniform(-1, 1,
                                                  len(self.joint_angles))
        else:
            assert len(angles) == len(self.joint_angles)
            self.joint_angles = np.array(angles, dtype=float) * np.pi

        if goal is None:
            self.goal = np.ones(2)
            while np.linalg.norm(self.goal) > 1:
                self.goal = np.random.random(2)
        else:
            self.goal = goal

        return self.get_state()

    def get_state(self):
        link_positions, tcp_orientation = self.get_link_positions()
        tcp_position = link_positions[-1]

        state_joint_angles = self.joint_angles / np.pi

        state = np.concatenate([state_joint_angles, tcp_position,
                                np.sin(tcp_orientation),
                                np.cos(tcp_orientation)])

        return state

    def step(self, action):

        action = np.array(action)

        scaled_action = self.scales * action
        self.joint_angles += scaled_action

        self.joint_angles = np.arctan2(np.sin(self.joint_angles),
                                       np.cos(self.joint_angles))

        return self.get_state()

    def get_link_positions(self):
        accumulated_angle = np.zeros(1)

        link_positions = np.empty((len(self.joint_angles) + 1, 2))

        link_positions[0] = np.copy(self.position_base)

        for ii, (angle, link) in enumerate(
                zip(self.joint_angles, self.link_lengths)):
            accumulated_angle += angle

            link_positions[ii + 1][0] = link_positions[ii][0] + link * np.cos(
                accumulated_angle)
            link_positions[ii + 1][1] = link_positions[ii][1] + link * np.sin(
                accumulated_angle)

        link_positions /= self.max_length

        return link_positions, accumulated_angle

    def transition(self, state, action):

        angles = state[:, :-4] * np.pi
        action = action * torch.tensor(self.scales, device=state.device)

        new_angles = angles + action
        new_angles = torch.atan2(new_angles.sin(), new_angles.cos())

        next_state = torch.empty_like(state)

        next_state[:, :-4] = new_angles / np.pi

        accumulated_angles = new_angles.cumsum(-1)

        tcp_position = torch.zeros(len(next_state), 2).to(state.device)

        # accumulated_angles = torch.zeros(len(next_state)).to(state.device)

        for ii, (angle, link) in enumerate(
                zip(self.joint_angles, self.link_lengths)):
            tcp_position[:, 0] = tcp_position[:,
                                 0] + link * accumulated_angles[:, ii].cos()
            tcp_position[:, 1] = tcp_position[:,
                                 1] + link * accumulated_angles[:, ii].sin()

        next_state[:, -4:-2] = tcp_position
        next_state[:, -2] = accumulated_angles[:, -1].sin()
        next_state[:, -1] = accumulated_angles[:, -1].cos()

        return next_state

    def plot(self, ax=None, color="C0", alpha=1., plot_goal=False):  # pragma: no cover
        # plt.cla()
        # for stopping simulation with the esc key.
        # plt.gcf().canvas.mpl_connect('key_release_event',
        #         lambda event: [exit(0) if event.key == 'escape' else None])

        if ax is None:
            fig, ax = plt.subplots()

        link_positions, tcp_orientation = self.get_link_positions()

        codes = [Path.MOVETO] + [Path.LINETO] * len(self.joint_angles)

        path = Path(link_positions, codes)

        patch = patches.PathPatch(path, fill=False, lw=2, color=color,
                                  alpha=alpha)
        ax.add_patch(patch)

        plt.scatter(link_positions[:, 0], link_positions[:, 1], color=color,
                    alpha=alpha)

        if plot_goal:
            ax.scatter(self.goal[0], self.goal[1], color="k", marker="x")

        ax.set_xlim([-self.max_length, self.max_length])
        ax.set_ylim([-self.max_length, self.max_length])
        # plt.draw()
        # plt.pause(0.0001)


if __name__ == "__main__":
    configA = {"link_lengths": [1, 1],
               "scales": .3 * np.pi}
    configB = {"link_lengths": [.5, 1, .5, .6, .7],
               "scales": np.pi}

    config = configB

    arm = NLinkArm(**config)

    fig, ax = plt.subplots()

    steps = 5

    state = arm.reset(np.zeros_like(config["link_lengths"]))

    states, actions, next_states = [], [], []

    for step in range(steps):
        # action = np.array([1, .1])
        action = arm.action_space.sample()

        next_state = arm.step(action)

        states.append(state)
        actions.append(action)
        next_states.append(next_state)

        state = next_state

        # if step % 10 == 0:
        # arm.plot(ax=ax, alpha=step / steps)
        arm.plot(ax=ax, plot_goal=True)

    states = torch.tensor(states)
    actions = torch.tensor(actions)
    next_states = torch.tensor(next_states)


    next_states_pred = arm.transition(states, actions)

    print(next_states)
    print(next_states_pred)
    print(torch.all((next_states - next_states_pred).abs() < 1e-5))

    plt.show()
