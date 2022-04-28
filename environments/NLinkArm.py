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

        self.state_space = spaces.Box(-1., 1., (len(link_lengths),))
        self.action_space = spaces.Box(-1., 1., (len(link_lengths),))

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.zeros_like(link_lengths)

        self.position_base = np.zeros(2)

        if scales is None:
            self.scales = np.ones_like(self.joint_angles)
        elif type(scales) == list:
            assert len(scales) == len(self.joint_angles)
            self.scales = scales
        elif type(scales) in [int, float]:
            self.scales = np.ones_like(self.joint_angles) * scales

        self.goal = np.zeros(2)

        self.dht_params = [[None, 0., link_length, 0.] for link_length in link_lengths]
        self.joint_limits = [[-np.pi, np.pi] for _ in range(len(self.joint_angles))]

        self.reset()

    def reset(self, desired_state=None):
        if desired_state is None:
            self.joint_angles = np.random.uniform(-np.pi, np.pi, len(self.joint_angles))
        else:
            assert len(desired_state) == len(self.joint_angles)
            self.joint_angles = np.array(desired_state, dtype=float) * np.pi

        return self.get_state()

    def get_state(self):
        # link_positions, tcp_orientation = self.get_link_positions()
        # tcp_position = link_positions[-1]
        #
        state_joint_angles = self.joint_angles / np.pi
        #
        # state = np.concatenate([state_joint_angles, tcp_position,
        #                         np.sin(tcp_orientation),
        #                         np.cos(tcp_orientation)])

        return state_joint_angles

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

        return link_positions, accumulated_angle

    def transition(self, state, action):

        angles = state[:, :-4] * np.pi
        action = action * torch.tensor(self.scales, device=state.device)

        new_angles = angles + action
        new_angles = torch.atan2(new_angles.sin(), new_angles.cos())

        next_state = torch.empty_like(state)

        next_state[:, :-4] = new_angles / np.pi

        accumulated_angles = new_angles.cumsum(-1)

        tcp_position = torch.tensor(self.position_base).unsqueeze(0).repeat(state.shape[0], 1).to(state.device)

        # accumulated_angles = torch.zeros(len(next_state)).to(state.device)

        for ii, link in enumerate(self.link_lengths):
            tcp_position[:, 0] = tcp_position[:, 0] + \
                                 link * accumulated_angles[:, ii].cos()
            tcp_position[:, 1] = tcp_position[:, 1] + \
                                 link * accumulated_angles[:, ii].sin()

        # position tcp
        next_state[:, -4:-2] = tcp_position
        # orientation tcp
        next_state[:, -2] = accumulated_angles[:, -1].sin()
        next_state[:, -1] = accumulated_angles[:, -1].cos()

        return next_state

    def points_of_interest(self, state):

        angles = state[:, :-4] * np.pi

        accumulated_angles = angles.cumsum(-1)

        points_of_interest = torch.empty(state.shape[0], len(self.joint_angles) + 1, 2, device=state.device)

        points_of_interest[:, 0] = torch.tensor(self.position_base, device=state.device).unsqueeze(0).expand(
            state.shape[0], 2)

        for ii, link in enumerate(self.link_lengths):
            points_of_interest[:, ii + 1, 0] = points_of_interest[:, ii, 0] + link * accumulated_angles[:, ii].cos()
            points_of_interest[:, ii + 1, 1] = points_of_interest[:, ii, 1] + link * accumulated_angles[:, ii].sin()

        return points_of_interest

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

        ax.scatter(link_positions[:, 0], link_positions[:, 1], color=color,
                    alpha=alpha)

        if plot_goal:
            ax.scatter(self.goal[0], self.goal[1], color="k", marker="x")

        # ax.set_xlim([-self.max_length, self.max_length])
        # ax.set_ylim([-self.max_length, self.max_length])
        # plt.draw()
        # plt.pause(0.0001)


if __name__ == "__main__":
    configA = {"link_lengths": [1, 1, 1],
               "scales": .3 * np.pi}
    configB = {"link_lengths": [.5, 1, .5, .6, .7],
               "scales": np.pi}

    config = configA

    arm = NLinkArm(**config)

    fig, ax = plt.subplots()

    steps = 50

    state = arm.reset(np.zeros_like(config["link_lengths"]))

    arm.reset([0,0,.5])

    arm.plot()

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    plt.show()
    exit()

    states, actions, next_states = [], [], []

    for step in range(steps):
        action = arm.action_space.sample()

        next_state = arm.step(action)

        states.append(state)
        actions.append(action)
        next_states.append(next_state)

        next_state_batch = arm.transition(torch.tensor([state]), torch.tensor([action]))

        assert torch.all((torch.tensor([next_state]) - next_state_batch).abs() < 1e-5)

        state = next_state

        # if step % 10 == 0:
        # arm.plot(ax=ax, alpha=step / steps)
        arm.plot(ax=ax, plot_goal=True)

        # points_of_interest = arm.points_of_interest(torch.tensor([state]))[0]

        # plt.scatter(points_of_interest[:,0], points_of_interest[:,1], color="C2", marker="x")

    states = torch.tensor(states)
    actions = torch.tensor(actions)
    next_states = torch.tensor(next_states)

    # check points of interest
    points_of_interest = arm.points_of_interest(torch.tensor(states))
    points_of_interest = points_of_interest.reshape(-1, 2)
    plt.scatter(points_of_interest[:, 0], points_of_interest[:, 1], color="C2", marker="x")

    next_states_pred = arm.transition(states, actions)

    assert torch.all((next_states - next_states_pred).abs() < 1e-5)

    plt.show()
