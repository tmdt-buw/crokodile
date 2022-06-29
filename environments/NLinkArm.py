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
    def __init__(self, link_lengths, scales=np.pi, max_steps=20):

        # self.observation_space = spaces.Box(-1., 1., (len(link_lengths) + 4,))
        self.observation_space = spaces.Box(-1., 1., (len(link_lengths) + 2,))  # zwei weitere wegen goal
        self.action_space = spaces.Box(-1., 1., (len(link_lengths),))

        # fuer stable baselines
        self.metadata = {'render.modes': []}
        # self.reward_range = (-float('inf'), float('inf'))
        self.reward_range = (-1, 1)
        self.max_steps = max_steps
        self.step_counter = 0
        ##

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.zeros_like(link_lengths)

        self.position_base = np.zeros(2)
        # self.max_length = sum(self.link_lengths)

        if scales is None:
            self.scales = np.ones_like(self.joint_angles)
        elif type(scales) == list:
            assert len(scales) == len(self.joint_angles)
            self.scales = scales
        elif type(scales) in [int, float]:
            self.scales = np.ones_like(self.joint_angles) * scales

        self.goal = np.zeros(2)

        self.reset()

    # reward-Funktion von mir ergaenzt
    def reward_function(self, goal, state, done):
        distance = np.linalg.norm(state[-4:-2] - goal)  # tcp [-4:-2]
        achieved = distance < 0.05
        # achieved = distance < 0.15
        if achieved:
            reward = 1
        elif done:
            reward = -1
        else:
            reward = np.exp(-5 * np.linalg.norm(state[-4:-2] - goal)) - 1

        return reward
    ##

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

        state = np.concatenate([state_joint_angles, self.goal])

        return state

    def step(self, action, last_action_of_sequence=True):   # nur dann counter erhoehen
        ## anpassung stable_baseline
        if last_action_of_sequence:
            self.step_counter += 1
        ###

        action = np.array(action)

        scaled_action = self.scales * action
        self.joint_angles += scaled_action

        self.joint_angles = np.arctan2(np.sin(self.joint_angles),
                                       np.cos(self.joint_angles))

        # anpassung fuer stable-baselines
        # return self.get_state()

        if self.step_counter == self.max_steps:
            done = True
            self.step_counter = 0
        else:
            done = False

        get_next_state = self.get_state()
        get_tcp = self.get_tcp(torch.from_numpy(get_next_state)).squeeze()
        # get_reward = self.reward_function(self.goal, get_next_state, done)

        info = {}
        # distance = np.linalg.norm(get_next_state[-4:-2] - self.goal)  # tcp [-4:-2]

        distance = np.linalg.norm(get_tcp[:2] - self.goal)
        # achieved = distance < 0.05
        achieved = distance < 0.1
        if achieved:
            reward = 1
            done = True
        elif done:
            reward = -1
        else:
            reward = np.exp(-5 * distance) - 1
            reward = reward/10
        reward *= 100
        return get_next_state, reward, done, info

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

        # link_positions /= self.max_length

        return link_positions, accumulated_angle

    def transition(self, state, action):

        angles = state[:, :-4] * np.pi
        action = action * torch.tensor(self.scales, device=state.device)

        new_angles = angles + action
        new_angles = torch.atan2(new_angles.sin(), new_angles.cos())

        next_state= new_angles / np.pi

        return next_state

    def get_tcp(self, state):
        angles = state * np.pi
        accumulated_angles = angles.cumsum(-1).unsqueeze(0)

        tcp_state = torch.tensor(self.position_base).unsqueeze(0).repeat(1, 2).to(state.device)

        for ii, link in enumerate(self.link_lengths):
            tcp_state[:, 0] = tcp_state[:, 0] + \
                                 link * accumulated_angles[:, ii].cos()
            tcp_state[:, 1] = tcp_state[:, 1] + \
                                 link * accumulated_angles[:, ii].sin()


        # orientation tcp
        tcp_state[:, 2] = accumulated_angles[:, -1].sin()
        tcp_state[:, 3] = accumulated_angles[:, -1].cos()

        return tcp_state

    def points_of_interest(self, state):

        angles = state[:, :-4] * np.pi

        accumulated_angles = angles.cumsum(-1)

        points_of_interest = torch.empty(state.shape[0], len(self.joint_angles) + 1, 2)

        points_of_interest[:,0] = torch.tensor(self.position_base).unsqueeze(0).expand(state.shape[0],2).to(state.device)

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

    arm.reset([0,0.25,.25])

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
