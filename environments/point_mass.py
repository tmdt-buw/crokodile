from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from gym import spaces


class PointMass:
    def __init__(self, dof=2, action_type="velocity",
                 state_transformation=None, action_transformation=None):
        self.position = np.zeros(dof)
        self.velocity = np.zeros(dof)

        assert action_type in ["velocity", "acceleration"]
        self.action_type = action_type

        if state_transformation is not None:
            self.state_transformation = state_transformation
        else:
            self.state_transformation = lambda state: state

        if action_transformation is not None:
            self.action_transformation = action_transformation
        else:
            self.action_transformation = lambda action: action

        self.action_space = spaces.Box(-1., 1., shape=(dof,))

        self.observation_space = spaces.Dict({
            "position": spaces.Box(-1., 1., shape=(dof,)),
            "velocity": spaces.Box(-1., 1., shape=(dof,))
        })

        self.trajectory = []

        self.reset()

    def get_observation(self):
        observation = {
            "position": self.position,
            "velocity": self.velocity,
        }

        return observation

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action)

        self.trajectory.append(deepcopy(action))
        action = self.action_transformation(action)

        if self.action_type == "velocity":
            self.velocity = action
        elif self.action_type == "acceleration":
            self.velocity += action
        else:
            raise NotImplementedError()

        self.velocity = np.clip(
            self.velocity,
            self.observation_space["velocity"].low,
            self.observation_space["velocity"].high
        )

        self.position += self.velocity

        self.position = np.clip(
            self.position,
            self.observation_space["position"].low,
            self.observation_space["position"].high
        )

        observation = self.get_observation()

        self.trajectory.append(deepcopy(observation))

        observation = self.state_transformation(observation)

        return observation

    def reset(self, desired_state=None):
        if desired_state is None:
            self.position = self.observation_space["position"].sample()
            self.velocity = self.observation_space["velocity"].sample()
        else:
            assert self.observation_space["position"].contains(
                desired_state["position"])
            assert self.observation_space["velocity"].contains(
                desired_state["velocity"])

            self.position = desired_state["position"]
            self.velocity = desired_state["velocity"]

        observation = self.get_observation()
        self.trajectory = [deepcopy(observation)]

        observation = self.state_transformation(observation)

        return observation

    def plot_trajectory(self, ax=None, color=None):
        if ax is None:
            fig, ax = plt.subplots()

        states = self.trajectory[::2]
        positions = np.array([state["position"] for state in states])

        ax.plot(positions[:, 0], positions[:, 1], color=color)
        ax.set_xlabel("x")
        ax.set_ylabel("y")


if __name__ == "__main__":

    pm1 = PointMass(action_type="acceleration",
                    action_transformation=lambda action: action * np.array(
                        [.1, .1]),
                    state_transformation=lambda state: {
                        "position": state["position"] * np.array([1., 1.]),
                        "velocity": state["velocity"] * np.array([1., 1.]),
                    })
    pm2 = PointMass(action_type="acceleration",
                    action_transformation=lambda action: action * np.array(
                        [.1, .1]),
                    state_transformation=lambda state: {
                        "position": state["position"] * np.array([.1, .1]),
                        "velocity": state["velocity"] * np.array([1., 1.]),
                    })

    fig, ax = plt.subplots()

    for epoch in range(25):
        pm1.reset({
            "position": np.zeros(2),
            "velocity": np.zeros(2),
        })
        pm2.reset({
            "position": np.zeros(2),
            "velocity": np.zeros(2),
        })

        for step in range(100):
            pm1.step(pm1.action_space.sample())
            pm2.step(pm2.action_space.sample())

        pm1.plot_trajectory(ax, color=f"C0")
        # pm1.plot_trajectory(ax, color=f"C{epoch % 10}")
        pm2.plot_trajectory(ax, color="C1")

    plt.show()
