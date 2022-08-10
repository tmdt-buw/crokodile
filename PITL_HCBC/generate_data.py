import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

import gym
import numpy as np
import os

import ray._private.utils

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.offline.dataset_writer import DatasetWriter
import torch
import pybullet as p
# from environments.environments_robot_task.robots import get_robot
from environments import get_env
from environments.experts import get_expert
from utils.nn import Sawtooth, KinematicChainLoss, get_weight_matrices
from copy import deepcopy

if __name__ == "__main__":
    # Setup environments
    p.connect(p.DIRECT)

    env_A = get_env({
        "name": "robot-task",
        "robot_config": {
            "name": "panda",
            "scale": .1,
            "sim_time": .1,
        },
        "task_config": {
            "name": "reach"
        },
        "bullet_client": p
    })

    expert_A = get_expert(env_A)

    env_B = get_env({
        "name": "robot-task",
        "robot_config": {
            "name": "ur5",
            "scale": .5,
            "sim_time": .1,
            "offset": (1, 0, 0)
        },
        "task_config": {
            "name": "reach",
            "offset": (1, 0, 0)
        },
        "bullet_client": p
    })

    angle2pi = Sawtooth(-torch.pi, torch.pi, -torch.pi, torch.pi)

    for link_A in range(p.getNumJoints(env_A.robot.model_id)):
        p.setCollisionFilterGroupMask(env_A.robot.model_id, link_A, 0, 0)

    for link_A in range(p.getNumJoints(env_A.robot.model_id)):
        p.setCollisionFilterGroupMask(env_A.robot.model_id, link_A, 0, 0)

    for link_B in range(p.getNumJoints(env_B.robot.model_id)):
        p.setCollisionFilterGroupMask(env_B.robot.model_id, link_B, 0, 0)

    link_positions_A = env_A.robot.forward_kinematics(
        torch.Tensor(env_A.robot.get_state()["arm"]["joint_positions"]).unsqueeze(0))[0, :, :3, -1]
    link_positions_B = env_B.robot.forward_kinematics(
        torch.Tensor(env_B.robot.get_state()["arm"]["joint_positions"]).unsqueeze(0))[0, :, :3, -1]

    weight_matrix_p, weight_matrix_o = get_weight_matrices(link_positions_A, link_positions_B, 100)

    kcl_loss = KinematicChainLoss(weight_matrix_p, weight_matrix_o, reduction=False)


    def map_observation(observation_A, init_observation_B=None, max_actions=1):
        state_A = observation_A["state"]
        goal_A = observation_A["goal"]

        angles_A = env_A.robot.state2angle(torch.Tensor(state_A["robot"]["arm"]["joint_positions"]).unsqueeze(0))

        poses_A = env_A.robot.forward_kinematics(angles_A)

        ik_solutions_B = env_B.robot.inverse_kinematics(poses_A[0, -1:])
        ik_solutions_B = ik_solutions_B[~ik_solutions_B.isnan().any(-1)]

        if len(ik_solutions_B):
            angles_B = angle2pi(ik_solutions_B)
            states_arm_B = env_B.robot.angle2state(angles_B)

            if init_observation_B is not None:
                init_state_arm_B = init_observation_B["state"]["robot"]["arm"]["joint_positions"]

                # remove invalid actions
                actions = (states_arm_B - init_state_arm_B) / env_B.robot.scale

                valid_action_mask = actions.abs().max(-1)[0] <= max_actions

                states_arm_B = states_arm_B[valid_action_mask]
                angles_B = angles_B[valid_action_mask]

            if len(states_arm_B):
                poses_B = env_B.robot.forward_kinematics(angles_B)
                loss = kcl_loss(poses_A.repeat(len(poses_B), 1, 1, 1), poses_B).squeeze()

                best_id = loss.argmin(0)

                state_arm_B = states_arm_B[best_id]
                position_tcp_B = poses_B[best_id][-1, :3, -1]

                state_B = deepcopy(state_A)
                state_B["robot"]["arm"]["joint_positions"] = state_arm_B

                goal_B = deepcopy(goal_A)
                if "tcp_position" in goal_B["achieved"]:
                    goal_B["achieved"]["tcp_position"] = position_tcp_B

                observation_B = {
                    "state": state_B,
                    "goal": goal_B
                }
            else:
                observation_B = None
        else:
            observation_B = None

        return observation_B


    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/bc_data_ur5_reach")
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder

    from ray.rllib.offline.json_writer import JsonWriter

    writer = JsonWriter(data_folder)

    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations. For CartPole a no-op
    # preprocessor is used, but this may be relevant for more complex envs.
    prep = get_preprocessor(env_B.observation_space)(env_B.observation_space)
    print("The preprocessor is", prep)

    total_samples = 10_000

    pbar = tqdm(total=total_samples, desc="Progress")

    while batch_builder.count < total_samples:

        observation_A = env_A.reset()

        state_A = observation_A["state"]
        goal = observation_A["goal"]

        # prev_action = np.zeros_like(env_B.action_space.sample())
        prev_reward = 0
        done = False
        # t = 0
        trajectory_A = [observation_A]

        while not done:
            action_A = expert_A.predict(observation_A["state"], observation_A["goal"])

            if action_A is None:
                break

            next_observation_A, reward, done, info = env_A.step(action_A)

            trajectory_A.append(action_A)
            trajectory_A.append(next_observation_A)

            observation_A = next_observation_A

        if env_A.success_criterion(observation_A["goal"]):

            # observation_B = map_observation(observation_A)
            # if observation_B is None:
            #     continue

            # trajectory_B = [observation_B]
            trajectory_B = []

            from itertools import zip_longest

            for observation_A, action_A in zip(trajectory_A[::2], [None] + trajectory_A[1::2]):
                if not len(trajectory_B):
                    observation_B = map_observation(observation_A)
                    if observation_B is not None:
                        trajectory_B.append(observation_B)
                else:
                    observation_B = map_observation(observation_A, trajectory_B[-1])

                    if observation_B is None:
                        trajectory_B = []
                        continue

                    # if action_A is not None:
                    action_arm_B = (observation_B["state"]["robot"]["arm"]["joint_positions"] -
                                    trajectory_B[-1]["state"]["robot"]["arm"]["joint_positions"]) / env_B.robot.scale

                    action_B = gym.spaces.unflatten(env_A.robot.action_space_, action_A)
                    action_B["arm"] = action_arm_B.numpy()

                    action_B = gym.spaces.flatten(env_B.robot.action_space_, action_B)

                    trajectory_B.append(action_B)
                    trajectory_B.append(observation_B)
                    # else:
                    #     trajectory_B = [observation_B]

            for observation_B, action_B in zip(trajectory_B[::2], trajectory_B[1::2]):
                batch_builder.add_values(
                    # t=t,
                    # eps_id=eps_id,
                    # agent_index=0,
                    obs=prep.transform(observation_B),
                    actions=action_B,
                    # action_prob=1.0,  # put the true action probability here
                    # action_logp=0.0,
                    # rewards=rew,
                    # prev_actions=prev_action,
                    # prev_rewards=prev_reward,
                    # dones=done,
                    # infos=info,
                    # new_obs=prep.transform(new_obs),
                )

                pbar.update(1)
                pbar.refresh()

                # writer.write(batch_builder.build_and_reset())

    writer.write(batch_builder.build_and_reset())
