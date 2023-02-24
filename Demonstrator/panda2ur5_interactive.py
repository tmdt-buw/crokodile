import torch
import pybullet as p
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from copy import deepcopy

import pybullet as p
import torch

from environments import get_env
from environments.environments_robot_task.robots import get_robot
from environments.experts import get_expert
from utils.nn import KinematicChainLoss, Sawtooth, get_weight_matrices

p.connect(p.GUI)
# p.connect(p.DIRECT)
p.setRealTimeSimulation(True)

# robot_A = get_robot({
#     "name": "panda",
#     "scale": .1,
#     "sim_time": .1,
# }, p)
env_A = get_env(
    {
        "name": "robot-task",
        "robot_config": {"name": "panda", "scale": 0.1, "sim_time": 0.1},
        "task_config": {"name": "pick_place", "single_shot_gripper": True},
        "bullet_client": p,
    }
)

expert_A = get_expert(env_A)

env_B = get_env(
    {
        "name": "robot-task",
        "robot_config": {
            "name": "ur5",
            "scale": 0.1,
            "sim_time": 0.1,
            "offset": (1, 0, 0),
        },
        "task_config": {"name": "pick_place", "offset": (1, 0, 0)},
        "bullet_client": p,
    }
)

angle2pi = Sawtooth(-torch.pi, torch.pi, -torch.pi, torch.pi)

for link_A in range(p.getNumJoints(env_A.robot.model_id)):
    p.setCollisionFilterGroupMask(env_A.robot.model_id, link_A, 0, 0)

for link_B in range(p.getNumJoints(env_B.robot.model_id)):
    p.setCollisionFilterGroupMask(env_B.robot.model_id, link_B, 0, 0)

link_positions_A = env_A.robot.forward_kinematics(
    torch.Tensor([env_A.robot.get_state()["arm"]["joint_positions"]])
)[0, :, :3, -1]
link_positions_B = env_B.robot.forward_kinematics(
    torch.Tensor([env_B.robot.get_state()["arm"]["joint_positions"]])
)[0, :, :3, -1]

weight_matrix_p, weight_matrix_o = get_weight_matrices(
    link_positions_A, link_positions_B, 100
)

kcl_loss = KinematicChainLoss(weight_matrix_p, weight_matrix_o, reduction=False)


def map_state(state_A, init_state_B=None):
    angles_A = env_A.robot.state2angle(
        torch.Tensor([state_A["robot"]["arm"]["joint_positions"]])
    )

    poses_A = env_A.robot.forward_kinematics(angles_A)
    # print(angles_A)

    ik_solutions_B = env_B.robot.inverse_kinematics(poses_A[0, -1:])
    ik_solutions_B = ik_solutions_B[~ik_solutions_B.isnan().any(-1)]

    if len(ik_solutions_B):
        angles_B = angle2pi(ik_solutions_B)
        states_B = env_B.robot.angle2state(angles_B)

        if init_state_B is not None:
            # remove invalid actions
            actions = (states_B - init_state_B) / env_B.robot.scale
            # mask = actions.abs().max(-1)[0] < 1.
            # if mask.any():
            #     states_B = states_B[mask]
            #     angles_B = angles_B[mask]
            best_state_B = states_B[actions.abs().max(-1)[0].argmin(0)]
        else:
            poses_B = env_B.robot.forward_kinematics(angles_B)

            loss = kcl_loss(poses_A.repeat(len(poses_B), 1, 1, 1), poses_B).squeeze()

            # worst_state_B = states_B[loss.argmax(0).squeeze()]
            best_state_B = states_B[loss.argmin(0)]
    else:
        best_state_B = None

    return best_state_B


bc_states_B, bc_actions_B = [], []
multistep = False

while len(bc_states_B) < 100:
    observation_A = env_A.reset()

    state_A = observation_A["state"]
    goal = observation_A["goal"]

    done_A = False

    state_B = map_state(state_A)

    if state_B is not None:
        env_B.reset(
            {
                "state": {"robot": {"arm": {"joint_positions": state_B}}},
                "goal": goal,
            },
            force=True,
        )
    else:
        env_B.reset({"goal": goal}, force=True)

    trajectory_B = [state_B]

    while not done_A:
        action_A = expert_A.predict(state_A, goal)

        if action_A is None:
            break

        observation_A, reward_A, done_A, info_A = env_A.step(action_A)

        state_A = observation_A["state"]
        goal = observation_A["goal"]

        state_B_ = map_state(state_A, state_B)

        if state_B_ is not None:
            print(state_B_)
            if state_B is not None:
                action_B = deepcopy(action_A)
                while not torch.isclose(state_B, state_B_, atol=1e-3).all():
                    action_B["arm"] = (state_B_ - state_B) / env_B.robot.scale

                    if action_B in env_B.action_space:
                        trajectory_B.append(state_B)
                        trajectory_B.append(action_B)

                        # observation_B, reward_B, done_B, info_B = env_B.step(action_B)
                        # state_B = torch.Tensor(observation_B["state"]["robot"]["arm"]["joint_positions"])
                    else:
                        action_B["arm"] = torch.clip(action_B["arm"], -1, 1)

                    observation_B, reward_B, done_B, info_B = env_B.step(action_B)
                    state_B = torch.Tensor(
                        observation_B["state"]["robot"]["arm"]["joint_positions"]
                    )

                    if not multistep:
                        break
            else:
                env_B.reset(
                    {
                        "state": {"robot": {"arm": {"joint_positions": state_B_}}},
                        "goal": goal,
                    },
                    force=True,
                )
                trajectory_B = []

    print(len(trajectory_B))

    bc_states_B += trajectory_B[:-1:2]
    bc_actions_B += trajectory_B[1::2]
