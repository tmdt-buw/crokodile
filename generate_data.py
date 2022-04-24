import numpy as np
import pybullet as p
import torch
from tqdm import tqdm


def generate_samples_karolos(robot, n):
    states = []
    tcp_poses = []

    for _ in tqdm(range(n)):
        state = robot.reset()

        state_joints_arm = state["joint_positions"][:len(robot.joints_arm)]
        tcp_position, tcp_orientation = robot.get_tcp_pose()

        state_joints_arm = torch.FloatTensor(state_joints_arm)
        tcp_position = torch.FloatTensor(tcp_position)
        tcp_orientation = torch.FloatTensor(p.getMatrixFromQuaternion(tcp_orientation)).reshape(3, 3)

        tcp_pose = torch.eye(4)
        tcp_pose[:3,-1:] = tcp_position.unsqueeze(-1)
        tcp_pose[:3,:3] = tcp_orientation

        states.append(state_joints_arm)
        tcp_poses.append(tcp_pose)

    states = torch.stack(states)
    tcp_poses = torch.stack(tcp_poses).unsqueeze(1)

    return states, tcp_poses


def generate_samples_karolos_transitions(robot, n):
    states = []
    actions = []
    next_states = []

    for _ in tqdm(range(n)):
        state = robot.reset()
        action = robot.action_space.sample()
        next_state = robot.step(action)

        state_joints_arm = state["arm"]["joint_positions"]
        state_joints_arm = torch.FloatTensor(state_joints_arm)

        action = torch.FloatTensor(action)

        next_state_joints_arm = next_state["arm"]["joint_positions"]
        next_state_joints_arm = torch.FloatTensor(next_state_joints_arm)

        states.append(state_joints_arm)
        actions.append(action)
        next_states.append(next_state_joints_arm)

    states = torch.stack(states)
    actions = torch.stack(actions)
    next_states = torch.stack(next_states)

    return states, actions, next_states


if __name__ == '__main__':
    from karolos.environments.environments_robot_task.robots import get_robot

    robot_name = "panda"
    n_samples_train = 10_000
    n_samples_test = 1_000
    robot = get_robot({
        "name": robot_name,
        "scale": .1,
        "sim_time": .1
    })

    states_train, actions_train, next_states_train = generate_samples_karolos_transitions(robot, n_samples_train)
    states_test, actions_test, next_states_test = generate_samples_karolos_transitions(robot, n_samples_test)

    torch.save({
        "states_train": states_train,
        "actions_train": actions_train,
        "next_states_train": next_states_train,
        "states_test": states_test,
        "actions_test": actions_test,
        "next_states_test": next_states_test,
    }, f"data/transitions/{robot_name}_{n_samples_train}_{n_samples_test}.pt")

    # if False:
    #     from karolos.environments.environments_robot_task.robots import get_robot
    #
    #     robot_name = "panda"
    #     n_samples_train = 10_000
    #     n_samples_test = 1_000
    #     robot = get_robot({
    #             "name": robot_name,
    #             "scale": .1,
    #             "sim_time": .1
    #         })
    #
    #     X_train, y_train = generate_samples_karolos(robot, n_samples_train)
    #     X_test, y_test = generate_samples_karolos(robot, n_samples_test)
    #     dht_params = robot.dht_params
    #     joint_limits = torch.tensor([joint.limits for joint in robot.joints_arm.values()])

    ########## 2d link arms

    # robot_name = "3link"
    # n_samples_train = 10_000
    # n_samples_test = 1_000
    #
    # if robot_name == "2link":
    #     dht_params = [
    #         {"a": .5, "alpha": 0, "d": 0},
    #         {"a": .5, "alpha": 0, "d": 0},
    #     ]
    # elif robot_name == "3link":
    #     dht_params = [
    #         {"a": 1/3, "alpha": 0, "d": 0},
    #         {"a": 1/3, "alpha": 0, "d": 0},
    #         {"a": 1/3, "alpha": 0, "d": 0},
    #     ]
    # else:
    #     raise ValueError()
    #
    # from learn_dht import DHT_Model
    # dht_model = DHT_Model(dht_params)
    #
    # X_train = torch.rand(n_samples_train, len(dht_params)) * 2 - 1
    # X_test = torch.rand(n_samples_train, len(dht_params)) * 2 - 1
    # y_train = dht_model(X_train)
    # y_test = dht_model(X_test)
    #
    # joint_limits = torch.ones(len(dht_params), 2)
    # joint_limits[:,0] *= -1
    # joint_limits *= np.pi

    # torch.save({
    #     "X_train": X_train,
    #     "y_train": y_train,
    #     "X_test": X_test,
    #     "y_test": y_test,
    #     "dht_params": dht_params,
    #     "joint_limits": joint_limits,
    # }, f"data/{robot_name}_{n_samples_train}_{n_samples_test}.pt")
