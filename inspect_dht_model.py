import time

import numpy as np
import pybullet as p
import torch

from dht import DHT_Model
from nn import Rescale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_results = True

if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from karolos.environments.environments_robot_task.robots import get_robot

    p.connect(p.GUI)
    # p.connect(p.DIRECT)
    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=70,
                                 cameraPitch=-27,
                                 cameraTargetPosition=(0, 0, 0)
                                 )


    def clear_lines(line_ids):
        for lineid in line_ids:
            p.removeUserDebugItem(lineid)


    def plot_pose(pose, length=.1):
        position = pose[:, -1]
        orientation = pose[:, :3]

        line_ids = []

        for idx in range(3):
            color = [0] * 3
            color[idx] = 1
            line_id = p.addUserDebugLine(position, position + length * orientation[:, idx], color)
            line_ids.append(line_id)

        return line_ids

    robot = get_robot({
        "name": "panda",
        "scale": .1,
        "sim_time": .1
    }, bullet_client=p)

    data_file = "data/panda_10000_1000.pt"
    model_file = "results/learn_dht/20220320-101846/2/model.pt"

    data = torch.load(data_file)

    X_train, y_train, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    state = np.zeros(len(robot.joints))

    state[:len(X_test[0])] = X_test[0]
    robot.reset(state)


    exit()
    dht_params = data["dht_params"]

    # model = DHT_Model(robot.dht_params)
    model = torch.nn.Sequential(
        Rescale(X_train.shape[1:]),
        DHT_Model(dht_params)
    )

    model.load_state_dict(torch.load(model_file))
    model.eval()

    line_ids = []

    while True:
        state = robot.reset()
        state_joints_arm = torch.FloatTensor(state["joint_positions"][:len(robot.joints_arm)])

        # joint_limits = torch.tensor([joint.limits for joint in robot.joints_arm.values()])

        #pre_model = Rescale(len(state_joints_arm))
        #pre_model.m.data = (joint_limits[:, 1] - joint_limits[:, 0]) / 2
        ##pre_model.c.data = (joint_limits[:, 1] - joint_limits[:, 0]) / 2 + joint_limits[:, 0]

        # angles = pre_model(state_joints_arm.unsqueeze(0))
        # dht_prediction = model(angles).cpu().detach().numpy()
        dht_prediction = model(state_joints_arm.unsqueeze(0)).cpu().detach().numpy()

        #for pose in dht_prediction[0, :, :3]:
        #    plot_pose(pose, .2)
        #    input()

        tcp_pose = dht_prediction[0, -1, : 3]
        print(tcp_pose)
        line_ids += plot_pose(tcp_pose)
        # print(tcp_pose)

        tcp_position, tcp_orientation = robot.get_tcp_pose()
        #
        tcp_position = np.expand_dims(tcp_position, 1)
        tcp_orientation = np.array(p.getMatrixFromQuaternion(tcp_orientation)).reshape(3, 3)
        tcp_pose = np.concatenate((tcp_orientation, tcp_position), axis=1)
        print(tcp_pose)

        #line_ids += plot_pose(tcp_pose)

        #
        input("Press Enter for new pose!")
        clear_lines(line_ids)
        line_ids = []