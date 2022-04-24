import pybullet as p
import torch
import numpy as np
from dht_learn import DHT_Model
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_results = True

if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from karolos.environments.environments_robot_task.robots import get_robot
    from karolos.agents.nn import NeuralNetwork

    p.connect(p.GUI)

    lines = []

    # p.connect(p.DIRECT)

    def plot_pose(pose, length=.1):
        position = pose[:, -1]
        orientation = pose[:, :3]

        for lineid in lines:
            p.removeUserDebugItem(lineid)

        for lineid in range(3):
            color = [0] * 3
            color[lineid] = 1
            line = p.addUserDebugLine(position, position + length * orientation[:, lineid], color)
            lines.append(line)


    robot = get_robot({
        "name": "panda",
        "scale": .1,
        "sim_time": .1
    }, bullet_client=p)

    dht_params = [
        {pname: pvalue for pname, pvalue in zip(["theta", "a", "d", "alpha"], dht_param) if pvalue is not None} for
        dht_param in robot.dht_params]

    modes = [0] * len(robot.joints_arm)

    model = torch.nn.Sequential(
        NeuralNetwork(len(modes), [('linear', len(modes)), ("tanh", None)] * 2),
        DHT_Model(modes, dht_params)
    )

    # model = torch.nn.DataParallel(model)


    while True:
        from generate_data import generate_samples_karolos
        state, pose = generate_samples_karolos(1)

        state = robot.reset(state)
        state_joints_arm = torch.FloatTensor(state["joint_positions"][:len(robot.joints_arm)])

        dht_prediction = model(state_joints_arm.unsqueeze(0)).cpu().detach().numpy()

        tcp_pose = dht_prediction[0, -1, :3]
        print(tcp_pose)

        # tcp_position, tcp_orientation = robot.get_tcp_pose()
        #
        # tcp_position = np.expand_dims(tcp_position, 1)
        # tcp_orientation = np.array(p.getMatrixFromQuaternion(tcp_orientation)).reshape(3, 3)
        #
        # tcp_pose = np.concatenate((tcp_orientation, tcp_position), axis=1)

        plot_pose(tcp_pose, .3)
        time.sleep(2)