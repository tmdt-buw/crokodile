"""
Checks if dht model generates correct tcp pose
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
import torch
import wandb
import yaml
from tqdm import tqdm

from models.dht import get_dht_model
from utils.nn import KinematicChainLoss

wandb_mode = "online"
# wandb_mode = "disabled"

sys.path.append(str(Path(__file__).resolve().parents[1]))

from environments.environments_robot_task.robots import get_robot


def clear_lines(line_ids, bullet_client=p):
    for lineid in line_ids:
        bullet_client.removeUserDebugItem(lineid)


def plot_pose(pose, length=.1, bullet_client=p):
    position = pose[:, -1]
    orientation = pose[:, :3]

    line_ids = []

    for idx in range(3):
        color = [0] * 3
        color[idx] = 1
        line_id = bullet_client.addUserDebugLine(position, position + length * orientation[:, idx], color)
        line_ids.append(line_id)

    return line_ids


def verify_calculated_poses(data_file, n=100, visualize=False):
    print("Verify", data_file)
    if visualize:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=70,
                                 cameraPitch=-27,
                                 cameraTargetPosition=(0, 0, 0)
                                 )

    robot = get_robot({
        "name": os.path.basename(data_file).split("_")[0],
        "scale": .1,
        "sim_time": .1
    }, bullet_client=p)

    data = torch.load(data_file)
    line_ids = []

    for _ in range(n):
        state = robot.reset()
        state_joints_arm_A = state["arm"]["joint_positions"]

        dht_model = get_dht_model(data["dht_params"], data["joint_limits"])

        poses = dht_model(torch.tensor(state_joints_arm_A, dtype=torch.float32).unsqueeze(0))[0].cpu().detach().numpy()

        for pose in poses[:, :3]:
            line_ids += plot_pose(pose, .2)

        # tcp_pose = poses[-1, :3]
        # print(tcp_pose)
        # line_ids += plot_pose(tcp_pose)

        position_sim, orientation_sim = robot.get_tcp_pose()
        orientation_sim = p.getMatrixFromQuaternion(orientation_sim)
        orientation_sim = np.array(orientation_sim).reshape(3, 3)

        position_dht = poses[-1, :3, -1]
        orientation_dht = poses[-1, :3, :3]

        if visualize:
            print(f"TCP pose")
            print(f"DHT\n\tPosition:{position_dht}\n\tOrientation:{orientation_dht}")
            print(f"Simulation\n\tPosition:{position_sim}\n\tOrientation:{orientation_sim}")
            print("#" * 30)

            input("Press Enter for new pose!")
        else:
            assert np.isclose(position_dht, position_sim,
                              atol=1e-2).all(), f"{position_dht}\n\n{position_sim}\n\n{np.isclose(position_dht, position_sim)}"
            assert np.isclose(orientation_dht, orientation_sim,
                              atol=1e-2).all(), f"{orientation_dht}\n\n{orientation_sim}\n\n{np.isclose(orientation_dht, orientation_sim)}"
        clear_lines(line_ids)
        line_ids = []


def perform_gradient_decent_ik(config=None, project=None, visualize=False, num_robots=2):
    with wandb.init(config=config, project=project, entity="bitter", mode=wandb_mode):
        config = wandb.config

        data = torch.load(config.data_file)

        dht_params = data["dht_params"]
        joint_limits = data["joint_limits"]

        dht_model = get_dht_model(dht_params, joint_limits)

        X = torch.rand(1000, len(joint_limits))
        y = dht_model(X).detach()

        weight_matrix_p = torch.zeros(len(data["dht_params"]), len(data["dht_params"]))
        weight_matrix_p[-1, -1] = 1
        weight_matrix_o = torch.zeros(len(data["dht_params"]), len(data["dht_params"]))
        weight_matrix_o[-1, -1] = 1

        loss_fn = KinematicChainLoss(weight_matrix_p, weight_matrix_o)

        angles = torch.nn.Parameter(torch.randn_like(X))
        optimizer = torch.optim.AdamW([angles], lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=50, min_lr=1e-6)

        angles_history = torch.empty((config.epochs, *angles.shape))

        for step in tqdm(range(config.epochs)):
            angles_history[step] = angles

            poses = dht_model(angles)

            loss, loss_p, loss_o = loss_fn(poses, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
                'loss': loss.item(),
                'loss_p': loss_p.item(),
                'loss_o': loss_o.item(),
                'lr': optimizer.param_groups[0]["lr"],
            }, step=step)

            scheduler.step(loss.item())

            if loss < 1e-3 or loss.isnan().any():
                break

        angles_history = angles_history[:step]

        torch.save({
            'angles': angles_history.cpu().detach(),
            'target': X.cpu().detach(),
        }, os.path.join(wandb.run.dir, "results.pt"))


def visualize_gradient_descent_ik(run_path):
    # run_path = "bitter/robot2robot_state_mapper/runs/2pgce3xw"

    file_config = wandb.restore("config.yaml", run_path, replace=True)

    with open(file_config.name, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)

    file_checkpoint = wandb.restore("results.pt", run_path, replace=True)
    results = torch.load(file_checkpoint.name, map_location="cpu")

    angles = results["angles"]
    target = results["target"]

    data_file = config["data_file"]["value"]

    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=70,
                                 cameraPitch=-27,
                                 cameraTargetPosition=(0, 0, 0)
                                 )

    master_robot = get_robot({
        "name": os.path.basename(data_file).split("_")[0],
        "scale": .1,
        "sim_time": .1,
    }, bullet_client=p)

    mimick_robot = get_robot({
        "name": os.path.basename(data_file).split("_")[0],
        "scale": .1,
        "sim_time": .1,
        "offset": (1, 1, 0)
    }, bullet_client=p)

    for angles_, target_ in zip(angles.transpose(0, 1), target):
        state_master = master_robot.state_space.sample()
        state_master["arm"]["joint_positions"] = target_
        master_robot.reset(state_master, force=True)

        state_mimick = mimick_robot.state_space.sample()

        for a in angles_:
            state_mimick["arm"]["joint_positions"] = a
            mimick_robot.reset(state_mimick, force=True)


def load_robot(data_file, bullet_client, **robot_kwargs):
    data_path = os.path.join(data_folder, data_file)
    data = torch.load(data_path)

    robot_name = data_file.split("_")[0]

    robot = get_robot({
        "name": robot_name,
        **data["robot_config"],
        **robot_kwargs
    }, bullet_client=bullet_client)

    return robot


if __name__ == '__main__':

    from models.dht import get_dht_model
    from utils.nn import Rescale, Sawtooth
    from torch.nn import Sequential

    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    data_file = "panda_10000_1000.pt"

    data_path = os.path.join(data_folder, data_file)
    data = torch.load(data_path)

    # p.connect(p.GUI)
    p.connect(p.DIRECT)

    robot_A = load_robot(data_file, p)
    robot_B = load_robot(data_file, p, offset=(1, 0, 0), deactivate_self_collision=True)

    dht_model = get_dht_model(data["dht_params"], data["joint_limits"])

    state_correction = Sequential(
        Rescale(-1, 1, data["joint_limits"][:, 0], data["joint_limits"][:, 1]),
        Sawtooth(-np.pi, np.pi, -np.pi, np.pi),
        Rescale(data["joint_limits"][:, 0], data["joint_limits"][:, 1], -1, 1),

    )

    for _ in range(10):
        state_A = robot_A.reset()
        state_B = robot_B.reset()

        state_A_arm = torch.FloatTensor([state_A["arm"]["joint_positions"]])
        state_B_arm = torch.FloatTensor([state_B["arm"]["joint_positions"]])
        state_B_arm.requires_grad = True

        poses_A = dht_model(state_A_arm).detach()

        # weight_matrix_p = torch.eye(poses_A.shape[1])
        # weight_matrix_o = torch.eye(poses_A.shape[1])

        weight_matrix_p = torch.zeros(poses_A.shape[1], poses_A.shape[1])
        weight_matrix_o = torch.zeros(poses_A.shape[1], poses_A.shape[1])

        weight_matrix_p[-1,-1] = 1
        weight_matrix_o[-1,-1] = 1

        optimizer = torch.optim.AdamW([state_B_arm], lr=1e-2)
        loss_fn_kcl = KinematicChainLoss(weight_matrix_p, weight_matrix_o)
        loss_fn = torch.nn.MSELoss()

        for i in range(1000):
            optimizer.zero_grad()

            poses_B = dht_model(state_B_arm)

            # loss, _, _ = loss_fn_kcl(poses_A, poses_B)
            loss = loss_fn(poses_A[:,-1,:], poses_B[:,-1,:])
            # loss = loss_fn(poses_A, poses_B)
            loss.backward()

            optimizer.step()

            state_B["arm"]["joint_positions"] = state_correction(state_B_arm)[0].cpu().detach().numpy()
            robot_B.reset(state_B, force=True)

            if loss.item() < 1e-4:
                break

        print(loss.item())

        time.sleep(.1)
        time.sleep(10)

    exit()
    visualize_gradient_descent_ik("bitter/dht_ik/runs/1b245dt9")

    for data_file in ["data/panda_10000_1000.pt"]:  # , "data/ur5_10000_1000.pt"]:
        verify_calculated_poses(data_file, 10)

        config = {
            "data_file": data_file,
            "lr": 1e-1,
            "epochs": 2_500,
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        perform_gradient_decent_ik(config, project="dht_ik", visualize=True)
