import torch
import pybullet as p
from environments.environments_robot_task.robots import get_robot
from environments import get_env
from utils.nn import Sawtooth, KinematicChainLoss, get_weight_matrices

p.connect(p.GUI)
# p.connect(p.DIRECT)
p.setRealTimeSimulation(True)

robot_A = get_robot({
    "name": "panda",
    "scale": .1,
    "sim_time": .1,
}, p)

robot_B = get_robot({
    "name": "ur5",
    "scale": .1,
    "sim_time": .1,
}, p)

angle2pi = Sawtooth(-torch.pi, torch.pi, -torch.pi, torch.pi)

for link_A in range(p.getNumJoints(robot_A.model_id)):
    p.setCollisionFilterGroupMask(robot_A.model_id, link_A, 0, 0)

for link_B in range(p.getNumJoints(robot_B.model_id)):
    p.setCollisionFilterGroupMask(robot_B.model_id, link_B, 0, 0)

link_positions_A = robot_A.forward_kinematics(torch.Tensor([robot_A.get_state()["arm"]["joint_positions"]]))[0, :, :3, -1]
link_positions_B = robot_B.forward_kinematics(torch.Tensor([robot_B.get_state()["arm"]["joint_positions"]]))[0, :, :3, -1]

weight_matrix_p, weight_matrix_o = get_weight_matrices(link_positions_A, link_positions_B, 100)

kcl_loss = KinematicChainLoss(weight_matrix_p, weight_matrix_o, reduction=False)

while True:
    state_A = robot_A.get_state()

    angles_A = robot_A.state2angle(torch.Tensor([state_A["arm"]["joint_positions"]]))

    poses_A = robot_A.forward_kinematics(angles_A)
    # print(angles_A)

    ik_solutions_B = robot_B.inverse_kinematics(poses_A[0, -1:])
    ik_solutions_B = ik_solutions_B[~ik_solutions_B.isnan().any(-1)]

    if len(ik_solutions_B):
        angles_B = angle2pi(ik_solutions_B)
        states_B = robot_B.angle2state(angles_B)
        poses_B = robot_B.forward_kinematics(angles_B)

        loss = kcl_loss(poses_A.repeat(len(poses_B), 1, 1, 1), poses_B)

        best_state_B = states_B[loss.argmin(0).squeeze()]
        # worst_state_B = states_B[loss.argmax(0).squeeze()]

        robot_B.reset({"arm": {"joint_positions": best_state_B}}, force=True)
    # else:
    #     robot_A.reset()