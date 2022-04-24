import torch
import os
import numpy as np
from robot2robot_correspondance import SpaceTranslator, Pose_Extractor

use_case = "2d_arm"  # ["2d_arm", "ur5_panda"]
experiment_base_dir = "results"
experiment_name = "20220126-154431"
results_dir = os.path.join(experiment_base_dir, use_case, experiment_name)

model_files = [f for f in os.listdir(results_dir) if f.startswith("models")]

if "models.pt" in model_files:
    model_file = "models.pt"
else:
    model_file = sorted(model_files, key=lambda x: int(x[7:-3]))[-1]

print("Using model", model_file)

checkpoint = torch.load(os.path.join(results_dir, model_file))

if use_case == "ur5_panda":
    configA = {
        "name": "ur5",
        "scale": .1,
        "sim_time": .1
    }

    configB = {
        "name": "panda",
        "scale": .1,
        "sim_time": .1,
        "offset": (1, 1, 0)
    }

    from karolos.environments.environments_robot_task.robots import get_robot

    environmentA = get_robot(configA)
    environmentB = get_robot(configB)

    state_spaceA = environmentA.state_space["joint_positions"]
    state_spaceB = environmentB.state_space["joint_positions"]

    dht_paramsA = environmentA.dht_params
    dht_paramsB = environmentB.dht_params

    joint_limitsA = [joint.limits for joint in environmentA.joints_arm.values()]
    joint_limitsB = [joint.limits for joint in environmentB.joints_arm.values()]
elif use_case == "2d_arm":
    from robot2robot_transfer.environments.NLinkArm import NLinkArm

    configA = {"link_lengths": [.4, .4, .4],
               "scales": .2 * np.pi}
    configB = {"link_lengths": [.3, .3, .3, .3],
               "scales": .3 * np.pi}

    environmentA = NLinkArm(**configA)
    environmentB = NLinkArm(**configB)

    state_spaceA = environmentA.state_space
    state_spaceB = environmentB.state_space

    dht_paramsA = environmentA.dht_params
    dht_paramsB = environmentB.dht_params

    joint_limitsA = environmentA.joint_limits
    joint_limitsB = environmentB.joint_limits
else:
    raise ValueError("Unknown use case")

translator_state_AB = SpaceTranslator(state_spaceA, state_spaceB)
translator_state_BA = SpaceTranslator(state_spaceB, state_spaceA)

translator_state_AB.load_state_dict(checkpoint["translator_state_AB"])
translator_state_BA.load_state_dict(checkpoint["translator_state_BA"])

poses_extractorA = Pose_Extractor(dht_paramsA, joint_limitsA)
poses_extractorB = Pose_Extractor(dht_paramsB, joint_limitsB)

import matplotlib.pyplot as plt

examples = 1

# fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(3*5, 3*5))
fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(examples*5, 2*5))

for ax in axes.flatten():
    stateA = environmentA.reset()
    statesA = torch.tensor([stateA], dtype=torch.float32)
    statesB = translator_state_AB(statesA)

    environmentB.reset(statesB[0].detach())
    environmentA.plot(ax, "C0")
    environmentB.plot(ax, "C1")

    ax.set(adjustable='box', aspect='equal')
    # ax.set_xlim(-1.5,1)
    # ax.set_ylim(-1,1)

    import matplotlib.pyplot as plt

    posesA = poses_extractorA(statesA)
    tcpA = posesA[0, -1, :3, 3].cpu().detach()

    posesB = poses_extractorB(statesB)
    tcpB = posesB[0, -1, :3, 3].cpu().detach()

    print(sum((tcpA - tcpB) ** 2))

    ax.scatter(tcpA[0], tcpA[1], c="r", marker="x")
    ax.scatter(tcpB[0], tcpB[1], c="r", marker="x")

plt.show()
