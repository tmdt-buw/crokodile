import time

import wandb
import os
import torch
from models.state_mapper import LitStateMapper
import pybullet as p
from environments.environments_robot_task.robots import get_robot
import tempfile

# reference can be retrieved in artifacts panel
# "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
# checkpoint_reference = "robot2robot/PITL/model-3llceeay:best"
domain_mapper_artifact = "robot2robot/PITL/model-hamj4vvw:best"

with tempfile.TemporaryDirectory() as dir:
    api = wandb.Api()
    artifact_dir = api.artifact(domain_mapper_artifact).download(dir)
    domain_mapper = LitStateMapper.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))

data_file_A = domain_mapper.hparams.data_file_A
data_file_B = domain_mapper.hparams.data_file_B

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

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

p.connect(p.GUI)
# p.connect(p.DIRECT)

robot_A = load_robot(data_file_A, p)
robot_B = load_robot(data_file_B, p)

for link_A in range(p.getNumJoints(robot_A.model_id)):
    p.setCollisionFilterGroupMask(robot_A.model_id, link_A, 0, 0)

for link_B in range(p.getNumJoints(robot_B.model_id)):
    p.setCollisionFilterGroupMask(robot_B.model_id, link_B, 0, 0)

while True:
    print("State")
    state_A = robot_A.reset()

    state_A_mappable = torch.tensor([state_A["arm"]["joint_positions"]], dtype=torch.float)
    state_B_mapped = domain_mapper.state_mapper_AB(state_A_mappable)[0]

    state_B = robot_B.state_space.sample()
    state_B["arm"]["joint_positions"] = state_B_mapped.cpu().detach().numpy()

    robot_B.reset(state_B, force=True)

    for _ in range(3):
        break
        try:
            print("Action")

            action_A = robot_A.action_space.sample()
            action_B = robot_B.action_space.sample()

            action_A_mappable = torch.tensor([action_A["arm"]], dtype=torch.float)
            action_B_mapped = domain_mapper.action_mapper_AB(action_A_mappable)[0]

            action_B["arm"] = action_B_mapped.cpu().detach().numpy()

            state_A = robot_A.step(action_A)
            robot_B.step(action_B)

        except Exception as e:
            print("Except", e)
            break

    time.sleep(3)
