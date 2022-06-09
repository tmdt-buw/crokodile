import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tensorboard import program
from gym import spaces
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from robot2robot_transfer.dht import DHT_Model
from robot2robot_transfer.utils.nn import NeuralNetwork

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DynamicsModel(nn.Module):
    def __init__(self, state_space, action_space, transition):
        super(DynamicsModel, self).__init__()

        self.state_space = state_space
        self.action_space = action_space

        self.state_space_low = torch.tensor(self.state_space.low).to(device)
        self.state_space_high = torch.tensor(self.state_space.high).to(device)

        self.transition = transition

    def forward(self, state, action):
        next_state = self.transition(state, action)
        # next_state = next_state.clip(self.state_space.low, self.state_space.high)

        next_state = torch.max(torch.min(next_state, self.state_space_high),
                               self.state_space_low)

        return next_state


class SpaceTranslator(nn.Module):
    def __init__(self, space1, space2):
        super(SpaceTranslator, self).__init__()

        space_shape_1 = space1.shape[0]
        space_shape_2 = space2.shape[0]

        translator_struct = [('linear', space_shape_1 + space_shape_2),
                             ('relu', None),  # ('dropout', 0.2),
                             ('linear', space_shape_1 + space_shape_2),
                             ('relu', None),  # ('dropout', 0.2),
                             ('linear', space_shape_2)]

        self.translator = NeuralNetwork(space_shape_1, translator_struct)

    def forward(self, sample):
        return self.translator(sample)


class Pose_Extractor(nn.Module):
    def __init__(self, dht_params, joint_limits):
        super(Pose_Extractor, self).__init__()

        self.dht_model = DHT_Model(dht_params)
        self.joint_limits = torch.nn.Parameter(torch.tensor(joint_limits))

    def forward(self, x):
        x = x[:, :len(self.joint_limits)] # [-1, 1]

        x = x + 1  # [0, 2]
        x = x / 2  # [0, 1]
        x = x * (self.joint_limits[:, 1] - self.joint_limits[:, 0]) # [0, ol - ul]
        x = x + self.joint_limits[:, 0] # [ul, ol]

        return self.dht_model(x)


class SpaceTranslatorMultivariate(nn.Module):
    def __init__(self, space1, space2):
        super(SpaceTranslatorMultivariate, self).__init__()

        space_shape_1 = space1.shape[0]
        space_shape_2 = space2.shape[0]

        translator_struct = [('linear', space_shape_1 + space_shape_2),
                             ('relu', None),  # ('dropout', 0.2),
                             ('linear', space_shape_1 + space_shape_2),
                             ('relu', None),  # ('dropout', 0.2),
                             ('linear', space_shape_2)]

        self.translator = NeuralNetwork(space_shape_1, translator_struct)

    def forward(self, sample, deterministic=True):

        x = self.translator(sample)

        mean, log_std = torch.split(x, x.shape[1] // 2, dim=1)

        if deterministic:
            out = torch.tanh(mean)
            log_prob = torch.zeros(log_std.shape[0]).unsqueeze_(-1)
        else:
            std = log_std.exp()

            normal = MultivariateNormal(mean, torch.diag_embed(std.pow(2)))
            out_base = normal.rsample()

            log_prob = normal.log_prob(out_base)
            log_prob.unsqueeze_(-1)

            out = torch.tanh(out_base)

            out_bound_compensation = torch.log(
                1. - out.pow(2) + np.finfo(float).eps).sum(dim=1,
                                                           keepdim=True)

            log_prob.sub_(out_bound_compensation)

        return out, log_prob


def sample_space(space, num_samples=1):
    samples = torch.tensor([space.sample() for _ in range(num_samples)], device=device)

    return samples


def loss_points_of_interest(points_of_interestA, points_of_interestB):
    tcpA = points_of_interestA[:, -1, :3, :]
    tcpB = points_of_interestB[:, -1, :3, :]

    loss_position = torch.nn.functional.mse_loss(tcpA[:, :, 3], tcpB[:, :, 3])
    # loss_orientation = torch.nn.functional.mse_loss(tcpA[:, :, :3], tcpB[:, :, :3])

    # loss = .7 * loss_position + .3 * loss_orientation

    return loss_position


if __name__ == "__main__":
    use_case = "2d_arm"  # ["2d_arm", "ur5_panda"]

    experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_base_dir = "results"
    results_dir = os.path.join(experiment_base_dir, use_case, experiment_name)
    os.makedirs(results_dir, True)

    epochs = 5_000
    samples = 128
    lr = 1e-3
    # action_embedding_size = 10

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

        action_embedding_size = 10

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

    action_spaceA = environmentA.action_space
    action_spaceB = environmentB.action_space

    poses_extractorA = Pose_Extractor(dht_paramsA, joint_limitsA).to(device)
    poses_extractorB = Pose_Extractor(dht_paramsB, joint_limitsB).to(device)

    # transitionA = environmentA.transition
    # transitionB = environmentB.transition
    #
    # dynamics_model_A = DynamicsModel(state_spaceA, action_spaceA,
    #                                  transitionA).to(device)
    # dynamics_model_B = DynamicsModel(state_spaceB, action_spaceB,
    #                                  transitionB).to(device)

    # action_embedding = spaces.Box(-1, 1, (action_embedding_size,))

    # translator_action_A0 = SpaceTranslator(environmentA.action_space, action_embedding).to(device)
    # translator_action_0A = SpaceTranslator(action_embedding, environmentA.action_space).to(device)
    # translator_action_B0 = SpaceTranslator(environmentB.action_space, action_embedding).to(device)
    # translator_action_0B = SpaceTranslator(action_embedding, environmentB.action_space).to(device)

    translator_state_AB = SpaceTranslator(state_spaceA, state_spaceB).to(device)
    translator_state_BA = SpaceTranslator(state_spaceB, state_spaceA).to(device)

    # optimizer_action_0A = torch.optim.Adam(translator_action_0A.parameters(), lr=lr)
    # optimizer_action_A0 = torch.optim.Adam(translator_action_A0.parameters(), lr=lr)
    # optimizer_action_0B = torch.optim.Adam(translator_action_0B.parameters(), lr=lr)
    # optimizer_action_B0 = torch.optim.Adam(translator_action_B0.parameters(), lr=lr)
    optimizer_state_AB = torch.optim.Adam(translator_state_AB.parameters(), lr=lr)
    optimizer_state_BA = torch.optim.Adam(translator_state_BA.parameters(), lr=lr)

    loss_function = torch.nn.MSELoss()

    writer = SummaryWriter(results_dir)

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', results_dir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    for epoch in tqdm(range(epochs)):
        # mode_action_sampling = np.random.choice(["action0", "actionA", "actionB"])
        mode_state_sampling = np.random.choice(["stateA", "stateB"])

        # translator_action_0A.zero_grad()
        # translator_action_A0.zero_grad()
        # translator_action_0B.zero_grad()
        # translator_action_B0.zero_grad()
        translator_state_AB.zero_grad()
        translator_state_BA.zero_grad()

        # if mode_action_sampling == "action0":
        #     actions0 = sample_space(action_embedding, samples)
        #
        #     actionsA = translator_action_0A(actions0)
        #     actionsB = translator_action_0B(actions0)
        #
        #     # cycle consistency
        #     loss = loss_function(translator_action_A0(actionsA), actions0)
        #     loss.backward(retain_graph=True)
        #     writer.add_scalar('Loss Skills 0A0', loss.item(), epoch)
        #
        #     loss = loss_function(translator_action_B0(actionsB), actions0)
        #     loss.backward(retain_graph=True)
        #     writer.add_scalar('Loss Skills 0B0', loss.item(), epoch)
        #
        # elif mode_action_sampling == "actionA":
        #     actionsA = sample_space(environmentA.action_space, samples)
        #
        #     actions0 = translator_action_A0(actionsA)
        #     actionsB = translator_action_0B(actions0)
        #
        #     # cycle consistency
        #     loss = loss_function(translator_action_0A(actions0), actionsA)
        #     loss.backward(retain_graph=True)
        #     writer.add_scalar('Loss Skills A0A', loss.item(), epoch)
        #
        # elif mode_action_sampling == "actionB":
        #     actionsB = sample_space(environmentB.action_space, samples)
        #
        #     actions0 = translator_action_B0(actionsB)
        #     actionsA = translator_action_0A(actions0)
        #
        #     # cycle consistency
        #     loss = loss_function(translator_action_0B(actions0), actionsB)
        #     loss.backward(retain_graph=True)
        #     writer.add_scalar('Loss Skills B0B', loss.item(), epoch)
        # else:
        #     raise NotImplementedError()

        if mode_state_sampling == "stateA":
            statesA = sample_space(state_spaceA, samples)

            statesB = translator_state_AB(statesA)

            # cycle consistency
            loss = loss_function(translator_state_BA(statesB), statesA)
            loss.backward(retain_graph=True)
            writer.add_scalar('Loss States ABA', loss.item(), epoch)

        elif mode_state_sampling == "stateB":
            statesB = sample_space(state_spaceB, samples)

            statesA = translator_state_BA(statesB)

            # cycle consistency
            loss = loss_function(translator_state_AB(statesA), statesB)
            loss.backward(retain_graph=True)
            writer.add_scalar('Loss States BAB', loss.item(), epoch)
        else:
            raise NotImplementedError()

        # cycle consistency next state
        # next_statesA = dynamics_model_A(statesA, actionsA)
        # next_statesB = dynamics_model_B(statesB, actionsB)

        # loss = loss_function(translator_state_BA(next_statesB), next_statesA)
        # loss.backward(retain_graph=True)
        # writer.add_scalar('Loss Next States ABA', loss.item(), epoch)
        #
        # loss = loss_function(translator_state_AB(next_statesA), next_statesB)
        # loss.backward(retain_graph=True)
        # writer.add_scalar('Loss Next States BAB', loss.item(), epoch)

        # state correspondance
        points_of_interest_statesA = poses_extractorA(statesA)
        points_of_interest_statesB = poses_extractorB(statesB)
        loss = loss_points_of_interest(points_of_interest_statesA, points_of_interest_statesB)
        loss.backward(retain_graph=True)
        writer.add_scalar('Loss States Correspondance', loss.item(), epoch)

        # points_of_interest_next_statesA = environmentA.points_of_interest(next_statesA)
        # points_of_interest_next_statesB = environmentB.points_of_interest(next_statesB)
        # loss = loss_points_of_interest(points_of_interest_next_statesA, points_of_interest_next_statesB)
        # loss.backward(retain_graph=True)
        # writer.add_scalar('Loss Next States Correspondance', loss.item(), epoch)

        writer.flush()

        # optimizer_action_0A.step()
        # optimizer_action_A0.step()
        # optimizer_action_0B.step()
        # optimizer_action_B0.step()

        optimizer_state_AB.step()
        optimizer_state_BA.step()

        if epoch % 500 == 0:
            checkpoint = {
                "configA": configA,
                "configB": configB,

                # "action_embedding_size": action_embedding_size,

                # "translator_action_0A": translator_action_0A.state_dict(),
                # "translator_action_A0": translator_action_A0.state_dict(),
                # "translator_action_0B": translator_action_0B.state_dict(),
                # "translator_action_B0": translator_action_B0.state_dict(),
                "translator_state_AB": translator_state_AB.state_dict(),
                "translator_state_BA": translator_state_BA.state_dict(),

                # "optimizer_action_0A": optimizer_action_0A.state_dict(),
                # "optimizer_action_A0": optimizer_action_A0.state_dict(),
                # "optimizer_action_0B": optimizer_action_0B.state_dict(),
                # "optimizer_action_B0": optimizer_action_B0.state_dict(),

                "optimizer_state_AB": optimizer_state_AB.state_dict(),
                "optimizer_state_BA": optimizer_state_BA.state_dict(),
            }
            torch.save(checkpoint, os.path.join(os.path.join(results_dir, f"models_{epoch}.pt")))

    checkpoint = {
        "configA": configA,
        "configB": configB,

        # "action_embedding_size": action_embedding_size,

        # "translator_action_0A": translator_action_0A.state_dict(),
        # "translator_action_A0": translator_action_A0.state_dict(),
        # "translator_action_0B": translator_action_0B.state_dict(),
        # "translator_action_B0": translator_action_B0.state_dict(),
        "translator_state_AB": translator_state_AB.state_dict(),
        "translator_state_BA": translator_state_BA.state_dict(),

        # "optimizer_action_0A": optimizer_action_0A.state_dict(),
        # "optimizer_action_A0": optimizer_action_A0.state_dict(),
        # "optimizer_action_0B": optimizer_action_0B.state_dict(),
        # "optimizer_action_B0": optimizer_action_B0.state_dict(),

        "optimizer_state_AB": optimizer_state_AB.state_dict(),
        "optimizer_state_BA": optimizer_state_BA.state_dict(),
    }
    torch.save(checkpoint, os.path.join(os.path.join(results_dir, "models.pt")))
