import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.distributions import MultivariateNormal

sys.path.insert(0, str(Path('.').absolute().parent))

from utils.nn import NeuralNetwork

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from environments.NLinkArm import NLinkArm


class DynamicsModel(nn.Module):
    def __init__(self, state_space, skill_space, transition):
        super(DynamicsModel, self).__init__()

        self.state_space = state_space
        self.skill_space = skill_space

        self.state_space_low = torch.tensor(self.state_space.low).to(device)
        self.state_space_high = torch.tensor(self.state_space.high).to(device)

        self.transition = transition

    def forward(self, state, skill):
        next_state = self.transition(state, skill)
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
    samples = torch.tensor([space.sample() for _ in range(num_samples)]).to(
        device)

    return samples


if __name__ == "__main__":
    epochs = 50_000
    samples = 128
    lr = 1e-3

    configA = {"link_lengths": [.4, .4, .4],
               "scales": .2 * np.pi}
    configB = {"link_lengths": [.3, .3, .3, .3],
               "scales": .3 * np.pi}

    skill_embedding_size = 10

    armA = NLinkArm(**configA)
    armB = NLinkArm(**configB)

    state_spaceA = armA.observation_space
    skill_spaceA = armA.action_space

    state_spaceB = armB.observation_space
    skill_spaceB = armB.action_space

    transitionA = armA.transition
    transitionB = armB.transition

    dynamics_model_A = DynamicsModel(state_spaceA, skill_spaceA,
                                     transitionA).to(device)
    dynamics_model_B = DynamicsModel(state_spaceB, skill_spaceB,
                                     transitionB).to(device)

    skill_embedding = spaces.Box(-1, 1, (skill_embedding_size,))

    experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_base_dir = "results/"

    translator_skill_0A = SpaceTranslator(skill_embedding,
                                          dynamics_model_A.skill_space).to(
        device)
    translator_skill_A0 = SpaceTranslator(dynamics_model_A.skill_space,
                                          skill_embedding).to(device)
    translator_skill_0B = SpaceTranslator(skill_embedding,
                                          dynamics_model_B.skill_space).to(
        device)
    translator_skill_B0 = SpaceTranslator(dynamics_model_B.skill_space,
                                          skill_embedding).to(device)

    translator_state_AB = SpaceTranslator(dynamics_model_A.state_space,
                                          dynamics_model_B.state_space).to(
        device)
    translator_state_BA = SpaceTranslator(dynamics_model_B.state_space,
                                          dynamics_model_A.state_space).to(
        device)

    writer = SummaryWriter(os.path.join(experiment_base_dir, experiment_name))

    optimizer_skill_0A = torch.optim.Adam(translator_skill_0A.parameters(),
                                          lr=lr)
    optimizer_skill_A0 = torch.optim.Adam(translator_skill_A0.parameters(),
                                          lr=lr)
    optimizer_skill_0B = torch.optim.Adam(translator_skill_0B.parameters(),
                                          lr=lr)
    optimizer_skill_B0 = torch.optim.Adam(translator_skill_B0.parameters(),
                                          lr=lr)
    optimizer_state_AB = torch.optim.Adam(translator_state_AB.parameters(),
                                          lr=lr)
    optimizer_state_BA = torch.optim.Adam(translator_state_BA.parameters(),
                                          lr=lr)

    loss_function = torch.nn.MSELoss()

    for epoch in tqdm(range(epochs)):
        mode_skill_sampling = np.random.choice(["skill0", "skillA", "skillB"])
        mode_state_sampling = np.random.choice(["stateA", "stateB"])

        translator_skill_0A.zero_grad()
        translator_skill_A0.zero_grad()
        translator_skill_0B.zero_grad()
        translator_skill_B0.zero_grad()
        translator_state_AB.zero_grad()
        translator_state_BA.zero_grad()

        if mode_skill_sampling == "skill0":
            skills0 = sample_space(skill_embedding, samples)

            skillsA = translator_skill_0A(skills0)
            skillsB = translator_skill_0B(skills0)

            # cycle consistency
            loss = loss_function(translator_skill_A0(skillsA), skills0)
            loss.backward(retain_graph=True)
            writer.add_scalar('Loss Skills 0A0', loss.item(), epoch)

            loss = loss_function(translator_skill_B0(skillsB), skills0)
            loss.backward(retain_graph=True)
            writer.add_scalar('Loss Skills 0B0', loss.item(), epoch)

        elif mode_skill_sampling == "skillA":
            skillsA = sample_space(dynamics_model_A.skill_space, samples)

            skills0 = translator_skill_A0(skillsA)
            skillsB = translator_skill_0B(skills0)

            # cycle consistency
            loss = loss_function(translator_skill_0A(skills0), skillsA)
            loss.backward(retain_graph=True)
            writer.add_scalar('Loss Skills A0A', loss.item(), epoch)

        elif mode_skill_sampling == "skillB":
            skillsB = sample_space(dynamics_model_B.skill_space, samples)

            skills0 = translator_skill_B0(skillsB)
            skillsA = translator_skill_0A(skills0)

            # cycle consistency
            loss = loss_function(translator_skill_0B(skills0), skillsB)
            loss.backward(retain_graph=True)
            writer.add_scalar('Loss Skills B0B', loss.item(), epoch)
        else:
            raise NotImplementedError()

        if mode_state_sampling == "stateA":
            statesA = sample_space(dynamics_model_A.state_space, samples)

            statesB = translator_state_AB(statesA)

            # cycle consistency
            loss = loss_function(translator_state_BA(statesB), statesA)
            loss.backward(retain_graph=True)
            writer.add_scalar('Loss States ABA', loss.item(), epoch)

        elif mode_state_sampling == "stateB":
            statesB = sample_space(dynamics_model_B.state_space, samples)

            statesA = translator_state_BA(statesB)

            # cycle consistency
            loss = loss_function(translator_state_AB(statesA), statesB)
            loss.backward(retain_graph=True)
            writer.add_scalar('Loss States BAB', loss.item(), epoch)
        else:
            raise NotImplementedError()

        # cycle consistency next state
        next_statesA = dynamics_model_A(statesA, skillsA)
        next_statesB = dynamics_model_B(statesB, skillsB)

        loss = loss_function(translator_state_BA(next_statesB), next_statesA)
        loss.backward(retain_graph=True)
        writer.add_scalar('Loss Next States ABA', loss.item(), epoch)

        loss = loss_function(translator_state_AB(next_statesA), next_statesB)
        loss.backward(retain_graph=True)
        writer.add_scalar('Loss Next States BAB', loss.item(), epoch)

        # state correspondance
        loss = 10. * loss_function(statesA[:, -4:-2], statesB[:, -4:-2]) #+ \
               # .0 * loss_function(statesA[:, -2:], statesB[:, -2:])
        loss.backward(retain_graph=True)
        writer.add_scalar('Loss States Correspondance', loss.item(), epoch)

        loss = 10. * loss_function(next_statesA[:, -4:-2],
                                  next_statesB[:, -4:-2]) #+ \
               # .0 * loss_function(next_statesA[:, -2:], next_statesB[:, -2:])
        loss.backward(retain_graph=True)
        writer.add_scalar('Loss Next States Correspondance', loss.item(),
                          epoch)

        writer.flush()

        optimizer_skill_0A.step()
        optimizer_skill_A0.step()
        optimizer_skill_0B.step()
        optimizer_skill_B0.step()

        optimizer_state_AB.step()
        optimizer_state_BA.step()

        if epoch % 1_000 == 0:
            checkpoint = {
                "configA": configA,
                "configB": configB,

                "skill_embedding_size": skill_embedding_size,

                "translator_skill_0A": translator_skill_0A.state_dict(),
                "translator_skill_A0": translator_skill_A0.state_dict(),
                "translator_skill_0B": translator_skill_0B.state_dict(),
                "translator_skill_B0": translator_skill_B0.state_dict(),
                "translator_state_AB": translator_state_AB.state_dict(),
                "translator_state_BA": translator_state_BA.state_dict(),

                "optimizer_skill_0A": optimizer_skill_0A.state_dict(),
                "optimizer_skill_A0": optimizer_skill_A0.state_dict(),
                "optimizer_skill_0B": optimizer_skill_0B.state_dict(),
                "optimizer_skill_B0": optimizer_skill_B0.state_dict(),

                "optimizer_state_AB": optimizer_state_AB.state_dict(),
                "optimizer_state_BA": optimizer_state_BA.state_dict(),
            }
            torch.save(checkpoint, os.path.join(
                os.path.join(experiment_base_dir, experiment_name,
                             "models.pt")))

    checkpoint = {
        "configA": configA,
        "configB": configB,

        "skill_embedding_size": skill_embedding_size,

        "translator_skill_0A": translator_skill_0A.state_dict(),
        "translator_skill_A0": translator_skill_A0.state_dict(),
        "translator_skill_0B": translator_skill_0B.state_dict(),
        "translator_skill_B0": translator_skill_B0.state_dict(),
        "translator_state_AB": translator_state_AB.state_dict(),
        "translator_state_BA": translator_state_BA.state_dict(),

        "optimizer_skill_0A": optimizer_skill_0A.state_dict(),
        "optimizer_skill_A0": optimizer_skill_A0.state_dict(),
        "optimizer_skill_0B": optimizer_skill_0B.state_dict(),
        "optimizer_skill_B0": optimizer_skill_B0.state_dict(),

        "optimizer_state_AB": optimizer_state_AB.state_dict(),
        "optimizer_state_BA": optimizer_state_BA.state_dict(),
    }
    torch.save(checkpoint, os.path.join(os.path.join(experiment_base_dir, experiment_name, "models.pt")))
