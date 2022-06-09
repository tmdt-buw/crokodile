import os
from datetime import datetime

import torch
import torch.nn as nn
from gym import spaces
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.nn import NeuralNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def sample_space(space, num_samples=1):
    samples = torch.tensor([space.sample() for _ in range(num_samples)]).to(
        device)

    return samples


if __name__ == '__main__':
    experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_base_dir = "results/"

    writer = SummaryWriter(os.path.join(experiment_base_dir, experiment_name))

    epochs = 50_000
    samples = 256
    lr = 1e-3

    spaceA = spaces.Box(-1., 1., (2,))
    spaceB = spaces.Box(-1., 1., (2,))
    spaceC = spaces.Box(-1., 1., (2,))

    space0 = spaces.Box(-1., 1., (2,))

    distance0 = lambda a, b: torch.linalg.norm(a - b)

    distanceAB = lambda a, b: torch.linalg.norm(a - b)
    # distanceBC = lambda b, c: torch.linalg.norm(b - 2*c)

    translator_A0 = SpaceTranslator(spaceA, space0).to(device)
    translator_0A = SpaceTranslator(space0, spaceA).to(device)
    translator_B0 = SpaceTranslator(spaceB, space0).to(device)
    translator_0B = SpaceTranslator(space0, spaceB).to(device)

    optimizer_state_A0 = torch.optim.Adam(translator_A0.parameters(), lr=lr)
    optimizer_state_0A = torch.optim.Adam(translator_0A.parameters(), lr=lr)
    optimizer_state_B0 = torch.optim.Adam(translator_B0.parameters(), lr=lr)
    optimizer_state_0B = torch.optim.Adam(translator_0B.parameters(), lr=lr)

    loss_function = torch.nn.MSELoss()

    for epoch in tqdm(range(epochs)):
        translator_A0.zero_grad()
        translator_0A.zero_grad()
        translator_B0.zero_grad()
        translator_0B.zero_grad()
        
        statesA = sample_space(spaceA, samples)
        statesB = sample_space(spaceB, samples)
        
        statesA0 = translator_A0(statesA)
        statesB0 = translator_B0(statesB)
        
        statesA0A = translator_0A(statesA0)
        statesA0B = translator_0B(statesA0)
        statesB0A = translator_0A(statesB0)
        statesB0B = translator_0B(statesB0)
        
        # autoencoder loss
        loss = loss_function(statesA0A, statesA)
        loss.backward(retain_graph=True)
        writer.add_scalar('Loss States A0A', loss.item(), epoch)
        
        loss = loss_function(statesB0B, statesB)
        loss.backward(retain_graph=True)
        writer.add_scalar('Loss States B0B', loss.item(), epoch)
        
        # state correspondance loss
        loss = loss_function(distanceAB(statesA0B, statesA), torch.zeros(samples, device=device))
        loss.backward(retain_graph=True)
        writer.add_scalar('Loss States Correspondance A0B', loss.item(), epoch)

        loss = loss_function(distanceAB(statesB0A, statesB), torch.zeros(samples, device=device))
        loss.backward(retain_graph=True)
        writer.add_scalar('Loss States Correspondance B0A', loss.item(), epoch)

        # embedding distance loss
        loss = loss_function(distance0(statesA0, statesB0),
                             distanceAB(statesA, statesB))
        loss.backward(retain_graph=True)
        writer.add_scalar('Loss Embedding Distance', loss.item(), epoch)

        writer.flush()

        optimizer_state_A0.step()
        optimizer_state_0A.step()
        optimizer_state_B0.step()
        optimizer_state_0B.step()