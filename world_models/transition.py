import os
import sys
from functools import cached_property
from pathlib import Path

import torch
from torch.nn import MSELoss

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import data_folder
from stage import LitStage
from utils.nn import create_network


class TransitionModel(LitStage):
    def __init__(self, config):
        super(TransitionModel, self).__init__(config)

    @cached_property
    def loss_function(self):
        return MSELoss()

    @cached_property
    def transition_model(self):
        data_path = os.path.join(data_folder, self.config[self.__class__.__name__]["data"])
        data = torch.load(data_path)

        transition_model = create_network(
            in_dim=data["trajectories_states_train"].shape[-1]
            + data["trajectories_actions_train"].shape[-1],
            out_dim=data["trajectories_states_train"].shape[-1],
            **self.config[self.__class__.__name__]["model"],
        )
        return transition_model

    def configure_optimizers(self):
        optimizer_model = torch.optim.AdamW(
            self.transition_model.parameters(),
            lr=self.config[self.__class__.__name__]["train"].get("lr", 3e-4),
        )
        return optimizer_model

    def loss(self, batch):
        trajectories_states, trajectories_actions = batch

        states = trajectories_states[:, :-1].reshape(-1, trajectories_states.shape[-1])
        next_states = trajectories_states[:, :-1].reshape(
            -1, trajectories_states.shape[-1]
        )

        actions = trajectories_actions.reshape(-1, trajectories_actions.shape[-1])

        states_actions = torch.concat(tensors=(states, actions), dim=-1)
        self.transition_model.to(states_actions)
        next_states_predicted = self.transition_model(states_actions)
        loss_transition_model = self.loss_function(next_states_predicted, next_states)
        return loss_transition_model

