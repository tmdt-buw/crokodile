import os
import sys
from pathlib import Path

import torch
from torch.nn import MSELoss

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nn import create_network
from lit_models.lit_model import LitModel
from config import data_folder


class TransitionModel(LitModel):
    def __init__(self, config):
        # self.model = LitTransitionModel(config)
        super(TransitionModel, self).__init__(config)


    def generate(self):
        self.loss_function = MSELoss()

        super(TransitionModel, self).generate()

    def get_model(self):
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
            self.model.parameters(),
            lr=self.config[self.__class__.__name__]["train"].get("lr", 3e-4),
        )
        return optimizer_model

    def train_dataloader(self):
        return self.get_dataloader(self.config[self.__class__.__name__]["data"], "train")

    def val_dataloader(self):
        return self.get_dataloader(self.config[self.__class__.__name__]["data"], "test", False)

    def forward(self, x):
        next_states = self.model(x)
        return next_states

    def loss(self, batch):
        trajectories_states, trajectories_actions = batch

        states = trajectories_states[:, :-1].reshape(-1, trajectories_states.shape[-1])
        next_states = trajectories_states[:, :-1].reshape(
            -1, trajectories_states.shape[-1]
        )

        actions = trajectories_actions.reshape(-1, trajectories_actions.shape[-1])

        states_actions = torch.concat(tensors=(states, actions), dim=-1)
        next_states_predicted = self.model(states_actions)
        loss_transition_model = self.loss_function(next_states_predicted, next_states)
        return loss_transition_model

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log(
            f"train_loss_{self.__class__.__name__}",
            loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log(
            f"validation_loss_{self.__class__.__name__}",
            loss,
            on_step=False,
            on_epoch=True,
        )
        return loss
