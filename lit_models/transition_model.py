import os
import sys
from pathlib import Path

import torch
from torch.nn import MSELoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nn import create_network
from lit_model import LitModel, LitTrainer
data_folder = "data"

class LitTransitionModel(LitModel):
    def __init__(
        self,
        config
    ):
        super(LitTransitionModel, self).__init__(config)
        self.loss_function = MSELoss()

    def get_model(self):
        data_path = os.path.join(data_folder, self.lit_config["data"])
        data = torch.load(data_path)

        transition_model = create_network(
            in_dim=data["trajectories_states_train"].shape[-1]
            + data["trajectories_actions_train"].shape[-1],
            out_dim=data["trajectories_states_train"].shape[-1],
            **self.lit_config["model"],
        )
        return transition_model

    def configure_optimizers(self):
        optimizer_model = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lit_config["train"].get("lr", 3e-4),
        )
        return optimizer_model

    def train_dataloader(self):
        return self.get_dataloader(self.lit_config["data"], "train")

    def val_dataloader(self):
        return self.get_dataloader(self.lit_config["data"], "test", False)

    def forward(self, x):
        with torch.no_grad():
            next_states = self.transition_model(x)
        return next_states

    def loss(self, batch):
        trajectories_states, trajectories_actions = batch

        states = trajectories_states[:, :-1].reshape(-1, trajectories_states.shape[-1])
        next_states = trajectories_states[:, :-1].reshape(
            -1, trajectories_states.shape[-1]
        )

        actions = trajectories_actions.reshape(-1, trajectories_actions.shape[-1])

        states_actions = torch.concat(tensors=(states, actions), dim=-1)
        next_states_predicted = self.transition_model(states_actions)
        loss_transition_model = self.loss_function(next_states_predicted, next_states)
        return loss_transition_model

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log(
            "train_loss_transition_model" + self.lit_config["log_suffix"],
            loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log(
            "validation_loss_transition_model" + self.lit_config["log_suffix"],
            loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

class TransitionModel(LitTrainer):
    def __init__(self, config):
        self.model_cls = config["TransitionModel"]["model_cls"]
        self.model_config = config["TransitionModel"]

        super(TransitionModel, self).__init__(config)

    def generate(self):
        super(TransitionModel, self).generate()
        self.train()

    @classmethod
    def get_relevant_config(cls, config):
        return super(TransitionModel, cls).get_relevant_config(config)



def main():
    data_file_A = "panda_5_200_40.pt"
    data_file_B = "ur5_5_200_40.pt"

    config_A = {
        "TransitionModel": {
            "model_cls": "transition_model",
            "data": data_file_A,
            "log_suffix": "_A",
            "model": {
                    "network_width": 256,
                    "network_depth": 3,
                    "dropout": 0.0,
                    "out_activation": "tanh",
            },
            "train": {
                    "batch_size": 512,
                    "lr": 1e-3,
            }

        }
    }

    model_str = "LitTransitionModel"
    callbacks_A = [
        ModelCheckpoint(monitor="validation_loss_transition_model_A", mode="min", filename=f"{model_str}_A"),
        EarlyStopping(
            monitor="validation_loss_transition_model_A",
            min_delta=5e-4,
            mode="min",
            patience=250,
        ),
    ]

    transition_model_A = LitTransitionModel(
        config_A
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        max_time="00:07:55:00",
        accelerator="cpu",
        callbacks=callbacks_A,
    )

    trainer.fit(transition_model_A)

if __name__ == "__main__":
    main()
