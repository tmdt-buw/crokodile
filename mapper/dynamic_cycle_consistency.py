import logging

import torch
from mapper_state import StateMapper
from pytorch_lightning.trainer.supporters import CombinedLoader

from stage import LitStage


class StateCycleConsistency(LitStage):
    def __init__(self, config):
        super(StateCycleConsistency, self).__init__(config)
        self.state_mapper_AB = StateMapper(config)
        self.state_mapper_BA = StateMapper(config)

    def get_model(self):
        logging.info(f"Models in {self.__class__.__name__} are loaded by config.")

    def configure_optimizers(self):
        optimizer_state_mapper_AB = torch.optim.AdamW(
            self.state_mapper_AB.parameters(),
            lr=self.config[self.__class__.__name__]["train"].get("lr", 3e-4),
        )
        optimizer_state_mapper_BA = torch.optim.AdamW(
            self.state_mapper_BA.parameters(),
            lr=self.config[self.__class__.__name__]["train"].get("lr", 3e-4),
        )
        return optimizer_state_mapper_AB, optimizer_state_mapper_BA

    def train_dataloader(self):
        dataloader_train_A = self.get_dataloader(
            self.config[self.__class__.__name__]["data"]["data_file_X"], "train"
        )
        dataloader_train_B = self.get_dataloader(
            self.config[self.__class__.__name__]["data"]["data_file_Y"], "train"
        )
        return CombinedLoader({"A": dataloader_train_A, "B": dataloader_train_B})

    def val_dataloader(self):
        dataloader_validation_A = self.get_dataloader(
            self.config[self.__class__.__name__]["data"]["data_file_X"], "test", False
        )
        dataloader_validation_B = self.get_dataloader(
            self.config[self.__class__.__name__]["data"]["data_file_Y"], "test", False
        )
        return CombinedLoader(
            {"A": dataloader_validation_A, "B": dataloader_validation_B}
        )

    def forward(self, x):
        logging.info(f"Forward pass is not implemented for {self.__class__.__name__}.")

    def loss(self, batch):
        batch_A = batch["A"]
        trajectories_states_X, _ = batch_A
        states_X = trajectories_states_X.reshape(-1, trajectories_states_X.shape[-1])

        batch_B = batch["B"]
        trajectories_states_Y, _ = batch_B
        states_Y = trajectories_states_Y.reshape(-1, trajectories_states_Y.shape[-1])

        # Cycle consistency loss X -> Y -> X
        states_X_mapped_AB = self.state_mapper_AB(states_X)
        states_X_pr = self.state_mapper_BA(states_X_mapped_AB)

        link_poses_X_gt = self.dht_models[0](states_X)
        link_poses_X_pr = self.dht_models[1](states_X_pr)

        self.state_mapper_AB.loss_function.to(link_poses_X_gt)
        (
            loss_state_mapper_XYX,
            loss_state_mapper_XYX_p,
            loss_state_mapper_XYX_o,
        ) = self.loss_function(link_poses_X_gt, link_poses_X_pr)

        # Cycle consistency loss Y -> X -> Y
        states_Y_mapped_BA = self.state_mapper_BA(states_Y)
        states_Y_pr = self.state_mapper_AB(states_Y_mapped_BA)

        link_poses_Y_gt = self.dht_models[1](states_Y)
        link_poses_Y_pr = self.dht_models[0](states_Y_pr)

        self.state_mapper_BA.loss_function.to(link_poses_Y_gt)

        (
            loss_state_mapper_YXY,
            loss_state_mapper_YXY_p,
            loss_state_mapper_YXY_o,
        ) = self.loss_function(link_poses_Y_gt, link_poses_Y_pr)

        cycle_loss = loss_state_mapper_XYX + loss_state_mapper_YXY
        cycle_loss_p = loss_state_mapper_XYX_p + loss_state_mapper_YXY_p
        cycle_loss_o = loss_state_mapper_XYX_o + loss_state_mapper_YXY_o

        return cycle_loss, cycle_loss_p, cycle_loss_o

    def training_step(self, batch, batch_idx):
        (cycle_loss, cycle_loss_p, cycle_loss_o) = self.loss(batch)
        self.log(
            f"train_loss_{self.log_id}",
            cycle_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"train_loss_{self.log_id}_p",
            cycle_loss_p,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"train_loss_{self.log_id}_o",
            cycle_loss_o,
            on_step=False,
            on_epoch=True,
        )
        return cycle_loss

    def validation_step(self, batch, batch_idx):
        (cycle_loss, cycle_loss_p, cycle_loss_o) = self.loss(batch)
        self.log(
            f"validation_loss_{self.log_id}",
            cycle_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"validation_loss_{self.log_id}_p",
            cycle_loss_p,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"validation_loss_{self.log_id}_o",
            cycle_loss_o,
            on_step=False,
            on_epoch=True,
        )
        return cycle_loss
