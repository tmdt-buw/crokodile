import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from models.model_split.state_mapper import LitStateMapper
from models.model_split.transition_model import LitTransitionModel
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from models.trajectory_encoder import TrajectoryEncoder
from itertools import chain

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nn import create_network, KinematicChainLoss
from utils.soft_dtw_cuda import SoftDTW

from config import data_folder


class LitActionMapper(pl.LightningModule):
    def __init__(
        self,
        data_file_A,
        data_file_B,
        action_mapper_config={},
        state_mapper_model: LitStateMapper = None,
        transition_model: LitTransitionModel = None,
        batch_size=32,
        num_workers=8,
        log_suffix="",
        weight_matrix_exponent=np.inf,
        **kwargs,
    ):
        super(LitActionMapper, self).__init__()
        self.save_hyperparameters()
        self.log_suffix = log_suffix
        if state_mapper_model:
            self.state_mapper = state_mapper_model
        else:
            raise ValueError("No state mapper model is provided.")

        if transition_model:
            self.transition_model = transition_model
        else:
            raise ValueError("No transition model is provided.")

        self.trajectory_encoder, self.policy = self.get_action_mapper(
            data_file_A, data_file_B, action_mapper_config
        )
        self.dtw_loss = self.get_loss(data_file_A, data_file_B)

    def get_action_mapper(self, data_file_X, data_file_B, action_mapper_config):
        data_path_X = os.path.join(data_folder, data_file_X)
        data_X = torch.load(data_path_X)

        data_path_B = os.path.join(data_folder, data_file_B)
        data_B = torch.load(data_path_B)

        trajectory_encoder = TrajectoryEncoder(
            state_dim=data_X["trajectories_states_train"].shape[-1],
            action_dim=data_X["trajectories_actions_train"].shape[-1],
            behavior_dim=action_mapper_config["behavior_dim"],
            max_len=data_X["trajectories_states_train"].shape[-1]
            + data_X["trajectories_actions_train"].shape[-1],
            **action_mapper_config["encoder"],
        )

        policy = create_network(
            in_dim=data_B["trajectories_states_train"].shape[-1]
            + action_mapper_config["behavior_dim"],
            out_dim=data_B["trajectories_actions_train"].shape[-1],
            **action_mapper_config["decoder"],
        )

        return trajectory_encoder, policy

    def get_loss(self, data_file_X, data_file_B):
        data_path_A = os.path.join(data_folder, data_file_X)
        data_A = torch.load(data_path_A)

        data_path_B = os.path.join(data_folder, data_file_B)
        data_B = torch.load(data_path_B)

        link_positions_A = self.state_mapper.dht_model_A(
            torch.zeros((1, data_A["trajectories_states_train"].shape[-1]))
        )[0, :, :3, -1]
        link_positions_B = self.state_mapper.dht_model_B(
            torch.zeros((1, data_B["trajectories_states_train"].shape[-1]))
        )[0, :, :3, -1]

        weight_matrix_AB_p, weight_matrix_AB_o = self.state_mapper.get_weight_matrices(
            link_positions_A, link_positions_B, self.hparams.weight_matrix_exponent
        )

        loss_soft_dtw_AB = SoftDTW(
            use_cuda=True,
            dist_func=KinematicChainLoss(
                weight_matrix_AB_p, weight_matrix_AB_o, reduction=False
            ),
        )

        return loss_soft_dtw_AB

    def configure_optimizers(self):
        optimizer_action_mapper = torch.optim.AdamW(
            chain(self.trajectory_encoder.parameters(), self.policy.parameters()),
            lr=self.hparams.action_mapper_config.get("lr", 3e-4),
        )
        return optimizer_action_mapper

    def get_dataloader(self, data_file, type="train", shuffle=True):
        data_path = os.path.join(data_folder, data_file)
        data = torch.load(data_path)

        trajectories_states = data[f"trajectories_states_{type}"].float()
        trajectories_actions = data[f"trajectories_actions_{type}"].float()

        dataloader = DataLoader(
            TensorDataset(trajectories_states, trajectories_actions),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )
        return dataloader

    def train_dataloader(self):
        dataloader_train_A = self.get_dataloader(self.hparams.data_file_A, "train")
        dataloader_train_B = self.get_dataloader(self.hparams.data_file_B, "train")
        return CombinedLoader({"A": dataloader_train_A, "B": dataloader_train_B})

    def val_dataloader(self):
        dataloader_test_A = self.get_dataloader(self.hparams.data_file_A, "test", False)
        dataloader_test_B = self.get_dataloader(self.hparams.data_file_B, "test", False)
        return CombinedLoader({"A": dataloader_test_A, "B": dataloader_test_B})

    def forward(self, states_A, actions_A):
        with torch.no_grad():
            # bz
            behaviors = self.trajectory_encoder(states_A, actions_A)
            # bs
            states_B = self.state_mapper(states_A[:, 0])

            for _ in range(states_A.shape[1]):
                # b(s+z)
                states_B_behaviors = torch.concat((states_B, behaviors), dim=-1)
                # ba
                actions_B = self.policy(states_B_behaviors)

        return actions_B

    def loss(self, batch):
        # bls, bla
        trajectories_states_A, trajectories_actions_A = batch

        # bz
        behaviors = self.trajectory_encoder(
            trajectories_states_A, trajectories_actions_A
        )

        trajectories_states_B = []

        # bs
        states_B = self.state_mapper(trajectories_states_A[:, 0])
        trajectories_states_B.append(states_B)

        for _ in range(trajectories_actions_A.shape[1]):
            # b(s+z)
            states_B_behaviors = torch.concat((states_B, behaviors), dim=-1)

            # ba
            actions_B = self.policy(states_B_behaviors)
            # b(s+a)
            states_actions_B = torch.concat((states_B, actions_B), dim=-1)
            # bs
            states_B = self.transition_model(states_actions_B)

            trajectories_states_B.append(states_B)

        # bls
        trajectories_states_B = torch.stack(trajectories_states_B).swapdims(0, 1)

        # (b*l)s
        states_A = trajectories_states_A.reshape(-1, trajectories_states_A.shape[-1])
        states_B = trajectories_states_B.reshape(-1, trajectories_states_B.shape[-1])

        # (b*l)p44
        link_poses_A = self.state_mapper.dht_model_A(states_A)
        link_poses_B = self.state_mapper.dht_model_B(states_B)

        # blp44
        link_poses_A = link_poses_A.reshape(
            *trajectories_states_A.shape[:2], *link_poses_A.shape[1:]
        )
        link_poses_B = link_poses_B.reshape(
            *trajectories_states_B.shape[:2], *link_poses_B.shape[1:]
        )

        loss = self.dtw_loss(link_poses_A, link_poses_B).mean()

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch["A"])
        self.log(
            "train_loss_action_mapper" + self.log_suffix,
            loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch["A"])
        self.log(
            "validation_loss_action_mapper" + self.log_suffix,
            loss,
            on_step=False,
            on_epoch=True,
        )
        return loss


def main():
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.loggers import TensorBoardLogger
    from config import wandb_config

    batch_size = 512
    action_mapper_config = {
        "behavior_dim": 64,
        "encoder": {
            "lr": 3e-3,
            "d_model": 16,
            "nhead": 4,
            "num_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 64,
        },
        "decoder": {
            "network_width": 256,
            "network_depth": 8,
            "dropout": 0.1,
            "out_activation": "tanh",
        },
    }

    model_str = f"action_mapper_{action_mapper_config['network_width']}_{action_mapper_config['network_depth']}_{batch_size}"
    logger = TensorBoardLogger("tb_logs", name="state_mapper", version=model_str)

    # wandb_config.update(
    #    {
    #        "group": "tmp",
    #        # "mode": "disabled"
    #    }
    # )
    # logger = WandbLogger(**wandb_config, log_model=True)

    callbacks_AB = [
        ModelCheckpoint(monitor="validation_loss_action_mapper_AB", mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="validation_loss_action_mapper_AB",
            min_delta=5e-4,
            mode="min",
            patience=250,
        ),
    ]

    callbacks_BA = [
        ModelCheckpoint(monitor="validation_loss_action_mapper_BA", mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="validation_loss_action_mapper_BA",
            min_delta=5e-4,
            mode="min",
            patience=250,
        ),
    ]

    data_file_A = "panda_5_20000_4000.pt"
    data_file_B = "ur5_5_20000_4000.pt"

    state_mapper_AB = LitStateMapper.load_from_checkpoint(
        "models/model_split/lightning_checkpoint/state_mapper/test_model/state_mapper_AB.ckpt"
    )
    transition_model_B = LitTransitionModel.load_from_checkpoint(
        "models/model_split/finished_models/transition_256_2_1024_B.ckpt"
    )

    state_mapper_model_AB = LitActionMapper(
        data_file_A,
        data_file_B,
        action_mapper_config,
        state_mapper_AB,
        transition_model_B,
        batch_size,
    )
    trainer = pl.Trainer(
        max_epochs=1500,
        max_time="00:07:55:00",
        accelerator="gpu",
        logger=logger,
        default_root_dir="models/model_split/lightning_checkpoint/action_mapper/",
        callbacks=callbacks_AB
    )
    trainer.fit(state_mapper_model_AB)

    state_mapper_BA = LitStateMapper.load_from_checkpoint(
        "models/model_split/lightning_checkpoint/state_mapper/test_model/state_mapper_AB.ckpt"
    )
    transition_model_A = LitTransitionModel.load_from_checkpoint(
        "models/model_split/finished_models/transition_256_2_1024_A.ckpt"
    )

    state_mapper_model_BA = LitActionMapper(
        data_file_A,
        data_file_B,
        action_mapper_config,
        state_mapper_BA,
        transition_model_A,
        batch_size,
    )

    trainer = pl.Trainer(
        max_epochs=1500,
        max_time="00:07:55:00",
        accelerator="gpu",
        logger=logger,
        default_root_dir="models/model_split/lightning_checkpoint/action_mapper/",
        callbacks=callbacks_BA
    )
    trainer.fit(state_mapper_model_BA)


if __name__ == "__main__":
    main()
