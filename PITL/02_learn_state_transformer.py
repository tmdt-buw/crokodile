"""
Train state mappings with dht models.
"""

import os
import sys
from pathlib import Path
import tempfile

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import wandb
from torch.multiprocessing import Process

from multiprocessing import cpu_count, set_start_method

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import wandb_config
from models.domain_mapper import LitDomainMapper

try:
    set_start_method('spawn')
except RuntimeError:
    pass

"""
Train state mappings with dht models.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import MSELoss
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.nn import NeuralNetwork, KinematicChainLoss, init_xavier_uniform
from models.dht import get_dht_model
from models.transformer import Seq2SeqTransformer
from models.transformer_encoder import Seq2SeqTransformerEncoder

from config import data_folder


class LitStateMapper(pl.LightningModule):
    def __init__(self,
                 data_file_A, data_file_B,
                 state_mapper_config={},
                 corrupt=False,
                 batch_size=32,
                 num_workers=1,
                 **kwargs
                 ):
        super(LitStateMapper, self).__init__()

        self.save_hyperparameters()

        data_path_A = os.path.join(data_folder, data_file_A)
        data_A = torch.load(data_path_A)

        data_path_B = os.path.join(data_folder, data_file_B)
        data_B = torch.load(data_path_B)

        self.dht_model_A = get_dht_model(data_A["dht_params"], data_A["joint_limits"])
        self.dht_model_B = get_dht_model(data_B["dht_params"], data_B["joint_limits"])

        self.state_mapper = Seq2SeqTransformer(max(data_A["states_train"].shape[1],
                                                   data_B["states_train"].shape[1]),
                                               **state_mapper_config)

        # link_positions_A = self.dht_model_A(torch.zeros((1, *data_A["states_train"].shape[1:])))[0, :, :3, -1]
        # link_positions_B = self.dht_model_B(torch.zeros((1, *data_B["states_train"].shape[1:])))[0, :, :3, -1]
        # # weight_matrix_AB_p, weight_matrix_AB_o = self.get_weight_matrices(link_positions_A, link_positions_B,
        # #                                                                   self.hparams.weight_matrix_exponent)
        # 
        # weight_matrix_AB_p = torch.zeros(link_positions_A.shape[0], link_positions_B.shape[0])
        # weight_matrix_AB_p[-1, -1] = 1.
        # weight_matrix_AB_o = torch.zeros(link_positions_A.shape[0], link_positions_B.shape[0])
        # weight_matrix_AB_o[-1, -1] = 1.

        # self.loss_fn_kinematics_AB = KinematicChainLoss(weight_matrix_AB_p, weight_matrix_AB_o)
        self.loss_fn = MSELoss(reduce=False)

        # manual optimization in training_step
        self.automatic_optimization = False

    """
        Maps states and actions A -> B
        Required by super class LightningModule

        Args:
            state_A
            action_A
        Returns:
            Mapped state and action
    """

    def forward(self, states, mode="A"):
        if mode == "A":
            dht_model = self.dht_model_A

        #
        # states_B_ = self.state_mapper_AB.get_dummB_tgt(states_A)
        #
        # for _ in range(states_B_.shape[-1] - 1):
        #     states_B = self.state_mapper_AB(states_A, states_B_)
        #     states_B_ = torch.nn.functional.pad(states_B[:, :-1], (1, 0), mode="constant", value=torch.nan)
        #
        # states_B = self.state_mapper_AB(states_A, states_B_)

        return states

    """
        Generate weights based on distances between relative positions of robot links

        Params:
            link_positions_{X, Y}: Positions of both robots in 3D space.
            weight_matrix_exponent_p: Parameter used to shape the position weight matrix by emphasizing similarity.
                weight = exp(-weight_matrix_exponent_p * distance)

        Returns:
            weight_matrix_XY_p: Weight matrix for positions
            weight_matrix_XY_p: Weight matrix for orientations. All zeros except weight which corresponds to the end effectors.
    """

    @staticmethod
    def get_weight_matrices(link_positions_X, link_positions_Y, weight_matrix_exponent_p, norm=True):
        link_positions_X = torch.cat((torch.zeros(1, 3), link_positions_X))
        link_lenghts_X = torch.norm(link_positions_X[1:] - link_positions_X[:-1], p=2, dim=-1)
        link_order_X = link_lenghts_X.cumsum(0)
        link_order_X = link_order_X / link_order_X[-1]

        link_positions_Y = torch.cat((torch.zeros(1, 3), link_positions_Y))
        link_lenghts_Y = torch.norm(link_positions_Y[1:] - link_positions_Y[:-1], p=2, dim=-1)
        link_order_Y = link_lenghts_Y.cumsum(0)
        link_order_Y = link_order_Y / link_order_Y[-1]

        weight_matrix_XY_p = torch.exp(
            -weight_matrix_exponent_p * torch.cdist(link_order_X.unsqueeze(-1), link_order_Y.unsqueeze(-1)))
        weight_matrix_XY_p = torch.nan_to_num(weight_matrix_XY_p, 1.)

        weight_matrix_XY_o = torch.zeros(len(link_positions_X), len(link_positions_Y))
        weight_matrix_XY_o[-1, -1] = 1

        if norm:
            weight_matrix_XY_p /= weight_matrix_XY_p.sum()
            weight_matrix_XY_o /= weight_matrix_XY_o.sum()

        return weight_matrix_XY_p, weight_matrix_XY_p

    def get_dataloader(self, data_file, mode="train"):
        data_path = os.path.join(data_folder, data_file)
        data = torch.load(data_path)

        states_train = data[f"states_{mode}"]
        actions_train = data[f"actions_{mode}"]
        next_states_train = data[f"next_states_{mode}"]

        dataloader_train = DataLoader(TensorDataset(states_train, actions_train, next_states_train),
                                      batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                                      shuffle=mode == "train", pin_memory=True)

        return dataloader_train

    """
        Generate dataloader used for training.
        Refer to pytorch lightning docs.
    """

    def train_dataloader(self):
        dataloader_train_A = self.get_dataloader(self.hparams.data_file_A, mode="train")
        dataloader_train_B = self.get_dataloader(self.hparams.data_file_B, mode="train")
        return CombinedLoader({"A": dataloader_train_A, "B": dataloader_train_B})

    """
        Generate dataloader used for validation.
        Refer to pytorch lightning docs.
    """

    def val_dataloader(self):
        dataloader_validation_A = self.get_dataloader(self.hparams.data_file_A, mode="test")
        dataloader_validation_B = self.get_dataloader(self.hparams.data_file_B, mode="test")
        return CombinedLoader({"A": dataloader_validation_A, "B": dataloader_validation_B})

    # def preprocess_batch(self, batch, dht_model, domain):
    #
    #
    # """
    #     Determine loss of state mapper on batch.
    # """

    def compute_loss_selfsupervised(self, batch, dht_model, domain):
        states, _, _ = batch
        states_ = torch.clone(states[:, :-1])
        target = torch.clone(states)

        positions_tcp = dht_model(states)[:, -1, :3, -1].detach()

        if self.hparams.corrupt:
            corruption_positions = torch.rand(states.shape[0], states.shape[1], device=states.device).argmax(1)
            corruption_mask = torch.zeros(states.shape[0], states.shape[1], device=states.device)

            corruption_mask[range(states.shape[0]), corruption_positions] = np.nan

            states = states + corruption_mask  # [:,:-1]
            # positions_tcp = positions_tcp + corruption_mask[:,-1:]

        padding_mask_src = torch.zeros((states.shape[0], self.state_mapper.max_len), device=states.device)
        padding_mask_src[:, states.shape[1]:] = 1.

        padding_mask_tgt = torch.clone(padding_mask_src)

        padding_mask_src = torch.nn.functional.pad(padding_mask_src, (1, 0), mode="constant", value=0)

        padding = self.state_mapper.max_len - states.shape[1]
        states = torch.nn.functional.pad(states, (0, padding), mode="constant", value=0)
        states_ = torch.nn.functional.pad(states_, (0, padding), mode="constant", value=0)
        target = torch.nn.functional.pad(target, (0, padding), mode="constant", value=0)
        corruption_mask = torch.nn.functional.pad(corruption_mask, (0, padding), mode="constant", value=0)

        src_domain = torch.zeros(states.shape[0], device=states.device, dtype=int) * domain
        tgt_domain = torch.zeros(states.shape[0], device=states.device, dtype=int) * domain

        prediction = self.state_mapper(states, positions_tcp, states_,
                                       src_domain, tgt_domain,
                                       padding_mask_src, padding_mask_tgt)

        if padding:
            prediction_positions_tcp = dht_model(prediction[:, -padding:])[:, -1, :3, -1].detach()
        else:
            prediction_positions_tcp = dht_model(prediction)[:, -1, :3, -1].detach()

        loss = self.loss_fn(target, prediction) * (1 - padding_mask_tgt) * corruption_mask.isnan().float().softmax(1)

        loss_position = torch.norm(positions_tcp - prediction_positions_tcp, p=2, dim=-1).mean()

        return loss.mean(), loss_position


    """
        ...
    """
    def compute_loss_transfer(self, batch_A, batch_B, dht_model_A, dht_model_B, domain_A, domain_B):
        states_A, _, _ = batch_A
        states_B, _, _ = batch_B

        positions_tcp_A = dht_model_A(states_A)[:, -1, :3, -1].detach()
        positions_tcp_B = dht_model_B(states_B)[:, -1, :3, -1].detach()


    #
    #     target_A = torch.clone(states_A)
    #
    #     positions_tcp_A = dht_model_A(states)[:, -1, :3, -1].detach()
    #
    #     padding_mask_src = torch.zeros((states.shape[0], self.state_mapper.max_len), device=states.device)
    #     padding_mask_src[:, states.shape[1]:] = 1.
    #
    #     padding_mask_tgt = torch.clone(padding_mask_src)
    #
    #     padding_mask_src = torch.nn.functional.pad(padding_mask_src, (1, 0), mode="constant", value=0)
    #
    #     padding = self.state_mapper.max_len - states.shape[1]
    #     states = torch.nn.functional.pad(states, (0, padding), mode="constant", value=0)
    #     states_ = torch.nn.functional.pad(states_, (0, padding), mode="constant", value=0)
    #     target = torch.nn.functional.pad(target, (0, padding), mode="constant", value=0)
    #     corruption_mask = torch.nn.functional.pad(corruption_mask, (0, padding), mode="constant", value=0)
    #
    #     src_domain = torch.zeros(states.shape[0], device=states.device, dtype=int) * domain
    #     tgt_domain = torch.zeros(states.shape[0], device=states.device, dtype=int) * domain
    #
    #     prediction = self.state_mapper(states, positions_tcp, states_,
    #                                    src_domain, tgt_domain,
    #                                    padding_mask_src, padding_mask_tgt)
    #
    #     if padding:
    #         prediction_positions_tcp = dht_model(prediction[:, -padding:])[:, -1, :3, -1].detach()
    #     else:
    #         prediction_positions_tcp = dht_model(prediction)[:, -1, :3, -1].detach()
    #
    #     loss = self.loss_fn(target, prediction) * (1 - padding_mask_tgt) * corruption_mask.isnan().float().softmax(1)
    #
    #     loss_position = torch.norm(positions_tcp - prediction_positions_tcp, p=2, dim=-1).mean()
    #
    #     return loss.mean(), loss_position


    """
        Perform training step. Customized behavior to enable accumulation of all losses into one variable.
        Refer to pytorch lightning docs.
    """

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        loss_denoise_A, loss_position_A = self.compute_loss_selfsupervised(batch["A"], self.dht_model_A, domain=0)
        loss_A = loss_denoise_A + loss_position_A
        self.manual_backward(loss_A)
        optimizer.step()

        optimizer.zero_grad()
        loss_denoise_B, loss_position_B = self.compute_loss_selfsupervised(batch["B"], self.dht_model_B, domain=1)
        loss_B = loss_denoise_B + loss_position_B
        self.manual_backward(loss_B)
        optimizer.step()

        self.log('training_loss_denoise_A', loss_denoise_A.item(), on_step=False, on_epoch=True)
        self.log('training_loss_pos_A', loss_position_A.item(), on_step=False, on_epoch=True)
        self.log('training_loss_A', loss_A.item(), on_step=False, on_epoch=True)
        self.log('training_loss_denoise_B', loss_denoise_B.item(), on_step=False, on_epoch=True)
        self.log('training_loss_pos_B', loss_position_B.item(), on_step=False, on_epoch=True)
        self.log('training_loss_B', loss_B.item(), on_step=False, on_epoch=True)
        # self.log('training_loss_B', loss_B.item(), on_step=False, on_epoch=True)
        # self.log('training_loss', loss.item(), on_step=False, on_epoch=True)

    """
        Perform validation step. Customized behavior to enable accumulation of all losses into one variable.
        Refer to pytorch lightning docs.
    """

    def validation_step(self, batch, batch_idx):
        loss_denoise_A, loss_position_A = self.compute_loss_selfsupervised(batch["A"], self.dht_model_A, domain=0)
        loss_A = loss_denoise_A + loss_position_A

        loss_denoise_B, loss_position_B = self.compute_loss_selfsupervised(batch["B"], self.dht_model_B, domain=1)
        loss_B = loss_denoise_B + loss_position_B

        # loss_B = self.compute_loss(batch["B"], self.dht_model_B)

        # loss = loss_A + loss_B

        self.log('validation_loss_A', loss_A.item(), on_step=False, on_epoch=True)
        self.log('validation_denoise_A', loss_denoise_A.item(), on_step=False, on_epoch=True)
        self.log('validation_loss_pos_A', loss_position_A.item(), on_step=False, on_epoch=True)
        self.log('validation_loss_B', loss_B.item(), on_step=False, on_epoch=True)
        self.log('validation_denoise_B', loss_denoise_B.item(), on_step=False, on_epoch=True)
        self.log('validation_loss_pos_B', loss_position_B.item(), on_step=False, on_epoch=True)

        # self.log('validation_loss_B', loss_B.item(), on_step=False, on_epoch=True)
        # self.log('validation_loss', loss.item(), on_step=False, on_epoch=True)

    """
        Helper function to generate all optimizers.
        Refer to pytorch lightning docs.
    """

    def configure_optimizers(self):
        optimizer_state_mapper = torch.optim.Adam(self.state_mapper.parameters(),
                                                  lr=self.hparams.state_mapper_config.get("lr", 3e-4))

        scheduler_state_mapper = {"scheduler": ReduceLROnPlateau(optimizer_state_mapper),
                                  "monitor": "validation_loss_state_mapper_AB",
                                  "name": "scheduler_optimizer_state_mapper_AB"
                                  }

        return optimizer_state_mapper


if __name__ == '__main__':
    CPU_COUNT = cpu_count()
    GPU_COUNT = torch.cuda.device_count()

    data_file_A = "panda_10000_1000.pt"
    data_file_B = "ur5_10000_1000.pt"

    wandb_config.update(
        {
            "notes": "add softmax loss weight",
            "group": "domain_mapper",
            "tags": ["state transformer", "self-supervised", "corruption loss"]
        }
    )

    config = {
        "data_file_A": data_file_A,
        "data_file_B": data_file_B,
        "state_mapper_config": {
            # "lr": 1e-2,
            # "d_model": 16,
            # "nhead": 4,
            # "num_encoder_layers": 2,
            # "num_decoder_layers": 2,
            # "dim_feedforward": 64,
        },
        "corrupt": True,
        "batch_size": 32,
        "max_epochs": 3_000,
    }

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        devices = -1 # torch.cuda.device_count()
        config["num_workers"] = cpu_count() // torch.cuda.device_count()
        # config["num_workers"] = cpu_count()
    else:
        devices = None
        config["num_workers"] = cpu_count()

    domain_mapper = LitStateMapper(
        **config
    )

    callbacks = [
        # ModelCheckpoint(monitor="validation_loss", mode="min"),
        LearningRateMonitor(logging_interval='epoch'),
        # EarlyStopping(monitor="validation_loss", mode="min", patience=1000)
    ]

    wandb_logger = WandbLogger(**wandb_config)

    # trainer = pl.Trainer()
    # trainer.fit(domain_mapper)
    #
    # exit()

    try:
        trainer = pl.Trainer(strategy=DDPStrategy(), accelerator="gpu", devices=devices,
                             precision=16,
                             logger=wandb_logger, max_epochs=config["max_epochs"], callbacks=callbacks)
        trainer.fit(domain_mapper)
    except:
        print("Mode 2")
        trainer = pl.Trainer(accelerator="gpu", devices=devices,
                             logger=wandb_logger, max_epochs=config["max_epochs"], callbacks=callbacks)
        trainer.fit(domain_mapper)
