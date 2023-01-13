import os
import logging
import glob

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from copy import deepcopy
import numpy as np

from lit_models.transition_model import LitTransitionModel
from lit_models.discriminator import LitDiscriminator
from lit_models.state_mapper import LitStateMapper
from lit_models.trajectory_mapper import LitTrajectoryMapper
from stage import Stage
from mapper import Mapper


class LitTrainer(Stage):
    model = None
    model_cls = None
    model_config = None
    run_id = None
    trainer = None

    def __init__(self, config):
        if self.model_cls == "transition_model":
            self.model_cls = LitTransitionModel
        elif self.model_cls == "discriminator":
            self.model_cls = LitDiscriminator
        elif self.model_cls == "state_mapper":
            self.model_cls = LitStateMapper
        elif self.model_cls == "trajectory_mapper":
            self.model_cls = LitTrajectoryMapper
        else:
            raise ValueError(f"Unknown model class {self.model_cls}")
        super(LitTrainer, self).__init__(config)

    def save(self, path=None):
        if path is None and self.config["cache"]["mode"] == "wandb":
            with wandb.init(
                id=self.run_id, **self.config["wandb_config"], resume="must"
            ) as run:
                self.save(self.tmpdir)

                artifact = wandb.Artifact(name=self.hash, type=self.__class__.__name__)
                artifact.add_dir(self.tmpdir)
                run.log_artifact(artifact)
        elif os.path.exists(path):
            logging.info("Pytorch Lightning models are saved while training.")
        else:
            raise ValueError(f"Invalid path: {path}")

    def generate(self):
        self.model = self.model_cls(self.config)

    def load(self, path=None):
        if path is None and self.config["cache"]["mode"] == "wandb":
            wandb_config = self.config["wandb_config"]
            wandb_checkpoint_path = (
                f"{wandb_config['entity']}/"
                f"{wandb_config['project']}/"
                f"{self.hash}:latest"
            )
            logging.info(f"wandb artifact: {wandb_checkpoint_path}")

            wandb.Api().artifact(wandb_checkpoint_path).download(self.tmpdir)

            checkpoint_folders = glob.glob(self.tmpdir + "**/**/*.ckpt", recursive=True)

            assert checkpoint_folders, f"No checkpoints found in {self.tmpdir}"

            if len(checkpoint_folders) > 1:
                logging.warning(
                    f"More than one checkpoint folder found: " f"{checkpoint_folders}"
                )

            checkpoint_path = checkpoint_folders[0]

            self.load(checkpoint_path)
        elif os.path.exists(path):
            # self.model_config.update({"num_workers": 1, "num_gpus": 0})
            self.model = self.model_cls.load_from_checkpoint(
                checkpoint_path=path,
                map_location=torch.device("cpu"),
                config=self.config,
            )
        else:
            raise ValueError(f"Invalid path: {path}")

    def train(self):
        with wandb.init(
            config=self.get_relevant_config(self.config),
            **self.config.get("wandb_config", {"mode": "disabled"}),
            group=self.__class__.__name__,
            tags=[self.hash],
        ) as run:
            self.run_id = run.id
            wandb_logger = WandbLogger(save_dir=self.tmpdir)
            callbacks = [
                ModelCheckpoint(
                    monitor=f"validation_loss_{self.model_cls.__name__}{self.model_config['log_suffix']}",
                    mode="min",
                    filename=f"checkpoint_{self.hash}",
                )
            ]

            self.trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                max_time="00:07:55:00",
                max_epochs=self.model_config["train"]["max_epochs"],
                logger=wandb_logger,
                callbacks=callbacks,
                fast_dev_run=False,
            )
            self.trainer.fit(self.model)

    @classmethod
    def get_relevant_config(cls, config):
        return {
            "data": config[cls.__name__]["data"],
            "log_suffix": config[cls.__name__]["log_suffix"],
            "model": config[cls.__name__]["model"],
            "train": config[cls.__name__]["train"],
            "callbacks": config[cls.__name__]["callbacks"],
        }


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


class Discriminator(LitTrainer):
    def __init__(self, config):
        self.model_cls = config["Discriminator"]["model_cls"]
        self.model_config = config["Discriminator"]

        super(Discriminator, self).__init__(config)

    def generate(self):
        super(Discriminator, self).generate()
        self.train()

    @classmethod
    def get_relevant_config(cls, config):
        return super(Discriminator, cls).get_relevant_config(config)


class StateMapper(LitTrainer):
    discriminator = None

    def __init__(self, config):
        self.model_cls = config["StateMapper"]["model_cls"]
        self.model_config = config["StateMapper"]

        super(StateMapper, self).__init__(config)

    def generate(self):
        super(StateMapper, self).generate()
        self.discriminator = Discriminator(self.config)
        self.model.discriminator = deepcopy(self.discriminator.model)
        del self.discriminator
        super(StateMapper, self).train()

    @classmethod
    def get_relevant_config(cls, config):
        return super(StateMapper, cls).get_relevant_config(config)


class TrajectoryMapper(LitTrainer):
    transition_model = None
    state_mapper = None

    def __init__(self, config):
        self.model_cls = config["TrajectoryMapper"]["model_cls"]
        self.model_config = config["TrajectoryMapper"]
        super(TrajectoryMapper, self).__init__(config)

    def generate(self):
        super(TrajectoryMapper, self).generate()
        # load transition model
        self.transition_model = TransitionModel(self.config)
        self.model.transition_model = deepcopy(self.transition_model.model)
        del self.transition_model
        # load state mapper
        self.state_mapper = StateMapper(self.config)
        self.model.state_mapper = deepcopy(self.state_mapper.model)
        del self.state_mapper
        super(TrajectoryMapper, self).train()

    @classmethod
    def get_relevant_config(cls, config):
        return super(TrajectoryMapper, cls).get_relevant_config(config)


class WeaSCLMapper(Mapper):
    trajectory_mapper = None

    def __init__(self, config):
        super(WeaSCLMapper, self).__init__(config)

    def map_trajectories(self, trajectories):
        for trajectory in trajectories:
            yield self.map_trajectory(trajectory)

    def generate(self):
        self.trajectory_mapper = TrajectoryMapper(self.config)

    def load(self):
        self.trajectory_mapper = TrajectoryMapper(self.config)
        self.trajectory_mapper.load()

    def map_trajectory(self, trajectory):
        joint_positions_source = np.stack(
            [
                obs["state"]["robot"]["arm"]["joint_positions"]
                for obs in trajectory["obs"]
            ]
        )
        joint_positions_source = torch.from_numpy(joint_positions_source).float()

        actions_source = np.stack([action["arm"] for action in trajectory["actions"]])
        actions_source = torch.from_numpy(actions_source).float()

        predicted_states = self.trajectory_mapper.model.state_mapper(
            joint_positions_source
        )
        predicted_actions = self.trajectory_mapper.model(
            (joint_positions_source, actions_source)
        )

        trajectory = deepcopy(trajectory)

        for old_state, new_state in zip(trajectory["obs"], predicted_states):
            old_state["state"]["robot"]["arm"]["joint_positions"] = (
                new_state.cpu().detach().numpy()
            )

        for old_action, new_action in zip(trajectory["actions"], predicted_actions):
            old_action["arm"] = new_action.cpu().detach().tolist()

        return trajectory
