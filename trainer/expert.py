import logging
import os
import re
import tempfile
import numpy as np
from ray.rllib.algorithms.bc import BC
from ray.rllib.algorithms.marwil import MARWIL
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from tqdm import tqdm

import wandb
from environments.environment_robot_task import EnvironmentRobotTask

register_env("robot_task", lambda config: EnvironmentRobotTask(config))
logging.getLogger().setLevel(logging.INFO)


from demonstrations import DemonstrationsTarget
from stage import Stage


class Trainer(Stage):
    model = None
    model_cls = None
    model_config = None
    run_id = None

    def __init__(self, config):
        if self.model_cls == "PPO":
            self.model_cls = PPO
        elif self.model_cls == "BC":
            self.model_cls = BC
        elif self.model_cls == "MARWIL":
            self.model_cls = MARWIL
        else:
            raise ValueError(f"Invalid model_cls: {self.model_cls}")

        super(Trainer, self).__init__(config)

    def save(self, path=None):
        if path is None and self.config["cache"]["mode"] == "wandb":
            with wandb.init(
                    id=self.run_id, **self.config["wandb_config"],
                    resume="must"
            ) as run:
                self.save(self.tmpdir)

                artifact = wandb.Artifact(name=self.hash,
                                          type=self.__class__.__name__)
                artifact.add_dir(self.tmpdir)
                run.log_artifact(artifact)
        elif os.path.exists(path):
            self.model.save(path)
        else:
            raise ValueError(f"Invalid path: {path}")

    def generate(self):
        self.model = self.model_cls(self.model_config)

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

            checkpoint_folders = [
                f
                for f in os.listdir(self.tmpdir)
                if re.match(r"^checkpoint_\d+$", f)
            ]

            assert checkpoint_folders, f"No checkpoints found in {self.tmpdir}"

            if len(checkpoint_folders) > 1:
                logging.warning(
                    f"More than one checkpoint folder found: "
                    f"{checkpoint_folders}"
                )

            checkpoint_path = os.path.join(self.tmpdir, checkpoint_folders[0])

            self.load(checkpoint_path)
        elif os.path.exists(path):
            self.model_config.update({"num_workers": 1, "num_gpus": 0})
            self.model = self.model_cls(self.model_config)
            self.model.restore(path)
        else:
            raise ValueError(f"Invalid path: {path}")

    def get_weights(self):
        self.model.get_weights()

    def train(self, max_epochs: int, success_threshold: float = 1.0):
        logging.info(
            f"Train {self.__class__.__name__} for {max_epochs} epochs "
            f"or until success ratio of {success_threshold} is achieved."
        )

        with wandb.init(
                config=self.get_relevant_config(self.config),
                **self.config["wandb_config"],
                group=self.__class__.__name__,
                tags=[self.hash],
        ) as run:
            self.run_id = run.id
            pbar = tqdm(range(max_epochs))

            for epoch in pbar:
                results = self.model.train()

                if "evaluation" in results:
                    results = results["evaluation"]

                episode_reward_mean = results.get("episode_reward_mean",
                                                  np.nan)
                success_mean = results["custom_metrics"].get("success_mean",
                                                             np.nan)

                description = ""

                if np.isfinite(episode_reward_mean):
                    description += f"avg. reward={episode_reward_mean:.3f} | "

                    run.log(
                        {
                            "episode_reward_mean": episode_reward_mean,
                        },
                        step=epoch,
                    )
                if np.isfinite(success_mean):
                    description += f"success ratio={success_mean:.3f}"

                    run.log(
                        {
                            "episode_success_mean": success_mean,
                        },
                        step=epoch,
                    )

                if description:
                    pbar.set_description(
                        f"avg. reward={episode_reward_mean:.3f} | "
                        f"success ratio={success_mean:.3f}"
                    )

                if success_mean > success_threshold:
                    break

    @classmethod
    def get_relevant_config(cls, config):
        return {
            cls.__name__: {
                "model": config[cls.__name__]["model"],
                "train": config[cls.__name__]["train"],
            }
        }


class Expert(Trainer):
    def __init__(self, config):
        self.model_cls = config["Expert"]["model_cls"]
        self.model_config = config["Expert"]["model"]
        self.model_config.update(config["EnvSource"])

        super(Expert, self).__init__(config)

    def generate(self):
        super(Expert, self).generate()
        self.train(**self.config["Expert"]["train"])

    @classmethod
    def get_relevant_config(cls, config):
        return super(Expert, cls).get_relevant_config(config)


class Pretrainer(Trainer):
    def __init__(self, config):
        self.model_cls = config["Pretrainer"]["model_cls"]
        self.model_config = config["Pretrainer"]["model"]
        self.model_config.update(config["EnvTarget"])

        super(Pretrainer, self).__init__(config)

    def generate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            demonstrations = DemonstrationsTarget(self.config)
            demonstrations.save(tmpdir)

            self.model_config.update(
                {
                    "input": tmpdir,
                    "input_config": {
                        "format": "json",
                        "postprocess_inputs": False,
                    },
                }
            )

            super(Pretrainer, self).generate()
            self.train(**self.config["Pretrainer"]["train"])

    @classmethod
    def get_relevant_config(cls, config):
        return {
            **super(Pretrainer, cls).get_relevant_config(config),
            **DemonstrationsTarget.get_relevant_config(config),
        }


class Apprentice(Trainer):
    def __init__(self, config):
        self.model_cls = config["Apprentice"]["model_cls"]
        self.model_config = config["Apprentice"]["model"]
        self.model_config.update(config["EnvTarget"])

        super(Apprentice, self).__init__(config)

    def generate(self):
        pretrainer = Pretrainer(self.config)
        weights = pretrainer.model.get_weights()
        del pretrainer

        super(Apprentice, self).generate()

        self.model.set_weights(weights)
        del weights
        self.train(**self.config["Apprentice"]["train"])

    @classmethod
    def get_relevant_config(cls, config):
        return {
            **super(Apprentice, cls).get_relevant_config(config),
            **Pretrainer.get_relevant_config(config),
        }
