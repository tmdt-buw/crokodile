import logging
import sys
import os
import re
from multiprocessing import cpu_count
from pathlib import Path
import tempfile
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.agents.marwil.bc import BCTrainer, BC_DEFAULT_CONFIG
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.offline.json_reader import JsonReader

import wandb
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from environments.environment_robot_task import EnvironmentRobotTask, Callbacks
from config import wandb_config

register_env("robot_task", lambda config: EnvironmentRobotTask(config))


class Apprentice:
    def __init__(self, config, bc_config=None):
        if bc_config is not None:
            self.bc_trainer = BCTrainer(tr_config)
        self.trainer = PPOTrainer(config)

    def pretrain(self, epochs):


    def train(self, max_epochs, success_threshold=1.):

        logging.info(f"Train agent for {max_epochs} epochs or until success ratio of {success_threshold} is achieved.")

        pbar = tqdm(range(max_epochs))

        for epoch in pbar:
            results = self.trainer.train()
            pbar.set_description(f"avg. reward={results['episode_reward_mean']:.3f} | "
                                 f"success ratio={results['custom_metrics']['success_mean']:.3f}")

            wandb.log({
                'episode_reward_mean': results['episode_reward_mean'],
            }, step=epoch)

            wandb.log({
                'episode_success_mean': results['custom_metrics']["success_mean"],
            }, step=epoch)

            if results['custom_metrics']["success_mean"] > success_threshold:
                break

    def save(self, path):
        return self.trainer.save(path)

    def save_to_wandb(self, artifact_config={}):
        with tempfile.TemporaryDirectory() as dir:
            checkpoint = self.save(dir)

            artifact_config_ = {"name": "expert", "type": "rllib.PPOTrainer"}
            artifact_config_.update(artifact_config)

            artifact = wandb.Artifact(**artifact_config_)
            artifact.add_dir(dir)
            wandb.log_artifact(artifact)

        return checkpoint

    def restore(self, path):
        self.trainer.restore(path)

    def restore_from_wandb(self, wandb_checkpoint_path=None):
        if wandb_checkpoint_path is None:
            wandb_checkpoint_path = f"{wandb.run.entity}/{wandb.run.project}/expert:latest"

        with tempfile.TemporaryDirectory() as dir:
            download_folder = wandb.use_artifact(wandb_checkpoint_path).download(dir)

            checkpoint_folder = os.path.join(download_folder, os.listdir(download_folder)[0])
            checkpoint_file = [f for f in os.listdir(checkpoint_folder) if re.match("^checkpoint-\d+$", f)][0]
            checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
            self.restore(checkpoint_path)

    def generate_demonstrations(self, num_demonstrations, max_trials=None):

        if max_trials and max_trials < num_demonstrations:
            logging.warning(f"max_trials ({max_trials}) is smaller than num_demonstrations ({num_demonstrations}). "
                            f"Setting max_trials to num_demonstrations.")
            max_trials = num_demonstrations

        trials = 0
        demonstrations = []

        env = self.trainer.env_creator(self.trainer.workers.local_worker().env_context)
        batch_builder = SampleBatchBuilder()

        pbar = tqdm(total=num_demonstrations)

        while len(demonstrations) < num_demonstrations:
            if max_trials:
                if trials >= max_trials:
                    break
                trials += 1
                pbar.set_description(f"trials={trials}/{max_trials} ({trials / max_trials * 100:.0f}%)")

            state = env.reset()

            done = False

            while not done:
                action = self.trainer.compute_action(state)
                next_state, reward, done, info = env.step(action)

                batch_builder.add_values(
                    obs=state,
                    actions=action,
                )

                state = next_state

            batch_builder.add_values(
                obs=state,
            )

            if env.success_criterion(state['goal']):
                demonstrations.append(batch_builder.build_and_reset())
                pbar.update(1)
            else:
                batch_builder.build_and_reset()

        logging.info(f"Generated {len(demonstrations)} demonstrations.")

        return demonstrations

    @staticmethod
    def save_demonstrations(demonstrations, path):
        writer = JsonWriter(path)

        for demonstration in demonstrations:
            writer.write(demonstration)

    @classmethod
    def save_demonstrations_to_wandb(cls, demonstrations, artifact_config={}):
        with tempfile.TemporaryDirectory() as dir:
            cls.save_demonstrations(demonstrations, dir)

            artifact_config_ = {"name": "demonstrations", "type": "rllib.JSONWriter"}
            artifact_config_.update(artifact_config)

            artifact = wandb.Artifact(**artifact_config_)
            artifact.add_dir(dir)
            wandb.log_artifact(artifact)

    #Todo: move to own data handler class
    @staticmethod
    def load_demonstrations_from_wandb(wandb_checkpoint_path=None):
        if wandb_checkpoint_path is None:
            wandb_checkpoint_path = f"{wandb.run.entity}/{wandb.run.project}/demonstrations:latest"

        with tempfile.TemporaryDirectory() as dir:
            download_folder = wandb.use_artifact(wandb_checkpoint_path).download(dir)

            checkpoint_file = [f for f in os.listdir(download_folder) if re.match("^output.*\.json$", f)][0]
            checkpoint_path = os.path.join(download_folder, checkpoint_file)

            reader = JsonReader(checkpoint_path)

            for demonstration in reader.read_all_files():
                yield demonstration


if __name__ == '__main__':
    config = {
        "framework": "torch",

        "callbacks": Callbacks,

        "model": {
            "fcnet_hiddens": [180] * 5,
            "fcnet_activation": "relu",
        },

        "num_sgd_iter": 3,
        "lr": 3e-4,

        "train_batch_size": 2496,
        "sgd_minibatch_size": 256,

        "env": "robot_task",
        "env_config": {
            "robot_config": {
                "name": "panda",
                "sim_time": .1,
                "scale": .1,
            },
            "task_config": {
                "name": "reach",
                "max_steps": 25,
                "accuracy": .03,
            },
        },

        # Parallelize environment rollouts.
        "num_workers": cpu_count(),
        "num_gpus": torch.cuda.device_count(),
    }

    # config.update({"num_workers": 1, "num_gpus": 1})

    # with wandb.init(config=config, **wandb_config):
    #     expert = Expert(config)
    #     expert.train(10_000, .9)
    #     expert.save_to_wandb()

    config.update({"num_workers": 1, "num_gpus": 1})
    #
    with wandb.init(config=config, **wandb_config):

        expert = Expert(config)
        expert.restore_from_wandb()
        demonstrations = expert.generate_demonstrations(10_000, 100_000)
        expert.save_demonstrations_to_wandb(demonstrations)

    with wandb.init(config=config, **wandb_config):
        demonstration = next(Expert.load_demonstrations_from_wandb())
        print(demonstration["obs"])
        print(demonstration["actions"])
