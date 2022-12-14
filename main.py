import logging
import sys
import os
import re
from copy import deepcopy

from pathlib import Path
import tempfile
import numpy as np

import networkx as nx
from ray.rllib.algorithms.marwil import MARWIL
from ray.rllib.algorithms.bc import BC
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.models.preprocessors import get_preprocessor

import wandb
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from environments.environment_robot_task import EnvironmentRobotTask, Callbacks
from environments.environments_robot_task.robots import get_robot

register_env("robot_task", lambda config: EnvironmentRobotTask(config))
logging.getLogger().setLevel(logging.INFO)


class Stage:
    def __init__(self, config):
        self.config = config

        if "cache" not in config or config["cache"]["mode"] == "disabled":
            load = False
            save = False
        else:
            load = config["cache"].get("load", True)
            if type(load) is str:
                load = self.__class__.__name__ == load
            elif type(load) is list:
                load = self.__class__.__name__ in load

            save = config["cache"].get("save", True)
            if type(save) is str:
                save = self.__class__.__name__ == save
            elif type(save) is list:
                save = self.__class__.__name__ in save

            assert type(load) is bool and type(save) is bool, f"Invalid cache config: {config['cache']}"

        if load:
            try:
                logging.info(f"Loading cache for {self.__class__.__name__}.")

                self.load()
                # Don't save again if we loaded
                save = False
            except:
                logging.warning(f"Loading cache not possible for {self.__class__.__name__}. Generating instead.")
                self.generate()
        else:
            logging.info(f"Generating {self.__class__.__name__}.")

            self.generate()

        if save:
            logging.info(f"Saving {self.__class__.__name__}.")

            self.save()

    def generate(self):
        raise NotImplementedError(f"generate() not implemented for {self.__class__.__name__}")

    def load(self):
        raise NotImplementedError(f"load() not implemented for {self.__class__.__name__}")

    def save(self):
        raise NotImplementedError(f"save() not implemented for {self.__class__.__name__}")


class Trainer(Stage):
    model = None
    model_cls = None
    model_config = None

    def __init__(self, config):
        super(Trainer, self).__init__(config)

    def save(self, path=None):
        if path is None and self.config["cache"]["mode"] == "wandb":
            with tempfile.TemporaryDirectory() as dir:
                self.save(dir)

                artifact = wandb.Artifact(name=self.__class__.__name__, type="Algorithm")
                artifact.add_dir(dir)
                wandb.log_artifact(artifact)
        elif os.path.exists(path):
            self.model.save(path)
        else:
            raise ValueError(f"Invalid path: {path}")

    def generate(self):
        self.model = self.model_cls(self.model_config)

    def load(self, path=None):
        if path is None and self.config["cache"]["mode"] == "wandb":
            wandb_checkpoint_path = f"{wandb.run.entity}/{wandb.run.project}/{self.__class__.__name__}:latest"

            with tempfile.TemporaryDirectory() as tmpdir:
                download_folder = wandb.use_artifact(wandb_checkpoint_path).download(tmpdir)

                checkpoint_folder = os.path.join(download_folder, os.listdir(download_folder)[0])

                self.load(checkpoint_folder)
        elif os.path.exists(path):
            self.model_config.update({"num_workers": 1, "num_gpus": 0, })
            self.model = self.model_cls(self.model_config)
            self.model.restore(path)
        else:
            raise ValueError(f"Invalid path: {path}")

    def get_weights(self):
        self.model.get_weights()

    def train(self, max_epochs: int, success_threshold: float = 1.):
        logging.info(f"Train agent for {max_epochs} epochs or until success ratio of {success_threshold} is achieved.")

        pbar = tqdm(range(max_epochs))

        for epoch in pbar:
            results = self.model.train()

            if "evaluation" in results:
                results = results["evaluation"]

            episode_reward_mean = results.get("episode_reward_mean", np.nan)
            success_mean = results['custom_metrics'].get("success_mean", np.nan)

            description = ""

            if np.isfinite(episode_reward_mean):
                description += f"avg. reward={episode_reward_mean:.3f} | "

                wandb.log({
                    f'{self.__class__.__name__}_episode_reward_mean': episode_reward_mean,
                }, step=epoch)
            if np.isfinite(success_mean):
                description += f"success ratio={success_mean:.3f}"

                wandb.log({
                    f'{self.__class__.__name__}_episode_success_mean': success_mean,
                }, step=epoch)

            if description:
                pbar.set_description(f"avg. reward={results['episode_reward_mean']:.3f} | "
                                     f"success ratio={results['custom_metrics'].get('success_mean', np.nan):.3f}")

            if results['custom_metrics'].get("success_mean", -1) > success_threshold:
                break


class Mapper(Stage):
    def __init__(self, config):
        super(Mapper, self).__init__(config)

    def __new__(cls, config):
        robot_source_config = config["env_source"]["env_config"]["robot_config"]
        robot_target_config = config["env_target"]["env_config"]["robot_config"]

        if robot_source_config == robot_target_config:
            return super(Mapper, cls).__new__(cls)
        elif config["mapper"]["type"] == "explicit":
            return super(Mapper, cls).__new__(MapperExplicit)
        else:
            raise ValueError(f"Invalid mapper type: {config['mapper']['type']}")

    def generate(self):
        # No need to generate anything
        pass

    def load(self):
        # No need to load anything
        pass

    def save(self):
        # No need to save anything
        pass

    def map_trajectories(self, trajectories):
        return trajectories

    def map_trajectory(self, trajectory):
        return trajectory


class MapperExplicit(Mapper):
    def __init__(self, config):
        super(MapperExplicit, self).__init__(config)

    def generate(self):
        self.robot_source = get_robot(self.config["env_source"]["env_config"]["robot_config"])
        self.robot_target = get_robot(self.config["env_target"]["env_config"]["robot_config"])

    def load(self):
        # For the explicit mapper, there is nothing to load
        self.generate()

    def map_trajectories(self, trajectories):
        for trajectory in trajectories:
            try:
                yield self.map_trajectory(trajectory)
            except (AssertionError, nx.exception.NetworkXNoPath) as e:
                pass

    def map_trajectory(self, trajectory):

        joint_positions_source = np.stack(
            [obs["state"]["robot"]["arm"]["joint_positions"] for obs in trajectory["obs"]])
        joint_positions_source = torch.from_numpy(joint_positions_source).float()

        joint_angles_source = self.robot_source.state2angle(joint_positions_source)
        tcp_poses = self.robot_source.forward_kinematics(joint_angles_source)[:, -1]
        joint_angles_target = self.robot_target.inverse_kinematics(tcp_poses)

        # map joint angles inside joint limits (if possible)
        while (mask := joint_angles_target < self.robot_target.joint_limits[:, 0]).any():
            joint_angles_target[mask] += 2 * np.pi

        while (mask := joint_angles_target > self.robot_target.joint_limits[:, 1]).any():
            joint_angles_target[mask] -= 2 * np.pi

        mask = (joint_angles_target < self.robot_target.joint_limits[:, 0]) & \
               (joint_angles_target > self.robot_target.joint_limits[:, 1])

        # invalidate states which are outside of joint limits
        joint_angles_target[mask] = torch.nan

        if joint_angles_target.isnan().any(-1).all(-1).any():
            # todo: replace with custom exception
            raise AssertionError("At least one state from trajectory could not be mapped.")

        joint_positions_target = self.robot_target.angle2state(joint_angles_target)

        G = nx.DiGraph()
        G.add_node("start")
        G.add_node("end")

        # add edges from start to first states
        for nn, jp in enumerate(joint_positions_target[0]):
            G.add_edge("s", f"0/{nn}", attr={"from": -1, "to": None, "weight": 0.})

        # add edges from last states to end
        for nn, jp in enumerate(joint_positions_target[-1]):
            G.add_edge(f"{len(joint_positions_target) - 1}/{nn}", "e",
                       attr={"from": (len(joint_positions_target) - 1, nn), "to": None, "weight": 0.})

        for nn, (jp, jp_next) in enumerate(zip(joint_positions_target[:-1], joint_positions_target[1:])):
            actions = (jp_next.unsqueeze(0) - jp.unsqueeze(1)) / self.robot_target.scale

            # todo integrate kinematic chain similarity

            # select only valid edges
            actions_max = torch.nan_to_num(actions, torch.inf).abs().max(-1)[0]
            # idx_valid = torch.where(actions_max.isfinite())
            idx_valid = torch.where(actions_max < 1.)

            assert idx_valid[0].shape[0] > 0, "no valid actions found"

            for xx, yy in zip(*idx_valid):
                G.add_edge(f"{nn}/{xx}", f"{nn + 1}/{yy}",
                           attr={"from": (nn, xx), "to": (nn + 1, yy), "weight": actions_max[xx, yy]})

        path = nx.dijkstra_path(G, "s", "e")[1:-1]

        idx = [int(node.split("/")[1]) for node in path]
        best_states = joint_positions_target[range(len(joint_positions_target)), idx]
        best_actions = (best_states[1:] - best_states[:-1]) / self.robot_target.scale

        trajectory = deepcopy(trajectory)

        for old_state, new_state in zip(trajectory["obs"], best_states):
            old_state["state"]["robot"]["arm"]["joint_positions"] = new_state.cpu().detach().numpy()

        for old_action, new_action in zip(trajectory["actions"], best_actions):
            old_action["arm"] = new_action.cpu().detach().tolist()

        return trajectory


class Expert(Trainer):
    def __init__(self, config):
        self.model_cls = PPO
        self.model_config = config["expert"]["model"]
        self.model_config.update(config["env_source"])

        super(Expert, self).__init__(config)

    def generate(self):
        super(Expert, self).generate()
        self.train(**self.config["expert"]["train"])


class Demonstrations(Stage):
    trajectories = None

    def __init__(self, config):
        super(Demonstrations, self).__init__(config)

    def save(self, path=None):
        if path is None and self.config["cache"]["mode"] == "wandb":
            with tempfile.TemporaryDirectory() as tmpdir:
                self.save(tmpdir)

                artifact = wandb.Artifact(name=self.__class__.__name__, type="Iterable[SampleBatch]")
                artifact.add_dir(tmpdir)
                wandb.log_artifact(artifact)
        elif os.path.exists(path):
            writer = JsonWriter(path)

            for demonstration in self.trajectories:
                writer.write(demonstration)
        else:
            raise ValueError(f"Invalid path: {path}")

    def load(self, path=None):
        if path is None and self.config["cache"]["mode"] == "wandb":
            with tempfile.TemporaryDirectory() as tmpdir:
                wandb_checkpoint_path = f"{wandb.run.entity}/{wandb.run.project}/{self.__class__.__name__}:latest"

                download_folder = wandb.use_artifact(wandb_checkpoint_path).download(tmpdir)

                checkpoint_file = [f for f in os.listdir(download_folder) if re.match("^output.*\.json$", f)][0]
                checkpoint_path = os.path.join(download_folder, checkpoint_file)

                self.load(checkpoint_path)
        elif os.path.exists(path):
            self.trajectories = list(JsonReader(path).read_all_files())
        else:
            raise ValueError(f"Invalid path: {path}")


class DemonstrationsSource(Demonstrations):
    def __init__(self, config):
        super(DemonstrationsSource, self).__init__(config)

    def generate(self):
        config = self.config["demonstrations_source"]

        max_trials = config.get("max_trials")
        num_demonstrations = config["num_demonstrations"]

        if max_trials and max_trials < num_demonstrations:
            logging.warning(f"max_trials ({max_trials}) is smaller than num_demonstrations ({num_demonstrations}). "
                            f"Setting max_trials to num_demonstrations.")
            max_trials = num_demonstrations

        expert = Expert(self.config).model

        # TODO: parallelize https://discuss.ray.io/t/sample-rule-based-expert-demonstrations-in-rllib/3065

        trials = 0
        self.trajectories = []

        env = expert.env_creator(expert.workers.local_worker().env_context)
        batch_builder = SampleBatchBuilder()

        pbar = tqdm(total=num_demonstrations)

        while len(self.trajectories) < num_demonstrations:
            if max_trials:
                if trials >= max_trials:
                    break
                trials += 1
                pbar.set_description(f"trials={trials}/{max_trials} ({trials / max_trials * 100:.0f}%)")

            state = env.reset()

            done = False

            while not done:
                action = expert.compute_single_action(state)
                next_state, reward, done, info = env.step(action)

                batch_builder.add_values(
                    eps_id=len(self.trajectories),
                    obs=state,
                    actions=action,
                    dones=done,
                )

                state = next_state

            batch_builder.add_values(
                obs=state,
            )

            if env.success_criterion(state['goal']):
                self.trajectories.append(batch_builder.build_and_reset())
                pbar.update(1)
            else:
                batch_builder.build_and_reset()

        logging.info(f"Generated {len(self.trajectories)} demonstrations.")


class DemonstrationsTarget(Demonstrations):
    def __init__(self, config):
        super(DemonstrationsTarget, self).__init__(config)

    def generate(self):
        # config = self.config["demonstrations_target"]

        demonstrations = DemonstrationsSource(self.config)
        mapper = Mapper(self.config)

        mapped_trajectories = mapper.map_trajectories(demonstrations.trajectories)
        num_demonstrations = len(demonstrations.trajectories)

        del demonstrations
        del mapper

        env = EnvironmentRobotTask(self.config["env_target"]["env_config"])
        preprocessor_obs = get_preprocessor(env.observation_space)(env.observation_space)
        preprocessor_actions = get_preprocessor(env.action_space)(env.action_space)
        del env

        self.trajectories = []

        for trajectory in tqdm(mapped_trajectories, total=num_demonstrations):
            trajectory["obs"] = [preprocessor_obs.transform(obs) for obs in trajectory["obs"]]
            trajectory["actions"] = [preprocessor_actions.transform(action) for action in trajectory["actions"]]

            self.trajectories.append(trajectory)

        logging.info(f"Generated {len(self.trajectories)} target demonstrations "
                     f"from {num_demonstrations} source demonstrations "
                     f"({len(self.trajectories) / num_demonstrations * 100:1f}%).")


class Pretrainer(Trainer):
    def __init__(self, config):
        self.model_cls = BC
        self.model_config = config["pretrainer"]["model"]
        self.model_config.update(config["env_target"])

        super(Pretrainer, self).__init__(config)

    def generate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            demonstrations = DemonstrationsTarget(self.config)
            demonstrations.save(tmpdir)

            self.model_config.update({
                "input": tmpdir,
                "input_config": {
                    "format": "json",
                    "postprocess_inputs": False,
                }})

            super(Pretrainer, self).generate()
            self.train(**self.config["pretrainer"]["train"])


class Apprentice(Trainer):
    def __init__(self, config):
        self.model_cls = PPO
        self.model_config = config["apprentice"]["model"]
        self.model_config.update(config["env_target"])

        super(Apprentice, self).__init__(config)

    def generate(self):
        pretrainer = Pretrainer(self.config)
        weights = pretrainer.model.get_weights()
        del pretrainer

        super(Apprentice, self).generate()

        self.model.set_weights(weights)
        del weights
        self.train(**self.config["apprentice"]["train"])
