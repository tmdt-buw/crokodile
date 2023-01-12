import logging
import os
import re
import tempfile
from collections import defaultdict
from copy import deepcopy
from multiprocessing import cpu_count

import networkx as nx
import numpy as np
import torch
from ray.rllib.algorithms.bc import BC
from ray.rllib.algorithms.marwil import MARWIL
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.offline.json_writer import JsonWriter
from ray.tune.registry import register_env
from tqdm import tqdm

import wandb
from environments.environment_robot_task import EnvironmentRobotTask
from environments.environments_robot_task.robots import get_robot
from orchestrator import Orchestrator
from utils.nn import KinematicChainLoss, get_weight_matrices

register_env("robot_task", lambda config: EnvironmentRobotTask(config))
logging.getLogger().setLevel(logging.INFO)


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


class Mapper(Stage):
    def __init__(self, config):
        super(Mapper, self).__init__(config)

    def __new__(cls, config):
        robot_source_config = config["EnvSource"]["env_config"]["robot_config"]
        robot_target_config = config["EnvTarget"]["env_config"]["robot_config"]

        if robot_source_config == robot_target_config:
            logging.warning(
                "Same source and target robot. "
                "If you are not debugging, this is probably a mistake."
            )
            return super(Mapper, cls).__new__(cls)
        elif config["Mapper"]["type"] == "explicit":
            return super(Mapper, cls).__new__(MapperExplicit)
        else:
            raise ValueError(
                f"Invalid mapper type: {config['Mapper']['type']}")

    def generate(self):
        # No need to generate anything
        pass

    def load(self):
        # No need to load anything
        pass

    def save(self):
        # No need to save anything
        pass

    @classmethod
    def get_relevant_config(cls, config):
        config_ = {
            cls.__name__: config.get(cls.__name__, {}),
        }

        obj = cls.__new__(cls, config)
        if cls.__name__ != obj.__class__.__name__:
            config_.update(obj.get_relevant_config(config))

        return config_

    def map_trajectories(self, trajectories):
        return trajectories

    def map_trajectory(self, trajectory):
        return trajectory


class MapperExplicit(Mapper):
    def __init__(self, config):
        super(MapperExplicit, self).__init__(config)

    def generate(self):
        self.robot_source = get_robot(
            self.config["EnvSource"]["env_config"]["robot_config"]
        )
        self.robot_target = get_robot(
            self.config["EnvTarget"]["env_config"]["robot_config"]
        )

        angles_source = torch.zeros((1,) +
                                    self.robot_source.state_space["arm"][
                                        "joint_positions"].shape)
        link_poses_source = self.robot_source.forward_kinematics(angles_source)
        link_positions_source = link_poses_source[0, :, :3, 3]

        angles_target = torch.zeros((1,) +
                                    self.robot_target.state_space["arm"][
                                        "joint_positions"].shape)
        link_poses_target = self.robot_target.forward_kinematics(angles_target)
        link_positions_target = link_poses_target[0, :, :3, 3]

        weight_matrices = get_weight_matrices(
            link_positions_source,
            link_positions_target,
            self.config["Mapper"]["weight_matrix_exponent"]
        )

        self.kcl = KinematicChainLoss(*weight_matrices, reduction=False)

    def load(self):
        # For the explicit mapper, there is nothing to load
        self.generate()

    @classmethod
    def get_relevant_config(cls, config):
        return {
            "EnvSource": {
                "env_config": {
                    "robot_config":
                        config["EnvSource"]["env_config"]["robot_config"]
                }
            },
            "EnvTarget": {
                "env_config": {
                    "robot_config":
                        config["EnvTarget"]["env_config"]["robot_config"]
                }
            },
            "Mapper": {
                "weight_kcl": config["Mapper"]["weight_kcl"],
                "weight_matrix_exponent":
                    config["Mapper"]["weight_matrix_exponent"],
            }
        }

    def map_trajectories(self, trajectories):
        for trajectory in trajectories:
            try:
                yield self.map_trajectory(trajectory)
            except (AssertionError, nx.exception.NetworkXNoPath) as e:
                pass

    def map_trajectory(self, trajectory):

        joint_positions_source = np.stack(
            [
                obs["state"]["robot"]["arm"]["joint_positions"]
                for obs in trajectory["obs"]
            ]
        )
        joint_positions_source = torch.from_numpy(
            joint_positions_source).float()

        joint_angles_source = self.robot_source.state2angle(
            joint_positions_source)
        poses_source = self.robot_source.forward_kinematics(
            joint_angles_source)
        poses_tcp = poses_source[:, -1]
        joint_angles_target = self.robot_target.inverse_kinematics(poses_tcp)
        poses_target = self.robot_target.forward_kinematics(
            joint_angles_target.flatten(0, -2))

        poses_source_ = poses_source.repeat_interleave(
            joint_angles_target.shape[1], 0)

        kcl = self.kcl(poses_source_, poses_target).squeeze().reshape(
            joint_angles_target.shape[:2])

        # .reshape(
        # *joint_angles_target.shape[:2], -1, 4, 4).shape

        # map joint angles inside joint limits (if possible)
        while (
                mask := joint_angles_target <
                        self.robot_target.joint_limits[:, 0]
        ).any():
            joint_angles_target[mask] += 2 * np.pi

        while (
                mask := joint_angles_target > self.robot_target.joint_limits[:,
                                              1]
        ).any():
            joint_angles_target[mask] -= 2 * np.pi

        mask = (joint_angles_target < self.robot_target.joint_limits[:, 0]) & \
               (joint_angles_target > self.robot_target.joint_limits[:, 1])

        # invalidate states which are outside of joint limits
        joint_angles_target[mask] = torch.nan

        if joint_angles_target.isnan().any(-1).all(-1).any():
            # todo: replace with custom exception
            raise AssertionError(
                "At least one state from trajectory could not be mapped."
            )

        joint_positions_target = self.robot_target.angle2state(
            joint_angles_target)

        G = nx.DiGraph()
        G.add_node("start")
        G.add_node("end")

        # add edges from start to first states
        for nn, jp in enumerate(joint_positions_target[0]):
            G.add_edge("s", f"0/{nn}",
                       attr={"from": -1, "to": None, "weight": 0.0})

        # add edges from last states to end

        for nn, (jp, kcl_) in enumerate(zip(joint_positions_target[-1],
                                            kcl[-1])):
            if torch.isfinite(kcl_):
                G.add_edge(
                    f"{len(joint_positions_target) - 1}/{nn}",
                    "e",
                    attr={
                        "from": (len(joint_positions_target) - 1, nn),
                        "to": None,
                        "weight": self.config["Mapper"]["weight_kcl"] * kcl_,
                    },
                )

        for nn, (jp, kcl_, jp_next) in enumerate(
                zip(joint_positions_target[:-1], kcl[:-1],
                    joint_positions_target[1:])
        ):
            actions = (jp_next.unsqueeze(0) - jp.unsqueeze(1)) / \
                      self.robot_target.scale

            # select only valid edges
            actions_max = torch.nan_to_num(actions, torch.inf).abs().max(-1)[0]
            # idx_valid = torch.where(actions_max.isfinite())
            idx_valid = torch.where(actions_max < 1.0)

            assert idx_valid[0].shape[0] > 0, "no valid actions found"
            assert kcl_[idx_valid[0]].isfinite().all(), "kcl is not finite"

            for xx, yy in zip(*idx_valid):
                G.add_edge(
                    f"{nn}/{xx}",
                    f"{nn + 1}/{yy}",
                    attr={
                        "from": (nn, xx),
                        "to": (nn + 1, yy),
                        "weight": actions_max[xx, yy] + self.config["Mapper"][
                            "weight_kcl"] * kcl_[xx],
                    },
                )

        path = nx.dijkstra_path(G, "s", "e")[1:-1]

        idx = [int(node.split("/")[1]) for node in path]
        best_states = joint_positions_target[
            range(len(joint_positions_target)), idx]
        best_actions = (best_states[1:] - best_states[
                                          :-1]) / self.robot_target.scale

        trajectory = deepcopy(trajectory)

        for old_state, new_state in zip(trajectory["obs"], best_states):
            old_state["state"]["robot"]["arm"]["joint_positions"] = (
                new_state.cpu().detach().numpy()
            )

        for old_action, new_action in zip(trajectory["actions"], best_actions):
            old_action["arm"] = new_action.cpu().detach().tolist()

        return trajectory


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


class Demonstrations(Stage):
    trajectories = None

    def __init__(self, config):
        super(Demonstrations, self).__init__(config)

    def save(self, path=None):
        if path is None and self.config["cache"]["mode"] == "wandb":
            with wandb.init(
                    config=self.get_relevant_config(self.config),
                    **self.config["wandb_config"],
                    group=self.__class__.__name__,
                    tags=[self.hash],
            ):
                self.save(self.tmpdir)

                artifact = wandb.Artifact(name=self.hash,
                                          type=self.__class__.__name__)
                artifact.add_dir(self.tmpdir)
                wandb.log_artifact(artifact)
        elif os.path.exists(path):
            writer = JsonWriter(path)

            for demonstration in self.trajectories:
                writer.write(demonstration)
        else:
            raise ValueError(f"Invalid path: {path}")

    def load(self, path=None):
        if path is None and self.config["cache"]["mode"] == "wandb":
            wandb_config = self.config["wandb_config"]
            wandb_checkpoint_path = f"{wandb_config['entity']}/" \
                                    f"{wandb_config['project']}/" \
                                    f"{self.hash}:latest"

            download_folder = (
                wandb.Api().artifact(wandb_checkpoint_path).download(
                    self.tmpdir)
            )

            self.load(download_folder)
        elif os.path.exists(path):
            logging.info(f"Loading {self.__class__.__name__} from {path}")
            self.trajectories = JsonReader(path).read_all_files()
        else:
            raise ValueError(f"Invalid path: {path}")


class DemonstrationsSource(Demonstrations):
    def __init__(self, config):
        super(DemonstrationsSource, self).__init__(config)

    def generate(self):
        config = self.config["DemonstrationsSource"]

        max_trials = config.get("max_trials", np.inf)
        num_demonstrations = config["num_demonstrations"]
        discard_unsuccessful = config.get("discard_unsuccessful", True)

        if max_trials < num_demonstrations:
            logging.warning(
                f"max_trials ({max_trials}) is smaller than "
                f"num_demonstrations ({num_demonstrations}). "
                f"Setting max_trials to num_demonstrations."
            )
            max_trials = num_demonstrations

        expert = Expert(self.config).model

        trials_launched = 0
        trials_completed = 0
        self.trajectories = []

        with Orchestrator(
                expert.workers.local_worker().env_context, cpu_count()
        ) as orchestrator:
            success_criterion = orchestrator.success_criterion

            responses = orchestrator.reset_all()

            eps_ids = {ii: ii for ii in range(cpu_count())}

            batch_builder = defaultdict(SampleBatchBuilder)

            pbar = tqdm(total=num_demonstrations)

            while len(self.trajectories) < num_demonstrations and \
                    trials_completed < max_trials:
                requests = []
                required_predictions = {}

                for env_id, response in responses:
                    func, data = response

                    if func == "reset":
                        if type(data) == AssertionError:
                            logging.warning(
                                f"Resetting the environment resulted in "
                                f"AssertionError: {data}.\n"
                                f"The environment will be reset again."
                            )
                            requests.append((env_id, "reset", None))
                        else:
                            if trials_launched < max_trials:
                                trials_launched += 1
                                pbar.set_description(
                                    f"trials={trials_launched}/{max_trials} "
                                    f"({trials_launched / max_trials * 100:.0f}%)"
                                )

                                # Update eps_id
                                eps_ids[env_id] = max(eps_ids.values()) + 1

                                required_predictions[env_id] = data
                    elif func == "step":
                        state, reward, done, info = data

                        success = success_criterion(state["goal"])
                        done |= success

                        batch_builder[env_id].add_values(dones=done,
                                                         rewards=reward)

                        if done:
                            if success or not discard_unsuccessful:
                                batch_builder[env_id].add_values(obs=state)

                                self.trajectories.append(
                                    batch_builder[env_id].build_and_reset()
                                )
                                pbar.update(1)
                            else:
                                del batch_builder[env_id]

                            trials_completed += 1
                            requests.append((env_id, "reset", None))
                        else:
                            required_predictions[env_id] = state
                    else:
                        raise NotImplementedError(
                            f"Undefined behavior for {env_id} | {response}"
                        )

                if required_predictions:
                    # Generate predictions
                    for env_id, action in expert.compute_actions(
                            required_predictions
                    ).items():
                        requests.append((env_id, "step", action))

                        batch_builder[env_id].add_values(
                            eps_id=eps_ids[env_id],
                            obs=required_predictions[env_id],
                            actions=action,
                        )

                responses = orchestrator.send_receive(requests)

        logging.info(f"Generated {len(self.trajectories)} demonstrations.")

    @classmethod
    def get_relevant_config(cls, config):
        return {
            cls.__name__: config[cls.__name__],
            **Expert.get_relevant_config(config),
        }


class DemonstrationsTarget(Demonstrations):
    def __init__(self, config):
        super(DemonstrationsTarget, self).__init__(config)

    def generate(self):
        # config = self.config["demonstrations_target"]

        demonstrations = DemonstrationsSource(self.config)
        mapper = Mapper(self.config)

        env = EnvironmentRobotTask(self.config["EnvTarget"]["env_config"])
        preprocessor_obs = get_preprocessor(env.observation_space)(
            env.observation_space
        )
        preprocessor_actions = get_preprocessor(env.action_space)(
            env.action_space)
        del env

        mapped_trajectories = mapper.map_trajectories(
            demonstrations.trajectories)

        self.trajectories = []

        for trajectory in tqdm(mapped_trajectories):
            trajectory["obs"] = [
                preprocessor_obs.transform(obs) for obs in trajectory["obs"]
            ]
            trajectory["actions"] = [
                preprocessor_actions.transform(action)
                for action in trajectory["actions"]
            ]

            self.trajectories.append(trajectory)

        logging.info(
            f"Generated {len(self.trajectories)} target demonstrations "
            # f"from {num_demonstrations} source demonstrations "
            # f"({len(self.trajectories) / num_demonstrations * 100:1f}%)."
        )

    @classmethod
    def get_relevant_config(cls, config):
        return {
            **DemonstrationsSource.get_relevant_config(config),
            **Mapper.get_relevant_config(config),
        }


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