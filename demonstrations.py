import logging
import os
from collections import defaultdict
from multiprocessing import cpu_count

import numpy as np
import wandb
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.offline.json_writer import JsonWriter
from tqdm import tqdm

from environments.environment_robot_task import EnvironmentRobotTask
from mapper import Mapper
from orchestrator import Orchestrator
from stage import Stage
from trainer.expert import Expert


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

                artifact = wandb.Artifact(name=self.hash, type=self.__class__.__name__)
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
            wandb_checkpoint_path = (
                f"{wandb_config['entity']}/"
                f"{wandb_config['project']}/"
                f"{self.hash}:latest"
            )

            download_folder = (
                wandb.Api().artifact(wandb_checkpoint_path).download(self.tmpdir)
            )

            self.load(download_folder)
        elif os.path.exists(path):
            logging.info(f"Loading {self.__class__.__name__} from {path}")
            self.trajectories = JsonReader(path).read_all_files()
        else:
            raise ValueError(f"Invalid path: {path}")


class RandomPolicy:
    def __init__(self, env_config):
        env = EnvironmentRobotTask(env_config)
        self.action_space = env.action_space

    def compute_actions(self, observations):
        return {env_id: self.action_space.sample() for env_id in observations.keys()}


class EnvironmentSampler(Demonstrations):
    def __init__(self, config):
        super(EnvironmentSampler, self).__init__(config)

    def generate(self):
        config = self.config["EnvironmentSampler"]

        max_trials = config.get("max_trials", np.inf)
        num_demonstrations = config["num_demonstrations"]
        discard_unsuccessful = config.get("discard_unsuccessful", False)

        policy_type = config.get("policy", "Random")
        env_config = self.config[config.get("env")]

        if policy_type == "Random":
            assert (
                discard_unsuccessful == False
            ), "discard_unsuccessful is not supported for Random policy"

            policy = RandomPolicy(env_config["env_config"])
        elif policy_type == "Expert":
            self.config["Expert"]["model_config"].update(env_config)
            policy = Expert(self.config).model
        else:
            raise ValueError(f"Invalid policy: {policy_type}")

        if max_trials < num_demonstrations:
            logging.warning(
                f"max_trials ({max_trials}) is smaller than "
                f"num_demonstrations ({num_demonstrations}). "
                f"Setting max_trials to num_demonstrations."
            )
            max_trials = num_demonstrations

        trials_launched = 0
        trials_completed = 0
        self.trajectories = []

        with Orchestrator(
            env_config["env_config"], config.get("num_workers", cpu_count())
        ) as orchestrator:
            success_criterion = orchestrator.success_criterion

            responses = orchestrator.reset_all()

            eps_ids = {ii: ii for ii in range(cpu_count())}

            batch_builder = defaultdict(SampleBatchBuilder)

            pbar = tqdm(total=num_demonstrations)

            while (
                len(self.trajectories) < num_demonstrations
                and trials_completed < max_trials
            ):
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

                        batch_builder[env_id].add_values(dones=done, rewards=reward)

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
                    for env_id, action in policy.compute_actions(
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

        demonstrations = EnvironmentSampler(self.config)
        mapper = Mapper(self.config)

        env = EnvironmentRobotTask(self.config["EnvTarget"]["env_config"])
        preprocessor_obs = get_preprocessor(env.observation_space)(
            env.observation_space
        )
        preprocessor_actions = get_preprocessor(env.action_space)(env.action_space)
        del env

        mapped_trajectories = mapper.map_trajectories(demonstrations.trajectories)

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
            **EnvironmentSampler.get_relevant_config(config),
            **Mapper.get_relevant_config(config),
        }
