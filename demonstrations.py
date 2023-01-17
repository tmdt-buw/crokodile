import logging
import os
from collections import defaultdict
from multiprocessing import cpu_count

import numpy as np
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.offline.json_writer import JsonWriter
from tqdm import tqdm

import wandb
from environments.environment_robot_task import EnvironmentRobotTask
from orchestrator import Orchestrator

from mapper import Mapper
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
