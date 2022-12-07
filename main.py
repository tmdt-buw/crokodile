import logging
import sys
import os
import re
from copy import deepcopy
from multiprocessing import cpu_count
from pathlib import Path
import tempfile
import numpy as np

import networkx as nx
from ray.rllib.algorithms.bc import BC_DEFAULT_CONFIG, BC
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.offline.json_reader import JsonReader

import wandb
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
import torch
from tqdm import tqdm
from ray.rllib.algorithms import Algorithm
from ray.rllib.policy.sample_batch import SampleBatch
from config import wandb_config
from typing import Iterable

sys.path.append(str(Path(__file__).resolve().parents[1]))
from environments.environment_robot_task import EnvironmentRobotTask, Callbacks
from environments.environments_robot_task.robots.robot import Robot
from environments.environments_robot_task.robots import get_robot

register_env("robot_task", lambda config: EnvironmentRobotTask(config))


def train(algorithm: Algorithm, max_epochs: int, success_threshold: float = 1., wandb_prefix=""):
    logging.info(f"Train agent for {max_epochs} epochs or until success ratio of {success_threshold} is achieved.")

    pbar = tqdm(range(max_epochs))

    for epoch in pbar:
        results = algorithm.train()
        pbar.set_description(f"avg. reward={results['episode_reward_mean']:.3f} | "
                             f"success ratio={results['custom_metrics']['success_mean']:.3f}")

        wandb.log({
            f'{wandb_prefix}episode_reward_mean': results['episode_reward_mean'],
        }, step=epoch)

        wandb.log({
            f'{wandb_prefix}episode_success_mean': results['custom_metrics']["success_mean"],
        }, step=epoch)

        if results['custom_metrics']["success_mean"] > success_threshold:
            break


def save_trainer(algorithm: Algorithm, path: str):
    return algorithm.save(path)


def save_trainer_to_wandb(algorithm: Algorithm, name: str):
    with tempfile.TemporaryDirectory() as dir:
        checkpoint = save_trainer(algorithm, dir)

        artifact = wandb.Artifact(name=name, type="Trainer")
        artifact.add_dir(dir)
        wandb.log_artifact(artifact)

    return checkpoint


def restore_trainer(algorithm: Algorithm, path: str):
    algorithm.restore(path)


def restore_trainer_from_wandb(trainer, wandb_checkpoint_path: str):
    if wandb_checkpoint_path is None:
        wandb_checkpoint_path = f"{wandb.run.entity}/{wandb.run.project}/expert:latest"

    with tempfile.TemporaryDirectory() as tmpdir:
        download_folder = wandb.use_artifact(wandb_checkpoint_path).download(tmpdir)

        checkpoint_folder = os.path.join(download_folder, os.listdir(download_folder)[0])
        restore_trainer(trainer, checkpoint_folder)


def generate_demonstrations(algorithm: Algorithm, num_demonstrations: int, max_trials: int = None):
    if max_trials and max_trials < num_demonstrations:
        logging.warning(f"max_trials ({max_trials}) is smaller than num_demonstrations ({num_demonstrations}). "
                        f"Setting max_trials to num_demonstrations.")
        max_trials = num_demonstrations

    trials = 0
    demonstrations = []

    env = algorithm.env_creator(algorithm.workers.local_worker().env_context)
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
            action = algorithm.compute_single_action(state)
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


def save_demonstrations(demonstrations: Iterable[SampleBatch], path: str):
    writer = JsonWriter(path)

    for demonstration in demonstrations:
        writer.write(demonstration)


def save_demonstrations_to_wandb(demonstrations: Iterable[SampleBatch], name: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_demonstrations(demonstrations, tmpdir)

        artifact = wandb.Artifact(name=name, type="List[SampleBatch]")
        artifact.add_dir(tmpdir)
        wandb.log_artifact(artifact)


def load_demonstrations_from_wandb(wandb_checkpoint_path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        download_folder = wandb.use_artifact(wandb_checkpoint_path).download(tmpdir)

        checkpoint_file = [f for f in os.listdir(download_folder) if re.match("^output.*\.json$", f)][0]
        checkpoint_path = os.path.join(download_folder, checkpoint_file)

        reader = JsonReader(checkpoint_path)

        for demonstration in reader.read_all_files():
            yield demonstration


def map_trajectories(robot_source: Robot, robot_target: Robot, trajectories: Iterable[SampleBatch]):
    for trajectory in trajectories:
        try:
            yield map_trajectory(robot_source, robot_target, trajectory)
        except (AssertionError, nx.exception.NetworkXNoPath):
            pass


def map_trajectory(robot_source: Robot, robot_target: Robot, trajectory: SampleBatch):
    joint_positions_source = np.stack([obs["state"]["robot"]["arm"]["joint_positions"] for obs in trajectory["obs"]])
    joint_positions_source = torch.from_numpy(joint_positions_source).float()

    joint_angles_source = robot_source.state2angle(joint_positions_source)
    tcp_poses = robot_source.forward_kinematics(joint_angles_source)[:, -1]
    joint_angles_target = robot_target.inverse_kinematics(tcp_poses)

    if joint_angles_target.isnan().any(-1).all(-1).any():
        # todo: replace with custom exception
        raise AssertionError("At least one state from trajectory could not be mapped.")

    joint_positions_target = robot_target.angle2state(joint_angles_target)

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
        actions = (jp_next.unsqueeze(0) - jp.unsqueeze(1)) / robot_target.scale

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
    best_actions = (best_states[1:] - best_states[:-1]) / robot_target.scale

    for old_state, new_state in zip(trajectory["obs"], best_states):
        old_state["state"]["robot"]["arm"]["joint_positions"] = new_state.cpu().detach().numpy()

    for old_action, new_action in zip(trajectory["actions"], best_actions):
        old_action["arm"] = new_action.cpu().detach().tolist()

    return trajectory


if __name__ == '__main__':
    config_model = {
        "fcnet_hiddens": [180] * 5,
        "fcnet_activation": "relu",
        "vf_share_layers": False,
    }

    config_algorithm = {
        "framework": "torch",

        "callbacks": Callbacks,

        "model": config_model,

        "num_sgd_iter": 3,
        "lr": 3e-4,

        "train_batch_size": 2496,
        "sgd_minibatch_size": 256,

        "disable_env_checking": True,

        # Parallelize environment rollouts.
        "num_workers": cpu_count(),
        "num_gpus": torch.cuda.device_count(),
    }

    config_robot_source = {
        "name": "panda",
        "sim_time": .1,
        "scale": .1,
    }

    config_robot_target = {
        "name": "ur5",
        "sim_time": .1,
        "scale": .2,
    }

    config_task = {
        "name": "reach",
        "max_steps": 25,
        "accuracy": .03,
    }

    config_env_source = {
        "robot_config": config_robot_source,
        "task_config": config_task,
    }

    config_env_target = {
        "robot_config": config_robot_target,
        "task_config": config_task,
    }

    config_expert = deepcopy(config_algorithm)
    config_expert["env"] = "robot_task"
    config_expert["env_config"] = deepcopy(config_env_source)

    config_apprentice = deepcopy(config_algorithm)
    config_apprentice["env"] = "robot_task"
    config_apprentice["env_config"] = deepcopy(config_env_target)
    config_apprentice["env"] = "robot_task"

    config_bc = deepcopy(BC_DEFAULT_CONFIG)
    config_bc.update({
        "framework": "torch",
        "model": config_model,

        "env": "robot_task",
        "env_config": deepcopy(config_env_target),

        "actions_in_input_normalized": True,
        "input_evaluation": ["simulation"],
        "callbacks": Callbacks,

        "num_workers": cpu_count(),
        # "num_gpus": torch.cuda.device_count(),

        # "evaluation_parallel_to_training": True,
        # "evaluation_interval": 1,
        # "evaluation_num_workers": 1,
        # "off_policy_estimation_method": "simulation",
    })

    args = {
        "train_expert": False,
        "generate_demonstrations": True,
        "map_demonstrations": True,
        "pretrain_apprentice": True,
        "train_apprentice": True
    }

    with wandb.init(config=config_expert, **wandb_config):

        if args["train_expert"]:
            expert = PPO(config_expert)

            train(expert, 10_000, .9, "expert_")
            save_trainer_to_wandb(expert, "expert")

            # release remote resources
            [worker.stop.remote() for worker in expert.workers.remote_workers()]
        else:
            config_expert.update({"num_workers": 1, "num_gpus": 0})
            expert = PPO(config_expert)
            restore_trainer_from_wandb(expert, f"{wandb.run.entity}/{wandb.run.project}/expert:latest")

        if args["generate_demonstrations"]:
            print("Generate demonstrations")

            demonstrations_source = generate_demonstrations(expert, 10_000, 50_000)
            save_demonstrations_to_wandb(demonstrations_source, "demonstrations_source")
        else:
            print("Load demonstrations")
            demonstrations_source = load_demonstrations_from_wandb(
                f"{wandb.run.entity}/{wandb.run.project}/demonstrations_source:latest")
        del expert

        if args["map_demonstrations"]:
            print("Map demonstrations")

            robot_source = get_robot(config_robot_source)
            robot_target = get_robot(config_robot_target)
            demonstrations_target = map_trajectories(robot_source, robot_target, demonstrations_source)
            save_demonstrations_to_wandb(demonstrations_target, "demonstrations_target")
        else:
            print("Load mapped demonstrations")

            demonstrations_target = load_demonstrations_from_wandb(
                f"{wandb.run.entity}/{wandb.run.project}/demonstrations_target:latest")

        if args["pretrain_apprentice"]:
            print("Pretrain apprentice")
            with tempfile.TemporaryDirectory() as tmpdir:
                save_demonstrations(demonstrations_target, tmpdir)

                config_bc.update({"input_config": {
                    "format": "json",
                    "path": tmpdir,
                    "postprocess_inputs": False,
                }})

                pretrainer = BC(config_bc)
                train(pretrainer, 5_000, .9, "pretrainer_")

            apprentice = PPO(config_apprentice)
            apprentice.set_weights(pretrainer.get_weights())
            save_trainer_to_wandb(apprentice, "apprentice_pretrained")

            pretrainer.cleanup()
            del pretrainer
        else:
            print("Load pretrained apprentice")
            apprentice = PPO(config_apprentice)
            restore_trainer_from_wandb(expert, f"{wandb.run.entity}/{wandb.run.project}/apprentice_pretrained:best")

        if args["train_apprentice"]:
            print("Train apprentice")
            train(apprentice, 10_000, .9, "apprentice_")
            save_trainer_to_wandb(apprentice, "apprentice")
            [worker.stop.remote() for worker in apprentice.workers.remote_workers()]
        else:
            print("Load apprentice")
            config_apprentice.update({"num_workers": 1, "num_gpus": 0})
            apprentice = PPO(config_apprentice)
            restore_trainer_from_wandb(apprentice, f"{wandb.run.entity}/{wandb.run.project}/apprentice:latest")
