import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from multiprocessing import cpu_count
from torch.cuda import device_count
import os
from tqdm import tqdm
from ray.rllib.agents.ppo import PPOTrainer
import wandb
import tempfile
from config import wandb_config

if __name__ == "__main__":
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/bc_data_ur5_reach")

    import ray.rllib.agents.marwil as marwil
    from ray.tune.registry import register_env
    from environments.environment_robot_task import EnvironmentRobotTask, Callbacks

    register_env("robot_task", lambda config: EnvironmentRobotTask(config))

    config = {
        "input": data_folder,

        "framework": "torch",

        "callbacks": Callbacks,

        "model": {
            "fcnet_hiddens": [180] * 5,
            "fcnet_activation": "relu",
            "vf_share_layers": False,

        },

        "env": "robot_task",
        "env_config": {
            "robot_config": {
                "name": "ur5",
                "scale": .5,
                "sim_time": .1,
                "offset": (1, 0, 0)
            },
            "task_config": {
                "name": "reach",
                "offset": (1, 0, 0)
            },
        },

        "actions_in_input_normalized": True,
        # "disable_env_checking": True,

        # Parallelize environment rollouts.
        # "num_workers": cpu_count(),
    }

    epochs_bc = 1_000
    epochs_ppo = 10_000

    bc_trainer = marwil.BCTrainer(config=config)
    learnt = False
    for i in tqdm(range(epochs_bc)):
        results = bc_trainer.train()

    config.update({
        "num_sgd_iter": 3,
        "lr": 3e-4,

        "train_batch_size": 2496,
        "sgd_minibatch_size": 256,

        "num_workers": cpu_count(),
        "num_gpus": device_count(),
    })

    config.pop("input")

    ppo_trainer = PPOTrainer(config)
    ppo_trainer.set_weights(bc_trainer.get_weights())

    with wandb.init(config=config, **wandb_config):

        pbar = tqdm(range(epochs_ppo))

        for epoch in pbar:
            results = ppo_trainer.train()
            pbar.set_description(f"avg. reward={results['episode_reward_mean']:.3f} | "
                                 f"success ratio={results['custom_metrics']['success_mean']:.3f}")
            # print(f"Epoch: {epoch} | avg. reward={results['episode_reward_mean']:.3f} | "
            #       f"success ratio={results['custom_metrics']['success_mean']:.3f}")

            wandb.log({
                'episode_reward_mean': results['episode_reward_mean'],
            }, step=epoch)

            wandb.log({
                'episode_success_mean': results['custom_metrics']["success_mean"],
            }, step=epoch)

            if results['custom_metrics']["success_mean"] > .97:
                break

        with tempfile.TemporaryDirectory() as dir:
            checkpoint = ppo_trainer.save(dir)

            artifact = wandb.Artifact('agent_panda_reach_ppo', type='rllib checkpoint')
            artifact.add_dir(dir)
            wandb.log_artifact(artifact)
