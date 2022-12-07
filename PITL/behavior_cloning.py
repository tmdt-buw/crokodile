## pre-training the bc agent
import os
import sys
import wandb
from tqdm import tqdm
from multiprocessing import cpu_count
from pathlib import Path
import tempfile

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import wandb_config
# pretrain the agent and then integrate the resulting weights to the ppo agent
'''
bc: offline dataset for the training phase
PPO: online Data --> environement = PyBullet

'''
import ray
ray.init(ignore_reinit_error=True)

# from ray.rllib.algorithms.bc import BC
from ray.rllib.agents.marwil.bc import BCTrainer, BC_DEFAULT_CONFIG
from ray.tune.registry import register_env
from ray.rllib.utils.test_utils import check_train_results

sys.path.append(str(Path(__file__).resolve().parents[1]))
from environments.environment_robot_task import EnvironmentRobotTask, Callbacks

register_env("robot_task", lambda config: EnvironmentRobotTask(config))

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/output-2022-08-30_21-19-43_worker-0_0.json")
default_config = BC_DEFAULT_CONFIG
config = {
    "input": data_path,
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
    "num_workers": 0, #cpu_count(),
    # "num_workers": 1,
    "num_gpus": 0,  #torch.cuda.device_count(),
    # "num_gpus": 1,
}

wandb_config.update({
    "group": "test_behavior_cloning",
    "mode": "disabled"
})

from ray.rllib.offline.wis_estimator import WeightedImportanceSamplingEstimator
from ray.rllib.offline.is_estimator import ImportanceSamplingEstimator

tr_config = config.copy()
tr_config["actions_in_input_normalized"] = True
# Without the evaluation, the agent returns nan values as a reward
tr_config["evaluation_parallel_to_training"] = True
tr_config["evaluation_interval"] = 1
tr_config["evaluation_num_workers"] = 1

tr_config.update({
    "evaluation_config": {
        "input": "sampler",
    },
    "input_config": {
        "format": "json",  # json or parquet
        # Path to data file or directory.
        "path": data_path,
        "postprocess_inputs": False,
        # Dataset allocates 0.5 CPU for each reader by default.
        # Adjust this value based on the size of your offline dataset.
        #"num_cpus_per_read_task": 0.5,
        "input_evaluation": ["is", "wis"],
    },

    "beta": 0.0,
})

from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.offline.wis_estimator import WeightedImportanceSamplingEstimator
from ray.rllib.offline.off_policy_estimator import OffPolicyEstimator
#from ray.rllib.offline.off_policy_estimator import FQETorchModel
# tr_config = {**default_config, **bc_config}

max_epochs = 25_000
with wandb.init(config=tr_config, **wandb_config):
    bccloning = BCTrainer(tr_config)

    min_reward = 0.5
    learnt = False
    for epoch in tqdm(range(max_epochs)):
        trained_bc = bccloning.train()
        eval_results = trained_bc.get("evaluation")

        wandb.log({
            'episode_reward_mean': eval_results['episode_reward_mean'],
        }, step=epoch)

        wandb.log({
            'episode_success_mean': eval_results['custom_metrics']["success_mean"],
        }, step=epoch)

        if trained_bc:
            print("iter={} R={}".format(epoch, eval_results["episode_reward_mean"]))
            # Learn until good reward is reached in the actual env.
            if eval_results["episode_reward_mean"] > min_reward:
                print("learnt!")
                learnt = True
                break

    if not learnt:
        raise ValueError(
            "`BC` did not reach {} reward from expert offline "
            "data!".format(min_reward)
        )

    # tbc_weights = trained_bc.get_weights()

    with tempfile.TemporaryDirectory() as dir:
        checkpoint = trained_bc.save(dir)

        artifact = wandb.Artifact('bc_agent', type='rllib checkpoint')
        artifact.add_dir(dir)
        wandb.log_artifact(artifact)

# !!! The configuration of both agents (bc and ppo) must be the same
# in order to match the weights from both agents!!!
# from ray.rllib.agents.ppo import PPOTrainer
# ppotrainer = PPOTrainer(config)
# #plot the the generated weights and compare with bctrainer
# ppotrainer.get_weigths()
# # integrate the bc weights generated into the ppo trainer ( for target domain training)
# # tool: set_weights() or trainer.get_policy().set_weights()
# ppotrainer.set_weigths(tbc_weights)
