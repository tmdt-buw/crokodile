import datetime
import os
import sys
from multiprocessing import cpu_count
from pathlib import Path
from copy import deepcopy

import wandb
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
import tempfile
import re
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
import gym
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.agents.marwil import BCTrainer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from environments.environment_robot_task import EnvironmentRobotTask, Callbacks
from environments.environment_policy_extension import EnvironmentPolicyExtension

from utils.utils import unwind_dict_values

from models.domain_mapper import LitDomainMapper
from config import *

register_env("robot_task", lambda config: EnvironmentRobotTask(config))
register_env("policy_extension", lambda config: EnvironmentPolicyExtension(config))

env_source_name = "robot_task"
env_source_config = {
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
}

env_target_name = "robot_task"
env_target_config = {
    "robot_config": {
        "name": "ur5",
        "sim_time": .1,
        "scale": .1,
    },
    "task_config": {
        "name": "reach",
        "max_steps": 25,
        "accuracy": .03,
    },
}

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

    # Parallelize environment rollouts.
    "num_workers": cpu_count(),
    # "num_workers": 0,
    "num_gpus": torch.cuda.device_count(),
    # "num_gpus": 1,
}

max_epochs = 1_000_000


class CustomNetwork(TorchModelV2, nn.Module):
    """Adapted from https://github.com/ray-project/ray/blob/master/rllib/examples/models/custom_loss_model.py"""

    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            source_env_config: dict,
            state_mapper,
            action_mapper
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.state_mapper = state_mapper
        self.action_mapper = action_mapper

        self.policy = TorchFC(**source_env_config, model_config=model_config, name="policy")

        self.logit_std = torch.nn.Linear(self.policy.num_outputs // 2, self.num_outputs // 2)

        self.logit_std.weight.data = torch.zeros_like(self.logit_std.weight.data)
        self.logit_std.bias.data = torch.zeros_like(self.logit_std.bias.data)

    def forward(self, input_dict, state, seq_lens):
        state_arm = input_dict["obs"]["state"]["robot"]["arm"]["joint_positions"]

        state_arm_mapped = self.state_mapper(state_arm)

        input_dict["obs"]["state"]["robot"]["arm"]["joint_positions"] = state_arm_mapped

        input_dict["obs_flat"] = unwind_dict_values(input_dict["obs"], framework="torch",
                                                    device=input_dict["obs_flat"].device)

        logits, state = self.policy.forward(input_dict, state, seq_lens)

        logits_mean, logits_std = torch.chunk(logits, 2, dim=1)

        logits_mean_arm, logits_mean_hand = logits_mean.split([self.policy.action_space["arm"].shape[0],
                                                               self.policy.action_space["hand"].shape[0]], dim=1)

        logits_mean_arm_mapped = self.action_mapper(logits_mean_arm)

        logits_std_mapped = self.logit_std(logits_std)

        logits_mapped = torch.concat([logits_mean_arm_mapped, logits_mean_hand, logits_std_mapped], dim=1)

        return logits_mapped, state

    @override(ModelV2)
    def value_function(self):
        return self.policy.value_function()


wandb_config.update({
    "group": "panda2ur5_reach_ppo",
    "tags": ["random_init"],
    "notes": "test if target task can be learned from scratch",
})

# Train for n iterations and report results (mean episode rewards).
with wandb.init(config=config, **wandb_config):
    ModelCatalog.register_custom_model(
        "custom", CustomNetwork
    )

    config_source = config.copy()
    config_source["env"] = env_source_name
    config_source["env_config"] = env_source_config
    config_source["num_workers"] = 0
    config_source["num_gpus"] = 0

    agent_source = PPOTrainer(config_source)

    agent_source_path = "agent_panda_reach_ppo:best"
    domain_mapper_artifact = "model-ajmxaubu:best"

    with tempfile.TemporaryDirectory() as dir:
        download_folder = wandb.use_artifact(agent_source_path).download(dir)

        checkpoint_folder = os.path.join(download_folder, os.listdir(download_folder)[0])
        checkpoint_file = [f for f in os.listdir(checkpoint_folder) if re.match("checkpoint-\d+$", f)][0]
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
        agent_source.restore(checkpoint_path)

    with tempfile.TemporaryDirectory() as dir:
        download_folder = wandb.use_artifact(domain_mapper_artifact).download(dir)
        domain_mapper = LitDomainMapper.load_from_checkpoint(os.path.join(download_folder, "model.ckpt"))


    # with tempfile.TemporaryDirectory() as dir:
    #     api = wandb.Api()
    #     artifact = api.artifact(config["warm_start"])
    #     artifact_dir = artifact.download(dir)
    #
    #     domain_mapper = LitDomainMapper.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))

    domain_mapper = LitDomainMapper(
        data_file_A="panda_10000_1000.pt",
        data_file_B="ur5_10000_1000.pt",
    )

    config_target = config.copy()
    config_target["env"] = env_target_name
    config_target["env_config"] = env_target_config

    config_target["model"]["custom_model"] = "custom"
    config_target["model"]["custom_model_config"] = {
        "state_mapper": domain_mapper.state_mapper_BA,
        "action_mapper": domain_mapper.action_mapper_AB,
        "source_env_config": {
            "obs_space": agent_source.get_policy().model.obs_space,
            "action_space": agent_source.get_policy().model.action_space,
            "num_outputs": agent_source.get_policy().model.num_outputs
        }
    }

    agent_target = PPOTrainer(config_target)

    # # overwrite policy
    target_weights = agent_target.get_weights()

    for k, v in agent_source.get_weights()['default_policy'].items():
        target_weights['default_policy'][f"policy.{k}"] = v

    agent_target.set_weights(target_weights)

    del agent_source

    for epoch in range(max_epochs):
        results = agent_target.train()
        print(f"Epoch: {epoch} | avg. reward={results['episode_reward_mean']:.3f} | "
              f"success ratio={results['custom_metrics']['success_mean']:.3f}")

        wandb.log({
            'episode_reward_mean': results['episode_reward_mean'],
        }, step=epoch)

        wandb.log({
            'episode_success_mean': results['custom_metrics']["success_mean"],
        }, step=epoch)

        if results['custom_metrics']["success_mean"] > .97:
            break

    # checkpoint = agent_target.save(wandb.run.dir)
