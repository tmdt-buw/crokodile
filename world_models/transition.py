import sys
from functools import cached_property
from pathlib import Path

import torch
from torch.nn import MSELoss

sys.path.append(str(Path(__file__).resolve().parents[1]))

from environments import get_env
from environments.environment_robot_task import EnvironmentRobotTask
from stage import LitStage
from utils.nn import create_network


class TransitionModel(LitStage):
    def __init__(self, config, **kwargs):
        super(TransitionModel, self).__init__(config, **kwargs)

    def init_models(self, config, **kwargs):
        # load environment to get correct state space dimensions
        env_config = config[self.env_type]["env_config"]
        env_config["name"] = config[self.env_type]["env"]
        env = get_env(env_config)

        dim_state = env.state_space["robot"]["arm"]["joint_positions"].shape[-1]
        dim_action = env.action_space["arm"].shape[-1]

        self.transition_model = create_network(
            in_dim=dim_state + dim_action,
            out_dim=dim_state,
            **config[self.__class__.__name__]["model"],
        )

    def get_state_dict(self):
        return self.transition_model.state_dict()

    def set_state_dict(self, state_dict):
        self.transition_model.load_state_dict(state_dict)

    def configure_optimizers(self):
        optimizer_model = torch.optim.AdamW(
            self.transition_model.parameters(),
            lr=self.config[self.__class__.__name__]["train"].get("lr", 3e-4),
        )
        return optimizer_model

    def loss(self, batch):
        trajectories_states, trajectories_actions = batch

        states = trajectories_states[:, :-1].reshape(-1, trajectories_states.shape[-1])
        next_states = trajectories_states[:, :-1].reshape(
            -1, trajectories_states.shape[-1]
        )

        actions = trajectories_actions.reshape(-1, trajectories_actions.shape[-1])

        states_actions = torch.concat(tensors=(states, actions), dim=-1)
        self.transition_model.to(states_actions)
        next_states_predicted = self.transition_model(states_actions)
        loss_transition_model = torch.nn.functional.mse_loss(
            next_states_predicted, next_states
        )
        return loss_transition_model

    @classmethod
    def get_relevant_config(cls, config):
        config_ = super().get_relevant_config(config)

        config_[cls.env_type] = config[cls.env_type]

        return config_


class TransitionModelSource(TransitionModel):
    env_type = "EnvSource"


class TransitionModelTarget(TransitionModel):
    env_type = "EnvTarget"
