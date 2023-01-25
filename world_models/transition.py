import sys
from functools import cached_property
from pathlib import Path

import torch
from torch.nn import MSELoss

sys.path.append(str(Path(__file__).resolve().parents[1]))

from environments.environment_robot_task import EnvironmentRobotTask
from stage import LitStage
from utils.nn import create_network


class TransitionModel(LitStage):
    def __init__(self, config, **kwargs):
        super(TransitionModel, self).__init__(config, **kwargs)

    @cached_property
    def loss_function(self):
        return MSELoss()

    @cached_property
    def transition_model(self):
        env = EnvironmentRobotTask(self.config["EnvTarget"]["env_config"])

        dim_state = env.state_space["robot"]["arm"]["joint_positions"].shape[-1]
        dim_action = env.action_space["arm"].shape[-1]

        transition_model = create_network(
            in_dim=dim_state + dim_action,
            out_dim=dim_state,
            **self.config[self.__class__.__name__]["model"],
        )
        return transition_model

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
        loss_transition_model = self.loss_function(next_states_predicted, next_states)
        return loss_transition_model

    @classmethod
    def get_relevant_config(cls, config):
        config_ = super().get_relevant_config(config)

        config_["EnvTarget"] = config["EnvTarget"]

        return config_
