from copy import deepcopy

import numpy as np
import torch

from lit_models.lit_trainer import LitTrainer, StateMapper
from lit_models.transition_model import TransitionModel
from mapper import Mapper


class TrajectoryMapper(LitTrainer):
    transition_model = None
    state_mapper = None

    def __init__(self, config):
        self.model_cls = config["TrajectoryMapper"]["model_cls"]
        self.model_config = config["TrajectoryMapper"]
        super(TrajectoryMapper, self).__init__(config)

    def generate(self):
        super(TrajectoryMapper, self).generate()
        # load transition model
        self.transition_model = TransitionModel(self.config)
        self.model.transition_model = deepcopy(self.transition_model.model)
        del self.transition_model
        # load state mapper
        self.state_mapper = StateMapper(self.config)
        self.model.state_mapper = deepcopy(self.state_mapper.model)
        del self.state_mapper
        super(TrajectoryMapper, self).train()

    @classmethod
    def get_relevant_config(cls, config):
        return super(TrajectoryMapper, cls).get_relevant_config(config)


class WeaSCLMapper(Mapper):
    trajectory_mapper = None

    def __init__(self, config):
        super(WeaSCLMapper, self).__init__(config)

    def map_trajectories(self, trajectories):
        for trajectory in trajectories:
            yield self.map_trajectory(trajectory)

    def generate(self):
        self.trajectory_mapper = TrajectoryMapper(self.config)

    def load(self):
        self.trajectory_mapper = TrajectoryMapper(self.config)
        self.trajectory_mapper.load()

    def map_trajectory(self, trajectory):
        joint_positions_source = np.stack(
            [
                obs["state"]["robot"]["arm"]["joint_positions"]
                for obs in trajectory["obs"]
            ]
        )
        joint_positions_source = torch.from_numpy(joint_positions_source).float()

        actions_source = np.stack([action["arm"] for action in trajectory["actions"]])
        actions_source = torch.from_numpy(actions_source).float()

        predicted_states = self.trajectory_mapper.model.state_mapper(
            joint_positions_source
        )
        predicted_actions = self.trajectory_mapper.model(
            (joint_positions_source, actions_source)
        )

        trajectory = deepcopy(trajectory)

        for old_state, new_state in zip(trajectory["obs"], predicted_states):
            old_state["state"]["robot"]["arm"]["joint_positions"] = (
                new_state.cpu().detach().numpy()
            )

        for old_action, new_action in zip(trajectory["actions"], predicted_actions):
            old_action["arm"] = new_action.cpu().detach().tolist()

        return trajectory
