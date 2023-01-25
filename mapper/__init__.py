import logging

from stage import Stage


class Mapper(Stage):
    def __init__(self, config, **kwargs):
        super(Mapper, self).__init__(config)

    def __new__(cls, config, **kwargs):
        robot_source_config = config["EnvSource"]["env_config"]["robot_config"]
        robot_target_config = config["EnvTarget"]["env_config"]["robot_config"]

        if robot_source_config == robot_target_config:
            logging.warning(
                "Same source and target robot. "
                "If you are not debugging, this is probably a mistake."
            )
            return super(Mapper, cls).__new__(cls)
        elif config["Mapper"]["type"] == "explicit":
            # import here to avoid circular import
            from .mapper_explicit import MapperExplicit
            config["MapperExplicit"] = config["Mapper"]
            return super(Mapper, cls).__new__(MapperExplicit)
        elif config["Mapper"]["type"] == "weascl":
            from .mapper_weascl import MapperWeaSCL
            config["MapperWeaSCL"] = config["Mapper"]
            return super(Mapper, cls).__new__(MapperWeaSCL)
        else:
            raise ValueError(f"Invalid mapper type: {config['Mapper']['type']}")

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
