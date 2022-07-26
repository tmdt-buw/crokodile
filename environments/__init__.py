def get_env(env_config):

    environment_name = env_config.pop("name")

    if environment_name == "robot-task":
        from .environment_robot_task import EnvironmentRobotTask
        env = EnvironmentRobotTask(env_config)
    else:
        raise ValueError(f"Unknown environment: {environment_name}")

    env.name = environment_name

    return env
