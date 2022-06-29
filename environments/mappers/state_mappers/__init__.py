def get_state_mapper(env_source, env_target):

    robot_source = env_source.robot.name
    robot_target = env_target.robot.name

    task_source = env_source.task.name
    task_target = env_target.task.name

    if robot_source == robot_target and task_source == task_target:
        return lambda state: state
    else:
        # todo: load mapping model
        def state_mapper(state):
            raise NotImplementedError()

        return state_mapper