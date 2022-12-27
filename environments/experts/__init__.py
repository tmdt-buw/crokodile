import warnings


def get_expert(env):
    if env.task.name == "reach":
        from .expert_reach import Expert

        expert = Expert(env)
    elif env.task.name == "pick_place":
        from .expert_pick_place import Expert

        expert = Expert(env)
    else:
        warnings.warn("Expert not available")
        expert = None

    return expert
