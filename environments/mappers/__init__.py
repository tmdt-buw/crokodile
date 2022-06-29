import warnings

from .action_mappers import get_action_mapper
from .state_mappers import get_state_mapper


def get_mapper(env_source, env_target):
    try:
        state_mapper = get_state_mapper(env_source, env_target)
    except:
        warnings.warn("State mapper not available")
        state_mapper = None

    try:
        action_mapper = get_action_mapper(env_source, env_target)
    except:
        warnings.warn("Action mapper not available")
        action_mapper = None

    return state_mapper, action_mapper
