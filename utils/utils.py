import os
import random

import numpy as np
import torch
from collections import OrderedDict


def unwind_dict_values(element, framework="np", device="cpu"):
    if type(element) not in (dict, OrderedDict):
        if framework == "np":
            element = np.array(element)
        return element

    values = []
    for key, value in element.items():
        values.append(unwind_dict_values(value, framework, device))

    if len(values):
        if framework == "np":
            return np.concatenate(values, axis=-1)
        elif framework == "torch":
            return torch.concat(values, dim=-1)
    else:
        if framework == "np":
            return np.array([])
        elif framework == "torch":
            return torch.tensor([], device=device)


def unwind_space_shapes(space, key_path=None):
    if space.shape is not None:
        return {key_path: space.shape}

    shapes = {}
    for key_, space_ in space.spaces.items():
        if key_path is None:
            key = key_
        else:
            key = f"{key_path}/{key_}"
        shapes.update(unwind_space_shapes(space_, key))

    return shapes


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
