from typing import Iterable, Union, Tuple
from numbers import Real, Integral
import numpy as np
import torch
from torch import nn
import gym
from gym import Space
from gym.spaces import Box, Discrete


def make_vector(x: Union[Real, Iterable[Real]],
                length: Integral,
                dtype=Union[str, np.dtype]) -> np.ndarray:
    
    if isinstance(x, Real):
        return np.array([x] * int(length), dtype=dtype)
    else:
        result = np.array(x, dtype=dtype)
        if len(result.shape) != 1:
            raise ValueError(
                "Cannot make a 1-D vector."
                + " The argument implies shape: "
                + repr(result.shape)
            )
        if len(result) != length:
            raise ValueError(
                "Cannot make a 1-D vector of length "
                + repr(length)
                + ". The argument implies length: "
                + repr(len(result))
            )
        return result
    

def readonly_view(x: np.ndarray):
    x = x[:]
    x.flags.writeable = False
    return x


def positive_float_or_none(x) -> Union[int, None]:
    if x is None:
        return None
    x = float(x)
    if x <= 0.0:
        x = None
    return x


def positive_int_or_none(x) -> Union[int, None]:
    if x is None:
        return None
    x = int(x)
    if x <= 0:
        x = None
    return x


def positive_float(x) -> float:
    org_x = x
    x = float(x)
    if x <= 0:
        raise ValueError(
            "Expected a positive real number, but received: "
            + repr(org_x)
        )
    return x


def positive_int(x) -> int:
    org_x = x
    x = int(x)
    if x <= 0:
        raise ValueError(
            "Expected a positive integer, but received: "
            + repr(org_x)
        )
    return x


def non_negative_float(x) -> float:
    org_x = x
    x = float(x)
    if x < 0:
        raise ValueError(
            "Expected a non-negative real number, but received: "
            + repr(org_x)
        )
    return x


def non_negative_int(x) -> int:
    org_x = x
    x = int(x)
    if x < 0:
        raise ValueError(
            "Expected a non-negative integer, but received: "
            + repr(org_x)
        )
    return x


def fill_parameters(net: nn.Module, vector: torch.Tensor):
    """Fill the parameters of a torch module (net) from a vector.

    Args:
        net: The torch module whose parameter values will be filled.
        vector: A 1-D torch tensor which stores the parameter values.
    """
    if vector.requires_grad:
        vector = vector.detach()
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = vector[address:address + n]
        address += n
    assert address == len(vector), (
        "This vector is larger than expected"
    )


def get_parameter_vector(net: nn.Module) -> torch.Tensor:
    """Get all the parameters of a torch module (net) into a vector

    Args:
        net: The torch module whose parameters will be extracted.

    Returns:
        The parameters of the module in a 1-D tensor.
    """
    all_vectors = []
    for p in net.parameters():
        all_vectors.append(p.data.view(-1).detach())
    return torch.cat(all_vectors)


def get_env_spaces(env_name: str, env_config: dict) -> Tuple[Space, Space]:
    dummy_env = gym.make(env_name, **env_config)
    return dummy_env.observation_space, dummy_env.action_space


def get_1D_box_length(s: Box) -> int:
    if (not isinstance(s, Box)) or (len(s.shape) != 1):
        raise ValueError(
            "Expected a 1D Box-typed space, but received: "
            + repr(s)
        )
    return len(s.low)


def get_action_space_length(s: Union[Box, Discrete]):
    if isinstance(s, Box):
        result = get_1D_box_length(s)
    elif isinstance(s, Discrete):
        result = s.n
    else:
        raise TypeError(
            "Don't know how to work with this action space: "
            + repr(s)
        )
    return result
