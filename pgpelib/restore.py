import numpy as np
import torch
from torch import nn
import pickle
import gym
from gym.spaces import Discrete, Box
from copy import deepcopy
from typing import Union
from os import PathLike
from .policies import Policy
from .runningstat import RunningStat


class ObsNormLayer(nn.Module):
    """Observation normalization layer as a torch module.

    This class is meant to be used for restoring the observation
    normalization behavior of a saved policy.
    Once initialized, it does NOT update its normalization data
    according to the inputs it further receives.

    This module is usually the very first layer of a policy.
    """

    def __init__(self, normalizer: RunningStat):
        """``__init__(...)``: Initialize the ObsNormLayer object.

        Args:
            normalizer: A RunningStat according from which the
                observation normalization data will be drawn.
        """
        nn.Module.__init__(self)
        self._constants = dict(
            stdev=torch.tensor(normalizer.stdev, dtype=torch.float32),
            mean=torch.tensor(normalizer.mean, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input, and return the result

        Args:
            x: The input tensor.
        Returns:
            The normalized output, as a tensor.
        """
        x = x - self._constants['mean']
        x = x / self._constants['stdev']
        return x

    def get_constants(self) -> dict:
        """Get the dictionary in which the normalization data is stored.

        Returns:
            The constants dictionary.
        """
        return self._constants


class ActClipLayer(nn.Module):
    """Action clipping layer as a torch module."""

    def __init__(self, action_space: Box):
        """``__init__(...)``: Initialize the ActClipLayer object.

        Args:
            action_space: Action space according to which the action
                clipping will be done.
        """
        nn.Module.__init__(self)
        self._constants = dict(
            lb=torch.tensor(action_space.low, dtype=torch.float32),
            ub=torch.tensor(action_space.high, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get the clipped copy of the input.

        Args:
            x: The input tensor.
        Returns:
            The clipped output, as a tensor.
        """
        x = torch.min(x, self._constants['ub'])
        x = torch.max(x, self._constants['lb'])
        return x

    def get_constants(self) -> dict:
        """Get the dictionary in which the action space data is stored.

        Returns:
            The constants dictionary.
        """
        return self._constants


def _get_as_policy_object(policy_storage:
                              Union[Policy, PathLike, str]) -> Policy:
    if isinstance(policy_storage, Policy):
        policy = policy_storage
        needs_deepcopy = True
    elif isinstance(policy_storage, (PathLike, str)):
        with open(policy_storage, 'rb') as fpolicy:
            policy = pickle.load(fpolicy)
    else:
        raise TypeError(
            "Expected a Policy object"
            + " or a file name (as a string or a PathLike object);"
            + " but received:"
            + repr(policy_storage)
        )

    return policy


def to_torch_module(policy_storage: Union[Policy, PathLike, str],
                    *,
                    norm_layer: bool=True,
                    clip_layer: bool=True) -> nn.Module:
    """Restore the saved policy to a torch module.

    Args:
        policy_storage: A Policy object, or a file name
            (as a str or a PathLike object).
            If given as a file name, it is assumed that the file name
            points to a pickle file, from which the policy is extracted.
        norm_layer: If given as True, the observation normalization
            mechanism will be added to the resulting torch module
            as a preprocessing layer.
            If given as False, no such normalization layer
            is added even if the stored policy does contain
            normalization data.
        clip_layer: If given as True, the action clipping mechanism
            will be added to the resulting torch module as the
            last layer.
            If given as False, no such clipping layer is added.
    Returns:
        The policy, as a torch module.
    """

    policy: Policy

    needs_deepcopy = False
    if isinstance(policy_storage, Policy):
        needs_deepcopy = True
    policy = _get_as_policy_object(policy_storage)

    has_obs_norm = (
        (policy._main_obs_stats is not None)
        and (policy._main_obs_stats.count > 0)
    )

    has_box_acspace = isinstance(policy._action_space, Box)

    if needs_deepcopy:
        net = deepcopy(policy._policy)
    else:
        net = policy._policy

    layers = []

    if has_obs_norm and norm_layer:
        layers.append(ObsNormLayer(policy._main_obs_stats))
    layers.append(net)
    if has_box_acspace and clip_layer:
        layers.append(ActClipLayer(policy._action_space))

    if len(layers) == 1:
        return layers[0]
    else:
        return nn.Sequential(*layers)


def policy_metadata(policy_storage: Union[Policy, PathLike, str]) -> dict:
    """Restore the saved policy's metadata to a dictionary.
    The metadata contains the gym environment's name ('env_name'),
    and the keyword arguments ('env_config') given to ``gym.make(...)``
    for re-creating the environment.
    This dictionary also contains the 'notes' attribute of the
    policy object.

    Args:
        policy_storage: A Policy object, or a file name
            (as a str or a PathLike object).
            If given as a file name, it is assumed that the file name
            points to a pickle file, from which the metadata is extracted.
    Returns:
        The metadata as a dictionary.
    """

    needs_deepcopy = False
    if isinstance(policy_storage, Policy):
        needs_deepcopy = True
    policy = _get_as_policy_object(policy_storage)

    if needs_deepcopy:
        f = deepcopy
    else:
        def f(something):
            return something

    return dict(
        env_name=f(policy._env_name),
        env_config=f(policy._env_config),
        notes=f(policy.notes)
    )

