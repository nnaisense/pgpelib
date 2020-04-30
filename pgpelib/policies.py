import torch
from torch import nn
import gym
from gym.spaces import Box, Discrete, Space
from copy import copy, deepcopy
import numpy as np
from typing import Optional, Union, Iterable, List, Dict, Tuple, Any
from numbers import Real, Integral
from .runningstat import RunningStat
from .misc import (
    fill_parameters,
    get_parameter_vector,
    positive_int_or_none,
    positive_int,
    positive_float,
    get_env_spaces,
    get_1D_box_length,
    get_action_space_length
)


ParamVector = Union[List[Real], np.ndarray]
Action = Union[List[Real], np.ndarray, Integral]


class Policy:
    """Base class for a policy."""

    def __init__(self,
                 *,
                 env_name: str,
                 env_config: Optional[dict]=None,
                 observation_normalization: bool=True,
                 seed: Optional[Integral]=None):
        """``__init__(...)``: Initialize the policy object.
        The initializer must be called from the initializer
        of the inheriting classes.

        Args:
            env_name: Expected as a string specifying the gym
                environment ID (e.g. 'Humanoid-v2').
            env_config: Expected as None, or as a dictionary
                containing the keyword arguments to be passed
                to ``gym.make`` when creating the environment.
            observation_normalization: Expected as boolean,
                specifying whether or not the observations
                are to be normalized.
            seed: Expected as None or as an integer.
                Pass here an integer for explicitly setting a
                random seed for the stochastic operations of
                the gym environment.
        """

        self._policy: nn.Module
        if bool(observation_normalization):
            self._main_obs_stats = RunningStat()
            self._collected_obs_stats = RunningStat()
        else:
            self._main_obs_stats = None
            self._collected_obs_stats = None

        if not isinstance(env_name, str):
            raise TypeError(
                "Environment name was expected as an str,"
                + " but it was received as: "
                + repr(env_name)
            )
        self._env_name = env_name
        if env_config is None:
            self._env_config = {}
        else:
            self._env_config = env_config

        self._env: Optional[gym.Env] = None
        self._observation_space, self._action_space = (
            get_env_spaces(self._env_name, self._env_config)
        )

        self._seed = seed

        self._collect_obs_stats = True

        self.notes: Any = None

    def _get_env(self) -> gym.Env:
        if self._env is None:
            self._env = gym.make(self._env_name, **(self._env_config))
        if self._seed is not None:
            self._env.seed(self._seed)
        return self._env

    def __getstate__(self):
        state = {"_env": None}
        for k, v in self.__dict__.items():
            if k != "_env":
                state[k] = v
        return state

    def __setstate__(self, state):
        state: dict
        for k, v in state.items():
            self.__dict__[k] = v

    def _use_policy(self, observation: Iterable[Real]) -> Action:
        x = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            action = self._policy(x).numpy()
        if isinstance(self._action_space, Box):
            action = np.clip(
                action,
                self._action_space.low,
                self._action_space.high
            )
        elif isinstance(self._action_space, Discrete):
            action = np.argmax(action)
        else:
            raise TypeError(
                "Cannot work with this action space: "
                + repr(self._action_space)
            )
        return action

    def run(self,
            *,
            decrease_rewards_by: Real=0.0,
            max_episode_length: Optional[Integral]=None) -> Tuple[float, int]:
        """Run an episode.

        Args:
            decrease_rewards_by: The reward at each timestep will be
                decreased by this given amount.
            max_episode_length: The maximum number of interactions
                allowed in an episode.
        Returns:
            A tuple (cumulative_reward, number_of_interactions).
        """

        max_episode_length = positive_int_or_none(max_episode_length)

        def normalized(obs):
            if self._main_obs_stats is not None:
                if self._collect_obs_stats:
                    self._main_obs_stats.update(obs)
                    self._collected_obs_stats.update(obs)
                return self._main_obs_stats.normalize(obs)
            else:
                return obs

        t = 0
        cumulative_reward = 0.0
        env = self._get_env()
        observation = env.reset()
        observation = normalized(observation)
        while True:
            action = self._use_policy(observation)
            observation, reward, done, info = env.step(action)
            observation = normalized(observation)
            t += 1
            reward -= decrease_rewards_by
            cumulative_reward += reward
            if max_episode_length is not None and t > max_episode_length:
                break
            if done:
                break

        return cumulative_reward, t

    def set_params_and_run(self,
                           policy_parameters: ParamVector,
                           *,
                           decrease_rewards_by: Real=0.0,
                           max_episode_length: Optional[Integral]=None) -> (
                                Tuple[float, int]):
        """Set the the parameters of the policy by copying them
        from the given parameter vector, then run an episode.

        Args:
            policy_parameters: The policy parameters to be used.
            decrease_rewards_by: The reward at each timestep will be
                decreased by this given amount.
            max_episode_length: The maximum number of interactions
                allowed in an episode.
        Returns:
            A tuple (cumulative_reward, number_of_interactions).
        """

        self.set_parameters(policy_parameters)
        return self.run(
            decrease_rewards_by=decrease_rewards_by,
            max_episode_length=max_episode_length
        )

    def _run_from_list(self,
                       policy_param_list: List[ParamVector],
                       *,
                       decrease_rewards_by: Real=0.0,
                       max_episode_length: Optional[Integral]=None) -> (
                           List[Tuple[float, int]]):

        results = []
        for policy_params in policy_param_list:
            results.append(
                self.set_params_and_run(
                    policy_params,
                    decrease_rewards_by=decrease_rewards_by,
                    max_episode_length=max_episode_length
                )
            )
        return results

    def _run_from_dict(self,
                       policy_param_dict: Dict[Any, ParamVector],
                       *,
                       decrease_rewards_by: Real=0.0,
                       max_episode_length: Optional[Integral]=None) -> (
                           Dict[Any, Tuple[float, int]]):

        results = {}
        for policy_key, policy_params in policy_param_dict.items():
            results[policy_key] = (
                self.set_params_and_run(
                    policy_params,
                    decrease_rewards_by=decrease_rewards_by,
                    max_episode_length=max_episode_length
                )
            )
        return results

    def set_params_and_run_all(self,
                               policy_params_all: Union[
                                   List[ParamVector],
                                   Dict[Any, ParamVector]
                               ],
                               *,
                               decrease_rewards_by: Real=0.0,
                               max_episode_length: Optional[Integral]=None) -> (
                                   Union[
                                       List[Tuple[float, int]],
                                       Dict[Any, Tuple[float, int]]
                                   ]
                               ):
        """For each of the items in the given parameters dictionary,
        set the the parameters of the policy by copying them
        from the given parameter vector, then run an episode.

        Args:
            policy_params_all: A dictionary, mapping a policy identifier
                to a policy parameter vector.
                For example, the policy identifier here could possibly
                be an integer specifying the index of the
                parameter vector within a batch of parameter vectors.
            decrease_rewards_by: The reward at each timestep will be
                decreased by this given amount.
            max_episode_length: The maximum number of interactions
                allowed in an episode.
        Returns:
            A dictionary where each item maps the policy identifier key
            to a tuple (cumulative_reward, number_of_interactions).
        """
        kwargs = dict(
            decrease_rewards_by=decrease_rewards_by,
            max_episode_length=max_episode_length
        )

        received_dict = (
            hasattr(policy_params_all, "keys")
            and hasattr(policy_params_all, "values")
        )
        
        if received_dict:
            return self._run_from_dict(policy_params_all, **kwargs)
        else:
            return self._run_from_list(policy_params_all, **kwargs)

    def set_parameters(self, parameters: ParamVector):
        """Set the parameters of the policy by copying the values
        from the given parameter vector.

        Args:
            parameters: The parameter vector.
        """
        #x = torch.as_tensor(parameters, dtype=torch.float32)
        if isinstance(parameters, np.ndarray):
            parameters = parameters.copy()
        x = torch.as_tensor(parameters, dtype=torch.float32)
        fill_parameters(self._policy, x)

    def get_parameters(self) -> np.ndarray:
        """Get the parameters of the policy as a 1-D numpy array.

        Returns:
            The parameter vector.
        """
        return get_parameter_vector(self._policy).numpy()

    def pop_collected_obs_stats(self) -> RunningStat:
        """Get the collected observation statistics.
        When this method is called, the contained collected
        statistics are removed.

        Returns:
            The collected observation statistics.
        """
        if self._collected_obs_stats is None:
            raise ValueError(
                "Observation stats are not configured to be collected,"
                " therefore, they cannot be popped."
            )

        result = self._collected_obs_stats
        self._collected_obs_stats = RunningStat()
        return result

    def set_main_obs_stats(self, obs_stats: RunningStat):
        """Set the observation statistics to be used for
        observation normalization.

        Args:
            obs_stats: A RunningStat object containing the statistics.
        """
        if obs_stats is None:
            raise ValueError(
                "The main observation stats cannot be given as None."
            )
        self._main_obs_stats = deepcopy(obs_stats)

    def get_main_obs_stats(self) -> Optional[RunningStat]:
        """Get the observation statistics used for
        observation normalization.

        Returns:
            A RunningStat object containing the statistics.
        """
        return self._main_obs_stats

    def update_main_obs_stats(self, obs_stats: Union[RunningStat, np.ndarray]):
        """Update the observation statistics used for
        observation normalization.

        Args:
            obs_stats: A RunningStat object or a numpy array
                (a numpy array representing a single observation vector).
        """
        if self._main_obs_stats is None:
            raise ValueError(
                "There is no observation stats to update."
                + " Was "
                + repr(self)
                + " initialized with observation_normalization=False?"
            )
        self._main_obs_stats.update(obs_stats)

    def get_parameters_count(self) -> int:
        """Get the number of parameters of the policy
        (also corresponds to the length of parameter vector).
        """
        return len(self.get_parameters())

    def get_collect_obs_stats(self) -> bool:
        """Get, as boolean, whether or not the policy is configured
        to collect observation statistics when running episodes.

        Returns:
            A boolean.
        """
        return self._collect_obs_stats

    def set_collect_obs_stats(self, b: bool):
        """Set, as boolean, whether or not the policy is to collect
        observation statistics when running episodes.

        Args:
            b: A boolean.
        """
        self._collect_obs_stats = bool(b)


class LinearPolicy(Policy):
    """A linear policy."""

    def __init__(self,
                 *,
                 env_name: str,
                 env_config: Optional[dict]=None,
                 observation_normalization: bool=True,
                 seed: Optional[Integral]=None,
                 bias: bool=True):
        """``__init__(...)``: Initialize the linear policy.

        Args:
            env_name: Expected as a string specifying the gym
                environment ID (e.g. 'Humanoid-v2').
            env_config: Expected as None, or as a dictionary
                containing the keyword arguments to be passed
                to ``gym.make`` when creating the environment.
            observation_normalization: Expected as boolean,
                specifying whether or not the observations
                are to be normalized.
            seed: Expected as None or as an integer.
                Pass here an integer for explicitly setting a
                random seed for the stochastic operations of
                the gym environment.
            bias: Expected as a boolean, specifying whether or
                not the linear policy will have bias parameters.
        """

        Policy.__init__(
            self,
            env_name=env_name,
            env_config=env_config,
            observation_normalization=observation_normalization,
            seed=seed
        )

        obs_length = get_1D_box_length(self._observation_space)
        act_length = get_action_space_length(self._action_space)
        self._policy = nn.Linear(obs_length, act_length, bias=bias)


class MLPPolicy(Policy):
    """A multi-layer perceptron policy."""

    ACTIVATION_CLS = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU
    }

    def __init__(self,
                 *,
                 env_name: str,
                 env_config: Optional[dict]=None,
                 observation_normalization: bool=True,
                 seed: Optional[Integral]=None,
                 hidden_size: Integral=64,
                 num_hidden: Integral=1,
                 hidden_activation: str="tanh",
                 output_activation: Optional[str]=None):
        """
        Args:
            env_name: Expected as a string specifying the gym
                environment ID (e.g. 'Humanoid-v2').
            env_config: Expected as None, or as a dictionary
                containing the keyword arguments to be passed
                to ``gym.make`` when creating the environment.
            observation_normalization: Expected as boolean,
                specifying whether or not the observations
                are to be normalized.
            seed: Expected as None or as an integer.
                Pass here an integer for explicitly setting a
                random seed for the stochastic operations of
                the gym environment.
            hidden_size: Expected as an integer, specifying
                the number of neurons in a hidden layer.
            num_hidden: Expected as an integer, specifying
                the number of hidden layers.
            hidden_activation: The activation function to be
                used by the hidden layer(s).
                Expected as 'tanh' or 'relu'.
            output_activation: Optional. The activation function
                to be used by the output layer.
                Can be given as 'tanh' or 'relu', or can be left
                as None.
        """

        Policy.__init__(
            self,
            env_name=env_name,
            env_config=env_config,
            observation_normalization=observation_normalization,
            seed=seed
        )

        obs_length = get_1D_box_length(self._observation_space)
        act_length = get_action_space_length(self._action_space)

        hidden_size = positive_int(hidden_size)
        num_hidden = positive_int(num_hidden)

        if hidden_activation is None:
            hidden_act_cls = None
        else:
            hidden_act_cls = self.ACTIVATION_CLS[hidden_activation]

        if output_activation is None:
            output_act_cls = None
        else:
            output_act_cls = self.ACTIVATION_CLS[output_activation]

        layers = []

        # first hidden layer
        layers.append(nn.Linear(obs_length, hidden_size))
        if hidden_act_cls is not None:
            layers.append(hidden_act_cls())

        # rest of the hidden layers (if any)
        for _ in range(1, num_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if hidden_act_cls is not None:
                layers.append(hidden_act_cls())

        # output layer
        layers.append(nn.Linear(hidden_size, act_length))
        if output_act_cls is not None:
            layers.append(output_act_cls())

        self._policy = nn.Sequential(*layers)

