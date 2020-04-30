from pgpelib.policies import Policy
from pgpelib.restore import to_torch_module, policy_metadata
from typing import List, Optional
import pickle
import sys
import torch
from torch import nn
import numpy as np
import gym
from copy import deepcopy


def evaluate_artifact(artifact_name: str,
                      ntimes: int):

    # Load the policy
    with open(artifact_name, 'rb') as fpolicy:
        policy_obj: Policy = pickle.load(fpolicy)

    # Get the env name and config
    metadata = policy_metadata(policy_obj)
    env_name = metadata['env_name']
    env_config = metadata['env_config']

    # Create the environment
    env: gym.Env = gym.make(env_name, **env_config)

    # Get the stored policy network
    policy_net: nn.Module = to_torch_module(policy_obj)

    # Define a nested function which takes an observation array,
    # uses the policy network to predict an action,
    # and return that action as a numpy array.
    def use_policy(x: np.ndarray) -> np.ndarray:
        nonlocal policy_net
        with torch.no_grad():
            y = policy_net(
                torch.tensor(x, dtype=torch.float32)
            ).numpy()
        return y

    # The outer loop which executes multiple trajectories
    cumulative_rewards = []
    for _ in range(ntimes):
        cumulative_reward = 0.0

        # Start a new trajectory by sampling the initial observation
        observation = env.reset()
        env.render()

        # Main loop of the trajectory
        while True:
            action = use_policy(observation)
            observation, reward, done, info = env.step(action)
            env.render()
            cumulative_reward += reward
            if done:
                break
        cumulative_rewards.append(cumulative_reward)

    print(cumulative_rewards)


def main(artifact_name: Optional[str]=None,
         ntimes: str="1"):

    if artifact_name is None or artifact_name == '--help':
        print("Usage:", sys.argv[0], "ARTIFACT_NAME.pickle [N]")
        print("  Visualize the agent saved in the specified artifact N times.")
        print("  By default, N is 1.")
        print()
    else:
        evaluate_artifact(artifact_name, int(ntimes))


if __name__ == "__main__":
    main(*(sys.argv[1:]))

