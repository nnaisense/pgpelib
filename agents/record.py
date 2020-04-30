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
import matplotlib.pyplot as plt


def evaluate_artifact(artifact_name: str,
                      ntimes: int,
                      frame_interval: int):

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

    img_counter = 0
    scene = 1
    def render():
        nonlocal scene, env, img_counter
        if img_counter == 0:
            img = env.render(mode='rgb_array')
            plt.imsave(str(scene).rjust(4, '0') + '.png', img)
            scene += 1
        img_counter = (img_counter + 1) % frame_interval

    # The outer loop which executes multiple trajectories
    cumulative_rewards = []
    for _ in range(ntimes):
        cumulative_reward = 0.0

        # Start a new trajectory by sampling the initial observation
        observation = env.reset()
        render()

        # Main loop of the trajectory
        while True:
            action = use_policy(observation)
            observation, reward, done, info = env.step(action)
            render()
            cumulative_reward += reward
            if done:
                break
        cumulative_rewards.append(cumulative_reward)

    print(cumulative_rewards)


def main(artifact_name: Optional[str]=None,
         ntimes: str="1",
         frame_interval: str="1"):

    if artifact_name is None or artifact_name == '--help':
        print("Usage:", sys.argv[0], "ARTIFACT_NAME.pickle [N [INTERVAL]]")
        print("  Record the agent's running into images over N episodes.")
        print("  The output files are 0001.png, 0002.png, ...")
        print("  By default, N is 1.")
        print("  There can be frame skipping if INTERVAL is set as 2 or more.")
        print("  By default, INTERVAL is set as 1, meaning every single frame")
        print("  is to be recorded.")
        print("  For example, setting INTERVAL as 2 means one in every 2 frames")
        print("  is to be recorded.")
        print()
    else:
        evaluate_artifact(artifact_name, int(ntimes), int(frame_interval))


if __name__ == "__main__":
    main(*(sys.argv[1:]))

