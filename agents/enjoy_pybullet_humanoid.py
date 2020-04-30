from pgpelib.restore import to_torch_module
from typing import List, Optional
import sys
import torch
from torch import nn
import numpy as np
import gym
import pybullet_envs
from time import sleep
from copy import deepcopy


def evaluate_artifact(artifact_name: str,
                      ntimes: int,
                      sleeping_time: float):

    # Create the environment
    env: gym.Env = gym.make("HumanoidBulletEnv-v0", render=True)

    # Get the stored policy network
    policy_net: nn.Module = to_torch_module(artifact_name)

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

    # Define a manual render function which forces the camera to follow
    # the agent.
    # Note that this nested function contains PyBullet-specific code.
    def render():
        nonlocal env, sleeping_time
        env.render()
        body_x, body_y, _ = env.unwrapped.robot.body_xyz
        env.unwrapped._p.resetDebugVisualizerCamera(3, 0, -40, [body_x + 1.6, body_y, 0])
        sleep(sleeping_time)

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
         sleeping_time: str="0.01"):

    if artifact_name is None or artifact_name == '--help':
        print("Usage:", sys.argv[0], "ARTIFACT_NAME.pickle [N [DT]]")
        print("  Evaluate the agent saved in the specified artifact N times.")
        print("  This visualizer contains PyBullet-specific code.")
        print("  In more details, it forces the camera to follow the agent.")
        print("  By default, N is 1.")
        print("  Between each visualized scene, DT amount of seconds is waited.")
        print("  By default, DT is 0.01")
        print()
    else:
        evaluate_artifact(artifact_name, int(ntimes), float(sleeping_time))


if __name__ == "__main__":
    main(*(sys.argv[1:]))

