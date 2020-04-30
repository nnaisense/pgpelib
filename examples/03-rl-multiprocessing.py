# Solving reinforcement learning problems using pgpelib with parallelization
# ==========================================================================
# 
# In this example, we are going to solve the `CartPole-v1` environment, and
# we will also use parallelization to decrease the execution time of the
# evolution.
# 
# `pgpelib` is agnostic when it comes to parallelization: the choice of
# parallelization library is left to the user.
#
# In the case of this example, we use `multiprocessing`.


from pgpelib import PGPE
from pgpelib.policies import LinearPolicy, MLPPolicy
from pgpelib.restore import to_torch_module

import numpy as np
import torch

import gym

import multiprocessing as mp


ENV_NAME = 'CartPole-v1'


policy = MLPPolicy(
    env_name=ENV_NAME,   # Name of the environment
    num_hidden=1,        # Number of hidden layers
    hidden_size=8,       # Size of a hidden layer
    hidden_activation='tanh',   # Activation function of the hidden layer
    
    # Whether or not to do online normalization on the observations
    # received from the environments.
    # In this tutorial, we set it to False just to keep things simple.
    # Note that, with observation_normalization, we would need to
    # synchronize the observation stats between the main process and the
    # worker processes.
    observation_normalization=False
)


def evaluate_solution(solution: np.ndarray):
    global policy
    fitness, _ = policy.set_params_and_run(solution)
    return fitness


def main():
    # Initial solution
    x0 = np.zeros(policy.get_parameters_count(), dtype='float32')

    # Below we initialize our PGPE solver.
    pgpe = PGPE(
        solution_length=policy.get_parameters_count(),
        popsize=250,
        center_init=x0,
        center_learning_rate=0.075,
        optimizer='clipup',
        optimizer_config={'max_speed': 0.15},
        stdev_init=0.08,
        stdev_learning_rate=0.1,
        stdev_max_change=0.2,
        solution_ranking=True,
        dtype='float32'
    )

    # Here, we make a pool of worker processes.
    # With the help of these workers, we aim to parallelize the
    # evaluation of the solutions.
    with mp.Pool(processes=mp.cpu_count()) as pool:
        num_iterations = 50

        # The main loop of the evolutionary computation
        for i in range(1, 1 + num_iterations):

            # Get the solutions from the pgpe solver
            solutions = pgpe.ask()

            # Evaluate the solutions in parallel and get the fitnesses
            fitnesses = pool.map(evaluate_solution, solutions)
            
            # Send the pgpe solver the received fitnesses
            pgpe.tell(fitnesses)
            
            print("Iteration:", i, "  median score:", np.median(fitnesses))

    print("Visualizing the center solution...")

    # Get the center solution
    center_solution = pgpe.center.copy()

    # Make the gym environment for visualizing the center solution
    env = gym.make(ENV_NAME)

    # Convert the center solution to a PyTorch module
    policy.set_parameters(center_solution)
    net = to_torch_module(policy)

    cumulative_reward = 0.0

    # Reset the environment, and get the observation of the initial
    # state into a variable.
    observation = env.reset()

    # Visualize the initial state
    env.render()

    # Main loop of the trajectory
    while True:
        with torch.no_grad():
            action = net(
                torch.as_tensor(observation, dtype=torch.float32)
            ).numpy()

        if isinstance(env.action_space, gym.spaces.Box):
            interaction = action
        elif isinstance(env.action_space, gym.spaces.Discrete):
            interaction = int(np.argmax(action))
        else:
            assert False, "Unknown action space"

        observation, reward, done, info = env.step(interaction)
        env.render()

        cumulative_reward += reward

        if done:
            break

    print("cumulative_reward", cumulative_reward)


if __name__ == "__main__":
    main()

