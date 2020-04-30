# Solving reinforcement learning problems using pgpelib with parallelization
# and with observation normalization
# ==========================================================================
# 
# This example demonstrates how to solve locomotion tasks.
# The following techniques are used:
#
# - dynamic population size
# - observation normalization
# - parallelization (using the ray library)
#

# Because we are using both parallelization and observation normalization,
# we will have to synchronize the observation stats between the remote
# workers and the main process.
# We demonstrate how to do this synchronization using ray,
# however the logic is applicable to other parallelization libraries.

from pgpelib import PGPE
from pgpelib.policies import Policy, LinearPolicy, MLPPolicy
from pgpelib.restore import to_torch_module
from pgpelib.runningstat import RunningStat

from typing import Tuple, Iterable
from numbers import Real

import numpy as np
import torch

import gym

import ray
import multiprocessing as mp

from time import sleep

import pickle

# Here is the gym environment to solve.
ENV_NAME = 'Walker2d-v2'

# The environment we are considering to solve is a locomotion problem.
# It defines an "alive bonus" to encourage the agent to stand on its
# feet without falling.
# However, such alive bonuses might drive the evolution process into
# generating agents which focus ONLY on standing on their feet (without
# progressing), just to collect these bonuses.
# We therefore remove this alive bonus by subtracting 1.0 at every
# simulator timestep.
DECREASE_REWARDS_BY = 1.0

# Ray supports stateful parallelization via remote actors.
# An actor is a class instance which lives on different process than
# the main process, and which stores its state.
# Here, we define a remote actor class (which will be instantiated
# multiple times, so that we will be able to use the instances for
# parallelized evaluation of our solutions)
@ray.remote
class Worker:
    def __init__(self, policy, decrease_rewards_by):
        policy: Policy
        self.policy: Policy = policy
        self.decrease_rewards_by = decrease_rewards_by
    
    def set_main_obs_stats(self, rs):
        # Set the main observation stats of the remote worker.
        # The goal of this function is to receive the observation
        # stats from the main process.
        rs: RunningStat
        self.policy.set_main_obs_stats(rs)
    
    def pop_collected_obs_stats(self):
        # Pop the observation stats collected by the worker.
        # At the time of synchronization, the main process will call
        # this method of each remote worker, and update its main
        # observation stats with those collected data.
        return self.policy.pop_collected_obs_stats()
    
    def run(self, d):
        # Run a each solution in the dictionary d.
        # The dictionary d is expected in this format:
        # { solution_index1: solution1,
        #   solution_index2: solution2,
        #   ...                         }
        # and the result will be:
        # { solution_index1: (cumulative_reward1, number_of_interactions1)
        #   solution_index2: (cumulative_reward2, number_of_interactions2)
        #   ...                                                            }
        return self.policy.set_params_and_run_all(
            d,
            decrease_rewards_by=self.decrease_rewards_by
        )


# Set the number of workers to be instantiated as the number of CPUs.
NUM_WORKERS = mp.cpu_count()

# List of workers.
# Initialized as a list containing `None`s in the beginning.
WORKERS = [None] * NUM_WORKERS

def prepare_workers(policy, decrease_rewards_by):
    # Fill the WORKERS list.

    # Initialize the ray library.
    ray.init()

    # For each index i of WORKERS list, fill the i-th element with a new
    # worker instance.
    for i in range(len(WORKERS)):
        WORKERS[i] = Worker.remote(policy, decrease_rewards_by)


Reals = Iterable[Real]
def evaluate_solutions(solutions: Iterable[np.ndarray]) -> Tuple[Reals, Reals]:
    # This function evaluates the given solutions in parallel.

    # Get the number of solutions
    nslns = len(solutions)

    if len(WORKERS) > nslns:
        # If the number of workers is greater than the number of solutions
        # then the workers that we are going to actually use here
        # is the first `nslns` amount of workers, not all of them.
        workers = WORKERS[:nslns]
    else:
        # If the number of solutions is equal to or greater than the
        # number of workers, then we will use all of the instantiated
        # workers.
        workers = WORKERS

    # Number of workers that are going to be used now.
    nworkers = len(workers)

    # To each worker, we aim to send a dictionary, each dictionary being
    # in this form:
    # { solution_index1: solution1, solution_index2: solution2, ...}
    # We keep those dictionaries in the `to_worker` variable.
    # to_worker[i] stores the dictionary to be sent to the i-th worker.
    to_worker = [dict() for _ in range(nworkers)]

    # Iterate over the solutions and assign them one by one to the
    # workers.
    i_worker = 0
    for i, solution in enumerate(solutions):
        to_worker[i_worker][i] = solution
        i_worker = (i_worker + 1) % nworkers

    # Each worker executes the solution dictionary assigned to itself.
    # The results are then collected to the list `worker_results`.
    # The workers do their tasks in parallel.
    worker_results = ray.get(
        [
            workers[i].run.remote(to_worker[i])
            for i in range(nworkers)
        ]
    )

    # Allocate a list for storing the fitnesses, and another list for
    # storing the number of interactions.
    fitnesses = [None] * nslns
    num_interactions = [None] * nslns

    # For each worker:
    for worker_result in worker_results:
        # For each solution and its index mentioned in the worker's
        # result dictionary:
        for i, result in worker_result.items():
            fitness, timesteps = result
            # Store the i-th solution's fitness in the fitnesses list
            fitnesses[i] = fitness
            # Store the i-th solution's number of interactions in the
            # num_interactions list.
            num_interactions[i] = timesteps

    # Return the fitnesses and the number of interactions lists.
    return fitnesses, num_interactions


def sync_obs_stats(main_policy: Policy):
    # This function synchronizes the observation stats of the
    # main process and of the main workers.

    # Collect observation stats from the remote workers
    collected_stats = ray.get(
        [
            worker.pop_collected_obs_stats.remote()
            for worker in WORKERS
        ]
    )

    # In the main process, update the main policy's
    # observation stats with the stats collected from the remote workers.
    for stats in collected_stats:
        main_policy.update_main_obs_stats(stats)

    # To each worker, send the main policy's up-to-date stats.
    ray.get(
        [
            worker.set_main_obs_stats.remote(
                main_policy.get_main_obs_stats()
            )
            for worker in WORKERS
        ]
    )


def main():
    # This is the main function.
    # The main evolution procedure will be defined here.

    # Make a linear policy.
    policy = LinearPolicy(
        env_name=ENV_NAME,  # Name of the environment
        observation_normalization=True
    )

    # Prepare the workers
    prepare_workers(policy, DECREASE_REWARDS_BY)

    # Initial solution
    x0 = np.zeros(policy.get_parameters_count(), dtype='float32')

    # The following are the Walker2d-v2 hyperparameters used in the paper:
    # ClipUp: A Simple and Powerful Optimizer for Distribution-based
    # Policy Evolution
    N = policy.get_parameters_count()
    max_speed = 0.015
    center_learning_rate = max_speed / 2.0
    radius = max_speed * 15
    # Compute the stdev_init from the radius:
    stdev_init = np.sqrt((radius ** 2) / N)
    popsize = 100
    popsize_max = 800

    # Below we initialize our PGPE solver.
    pgpe = PGPE(
        solution_length=N,
        popsize=popsize,
        popsize_max=popsize_max,
        num_interactions=int(popsize * 1000 * (3 / 4)),
        center_init=x0,
        center_learning_rate=center_learning_rate,
        optimizer='clipup',
        optimizer_config={'max_speed': max_speed},
        stdev_init=stdev_init,
        stdev_learning_rate=0.1,
        stdev_max_change=0.2,
        solution_ranking=True,
        dtype='float32'
    )

    num_iterations = 500

    # The main loop of the evolutionary computation
    for i in range(1, 1 + num_iterations):

        total_episodes = 0

        while True:
            # Get the solutions from the pgpe solver
            solutions = pgpe.ask()

            # Evaluate the solutions in parallel and get the fitnesses
            fitnesses, num_interactions = evaluate_solutions(solutions)

            sync_obs_stats(policy)
            
            # Send the pgpe solver the received fitnesses
            iteration_finished = pgpe.tell(fitnesses, num_interactions)

            total_episodes += len(fitnesses)

            if iteration_finished:
                break
        
        print(
            "Iteration:", i,
            "  median score:", np.median(fitnesses),
            "  num.episodes:", total_episodes
        )

    print("Visualizing the center solution...")

    # Get the center solution
    center_solution = pgpe.center.copy()

    # Make the gym environment for visualizing the center solution
    env = gym.make(ENV_NAME)

    # Load the center solution into the policy
    policy.set_parameters(center_solution)

    # Save the policy into a pickle file
    with open(__file__ + '.pickle', 'wb') as f:
        pickle.dump(policy, f)

    # Convert the policy to a PyTorch module
    net = to_torch_module(policy)

    while True:
        print("Please choose: 1> Visualize the agent  2> Quit")
        response = input(">>")

        if response == '1':
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
        elif response == '2':
            break
        else:
            print('Unrecognized response:', repr(response))


if __name__ == "__main__":
    main()

