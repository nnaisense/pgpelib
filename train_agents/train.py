import math
import numpy as np
from typing import List, Tuple, Optional, Any
from numbers import Integral, Real
from pgpelib import PGPE
from pgpelib.policies import MLPPolicy, LinearPolicy
from sacred import Experiment
from sacred.run import Run
from datetime import datetime
from collections import OrderedDict
import ray
import pickle
import os


class DontExperiment:
    @staticmethod
    def useless_decorator(f):
        return f

    config = useless_decorator
    capture = useless_decorator
    command = useless_decorator
    automain = useless_decorator
    main = useless_decorator


def datetime_to_str(dt: datetime) -> str:
    """Get a timestamp string from a datetime object.

    Args:
        dt: The datetime object.
    Returns:
        The timestamp string.
    """
    return dt.strftime("%Y_%m_%d_%H_%M_%S")


def simplified_env_name(env_name: str) -> str:
    """Get the simplified version of a gym environment name.
    In more details, if the given environment name
    contains the name of the exporting module,
    that module information is removed.
    For example: 'mymodule:MyEnv-v0' becomes 'MyEnv-v0'.

    Args:
        env_name: The gym environment name.
    Returns:
        The environment name, with its module part removed.
    """
    assert isinstance(env_name, str)
    where = env_name.find(":")
    if where >= 0:
        env_name = env_name[where + 1:]
    return env_name


if __name__ == "__main__":
    ex = Experiment()
else:
    ex = DontExperiment()


@ex.config
def config():
    env_name = "tinytraj_humanoid_bullet:TinyTrajHumanoidBulletEnv-v0"
    obs_norm = True              # Whether or not to use observation normalization

    niters = 1000                # The optimization is to last this many iterations (generations)
    popsize = 10000              # (Base) Population size
    popsize_max = 80000          # Upper bound for population size
    num_interactions = int(popsize * 200 * (3 / 4))    # Nr.of interactions expected from a generation (-1 disables dynamic popsize)

    max_speed = 0.15             # Maximum speed for ClipUp
    center_lr = max_speed / 2    # Learning rate for when updating the center solution (ClipUp's step size)
    radius = max_speed * 15      # Initial radius for the search distribution
    stdev_lr = 0.1               # Learning rate for when updating the standard deviation vector

    hidden_size = 64  # Nr.of neurons in a hidden layer of the policy.
    num_hidden = 1    # Nr.of layers. Give 0 or less for using linear policy.

    decrease_rewards_by = 0.0    # Decrease each reward by this amount
    max_episode_length = -1      # Limit the number of interactions in an episode to this value

    save_interval = 10           # Given as an integer n, save the current solution at every n generations

    re_eval_interval = 10        # Given as an integer n, re-eval the current solution at every n generations
    re_eval_num_episodes = 16    # During the re-evaluation of the current solution, use this many episodes
    re_eval_env_name = "pybullet_envs:HumanoidBulletEnv-v0"   # Do the re-evaluation in this environment. (empty string means same with env_name)

    num_cpus = -1   # When given as a positive integer n, ray will be initialized with num_cpus=n
    optimizer = 'clipup'  # The optimizer to use. 'clipup' or 'adam'
    solution_ranking = True  # Whether or not to rank the solutions by their fitnesses


@ray.remote
class ContainerActor:
    def __init__(self, contained_obj: Any):
        self._contained = contained_obj
    
    def call(self,
             method_name: str,
             args: list,
             kwargs: dict) -> Any:
        f = getattr(self._contained, method_name)
        return f(*args, **kwargs)
    
    def get(self, attr_name: str) -> Any:
        return getattr(self._contained, attr_name)
    
    def set(self, attr_name: str, value: Any):
        setattr(self._contained, attr_name, value)


@ex.automain
def main(_config: dict, _run: Run):
    if not ray.is_initialized():
        if _config['num_cpus'] > 0:
            ray.init(num_cpus=_config['num_cpus'])
        else:
            ray.init()

    if _config['hidden_size'] > 0 and _config['num_hidden'] > 0:
        policy_class = MLPPolicy
        policy_init = dict(
            env_name=_config['env_name'],
            observation_normalization=_config['obs_norm'],
            hidden_size=_config['hidden_size'],
            num_hidden=_config['num_hidden']
        )
    else:
        policy_class = LinearPolicy
        policy_init = dict(
            env_name=_config['env_name'],
            observation_normalization=_config['obs_norm']
        )

    print()
    print("Policy type:", policy_class)
    print("Policy config:", policy_init)
    print()

    main_policy = policy_class(**policy_init)
    N = main_policy.get_parameters_count()

    if _config['optimizer'] == 'clipup':
        optimizer_config = dict(
            max_speed=_config['max_speed']
        )
    else:
        optimizer_config = {}

    pgpe = PGPE(
        solution_length=N,
        popsize=_config['popsize'],
        optimizer=_config['optimizer'],
        optimizer_config=optimizer_config,
        center_learning_rate=_config['center_lr'],
        stdev_learning_rate=_config['stdev_lr'],
        stdev_init=math.sqrt((_config['radius'] ** 2) / N),
        num_interactions=_config['num_interactions'],
        popsize_max=_config['popsize_max'],
        solution_ranking=_config['solution_ranking']
    )

    ncpus = 0
    for node in ray.nodes():
        resources = node['Resources']
        ncpus += resources['CPU']
    ncpus = math.floor(ncpus)

    actor_seed = 1
    actors = []
    for _ in range(ncpus):
        actor_init = {"seed": actor_seed}
        actor_init.update(policy_init)
        actors.append(ContainerActor.remote(policy_class(**actor_init)))
        actor_seed += 1
    print("Number of actors:", len(actors))

    re_eval_actors = []
    for _ in range(_config['re_eval_num_episodes']):
        actor_init = {"seed": actor_seed}
        actor_init.update(policy_init)
        actor_init["env_name"] = (
            _config["re_eval_env_name"]
            if len(_config["re_eval_env_name"]) > 0
            else _config["env_name"]
        )
        re_eval_actors.append(ContainerActor.remote(policy_class(**actor_init)))
        actor_seed += 1

    if len(re_eval_actors) > 0:
        ray.get(
            [
                re_eval_actor.call.remote("set_collect_obs_stats", [False], {})
                for re_eval_actor in re_eval_actors
            ]
        )

    def evaluate_solutions(solutions: List[np.ndarray],
                           *,
                           decrease_rewards_by: Real=0.0,
                           max_episode_length: Optional[Integral]=None) -> (
                                List[Tuple[Real, Integral]]
                           ):
        nonlocal actors
        nonlocal main_policy
        
        num_solutions = len(solutions)
        num_actors = len(actors)
        
        actor_solutions = [dict() for _ in range(num_actors)]
        
        i_actor = 0
        for i_solution, solution in enumerate(solutions):
            actor_solutions[i_actor][i_solution] = solution
            i_actor = (i_actor + 1) % num_actors
        
        actor_tasks = []
        for i_actor, actor in enumerate(actors):
            if len(actor_solutions[i_actor]) == 0:
                break
            actor_tasks.append(
                actor.call.remote(
                    "set_params_and_run_all",
                    [actor_solutions[i_actor]],
                    dict(
                        decrease_rewards_by=decrease_rewards_by,
                        max_episode_length=max_episode_length
                    )
                )
            )
        
        results = [None for _ in range(num_solutions)]
        for task in actor_tasks:
            task_results = ray.get(task)
            for sln_index, eval_result in task_results.items():
                results[sln_index] = eval_result
        
        return results

    def sync_obs_stats():
        nonlocal actors
        nonlocal main_policy
        
        popped_stats = ray.get(
            [
                actor.call.remote("pop_collected_obs_stats", [], {})
                for actor in actors
            ]
        )
        
        for popped in popped_stats:
            main_policy.update_main_obs_stats(popped)
        
        updated_stats = main_policy.get_main_obs_stats()
        
        ray.get(
            [
                actor.call.remote("set_main_obs_stats", [updated_stats], {})
                for actor in actors
            ]
        )

    now = datetime.now()
    fname_prefix = "_".join(
        [
            datetime_to_str(now),
            simplified_env_name(_config['env_name']),
            str(os.getpid())
        ]
    ) + "_"
    fname_suffix = ".pickle"
    def save_artifact(solution: np.ndarray, artifact_name: str, notes: Any=None) -> str:
        nonlocal main_policy, fname_prefix, fname_suffix, _run
        main_policy.set_parameters(solution)
        main_policy.notes = notes
        fname = fname_prefix + artifact_name + fname_suffix
        with open(fname, "wb") as f:
            pickle.dump(main_policy, f)
        _run.add_artifact(fname)

    evolution_begintime = datetime.now()

    best_ever = None
    best_ever_fitness = None

    total_timesteps = 0
    for generation in range(1, 1 + _config['niters']):
        generation_begintime = datetime.now()
        pop_total_timesteps = 0
        while True:
            solutions = pgpe.ask()
            evaluations = evaluate_solutions(
                solutions,
                decrease_rewards_by=_config['decrease_rewards_by'],
                max_episode_length=_config['max_episode_length']
            )
            if _config['obs_norm']:
                sync_obs_stats()
            fitnesses = []
            timesteps = []

            for evaluation in evaluations:
                fitness, t = evaluation
                fitnesses.append(fitness)
                timesteps.append(t)
                pop_total_timesteps += t
                total_timesteps += t
            
            if pgpe.num_interactions is not None:
                done = pgpe.tell(fitnesses, timesteps)
            else:
                done = pgpe.tell(fitnesses)
            
            if done:
                break
        
        pop_best = None
        pop_best_fitness = None
        pop_worst = None
        pop_worst_fitness = None
        population = []
        pop_fitnesses = []
        for sln, sln_fit in pgpe:
            population.append(sln)
            pop_fitnesses.append(sln_fit)
            if pop_best is None or sln_fit > pop_best_fitness:
                pop_best = sln
                pop_best_fitness = sln_fit
            if pop_worst is None or sln_fit < pop_worst_fitness:
                pop_worst = sln
                pop_worst_fitness = sln_fit
            if best_ever is None or sln_fit > best_ever_fitness:
                best_ever = sln
                best_ever_fitness = sln_fit
        
        center_solution = pgpe.center

        median_fitness = np.median(pop_fitnesses)
        mean_fitness = np.mean(pop_fitnesses)

        generation_endtime = datetime.now()
        generation_elapsed = (
            (generation_endtime - generation_begintime).total_seconds()
        )

        status = OrderedDict(
            generation=generation,
            pop_median=median_fitness,
            pop_mean=mean_fitness,
            popsize=len(population),
            pop_timesteps=pop_total_timesteps,
            pop_elapsed=generation_elapsed,
            elapsed=(datetime.now() - evolution_begintime).total_seconds(),
            timesteps=total_timesteps,
            stdev_min=np.min(pgpe.stdev),
            stdev_max=np.max(pgpe.stdev),
            stdev_mean=np.mean(pgpe.stdev)
        )

        print()
        for status_key, status_value in status.items():
            print(status_key, ":", status_value)
            _run.log_scalar(status_key, status_value)
        _run.result = float(median_fitness)

        # save the center solution at every _config['save_interval']
        if generation % _config['save_interval'] == 0:
            save_artifact(center_solution, "iter" + str(generation))

        # re-evaluate the center solution at every _config['re_eval_interval']
        if _config['re_eval_interval'] > 0:
            if generation % _config['re_eval_interval'] == 0:
                if _config['obs_norm']:
                    obs_stats_for_re_eval = main_policy.get_main_obs_stats()
                    ray.get(
                        [
                            re_eval_actor.call.remote(
                                "set_main_obs_stats",
                                [obs_stats_for_re_eval],
                                {}
                            )
                            for re_eval_actor in re_eval_actors
                        ]
                    )

                re_eval_results = ray.get(
                    [
                        re_eval_actor.call.remote(
                            "set_params_and_run", [pgpe.center], {}
                        )
                        for re_eval_actor in re_eval_actors
                    ]
                )

                re_eval_fitnesses = [result[0] for result in re_eval_results]

                re_eval_metrics = dict(
                    re_eval_mean=np.mean(re_eval_fitnesses),
                    re_eval_median=np.median(re_eval_fitnesses),
                    re_eval_min=np.min(re_eval_fitnesses),
                    re_eval_max=np.max(re_eval_fitnesses),
                    re_eval_generation=generation,
                    re_eval_elapsed=(
                        (datetime.now() - evolution_begintime).total_seconds()
                    ),
                    re_eval_timesteps=total_timesteps
                )
                print()
                print("Re-evaluation {")
                for k, v in re_eval_metrics.items():
                    _run.log_scalar(k, v)
                    print("   ", k, ":", v)
                print("}")

    return float(median_fitness)

