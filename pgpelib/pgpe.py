import numpy as np
from numpy.random import RandomState
from typing import Union, List, Tuple, Dict, Iterable, Callable, Optional, Type
from numbers import Integral, Real
from .optimizers import Adam, ClipUp, Optimizer
from .ranking import compute_centered_ranks
from .misc import (
    make_vector,
    readonly_view,
    positive_float_or_none,
    positive_int_or_none,
    positive_float,
    positive_int,
    non_negative_float,
    non_negative_int
)


def compute_reinforce_update(solutions: List[np.ndarray],
                             fitnesses: np.ndarray,
                             scaled_noises: List[np.ndarray],
                             stdev: np.ndarray,
                             baseline: Union[Real, str]='average') -> (
                                 Tuple[np.ndarray, np.ndarray]):
    """Compute and return the updates for the center point
    and the standard deviation array using the REINFORCE update
    rules. This update rule was proposed in Williams (1992),
    and later used by the search algorithm called
    policy gradient with parameter-based exploration
    (PGPE; Sehnke et al. (2010)).

    References::

        Ronald J. Williams (1992).
        Simple statistical gradient-following algorithms for
        connectionist reinforcement learning.

        Frank Sehnke, Christian Osendorfer, Thomas Ruckstiess,
        Alex Graves, Jan Peters, Jurgen Schmidhuber (2010).
        Parameter-exploring Policy Gradients.

    Args:
        solutions: A list of numpy arrays, each being a solution
            (i.e. a parameter vector).
        fitnesses: A numpy array in which i-th number represents
            the fitness of the i-th solution.
        scaled_noises: A list of numpy arrays where the i-th array
            is the noise (added to the center of the search distribution)
            used for generating the i-th solution.
        stdev: The standard deviation array which describes the wideness
            of the current search distribution of PGPE.
        baseline: Pass here the string 'average' for using the average
            fitness as the baseline, or provide a real number for using
            a baseline constant.
    """

    if baseline == 'average':
        baseline = np.mean(fitnesses)
    else:
        baseline = float(baseline)

    dtype = solutions[0].dtype

    popsize = len(solutions)
    if popsize != len(fitnesses):
        raise ValueError(
            "The fitnesses sequence has a different length than"
            " the solutions sequence."
        )

    ndirs = popsize // 2

    all_scores = []
    all_avg_scores = []
    for i in range(0, popsize, 2):
        fit1 = fitnesses[i]
        fit2 = fitnesses[i + 1]
        all_scores.append(fit1 - fit2)
        all_avg_scores.append((fit1 + fit2) / 2.0)

    stdev_sq = stdev ** 2.0

    total_mu = 0.0
    total_sigma = 0.0
    for i_direction in range(ndirs):
        scaled_noise = scaled_noises[i_direction]
        direction_score = all_scores[i_direction]
        avg_dir_score = all_avg_scores[i_direction]
        total_mu += scaled_noise * direction_score * 0.5
        total_sigma += (
            (avg_dir_score - baseline)
            *
            (
                ((scaled_noise ** 2.0) - stdev_sq)
                / stdev
            )
        )

    update_mu = np.asarray(total_mu / ndirs, dtype=dtype)
    update_sigma = np.asarray(total_sigma / ndirs, dtype=dtype)

    return update_mu, update_sigma



class PGPE:
    """The PGPE (Policy gradient with parameter-based exploration)
    algorithm.

    Reference::

        Frank Sehnke, Christian Osendorfer, Thomas Ruckstiess,
        Alex Graves, Jan Peters, Jurgen Schmidhuber (2010).
        Parameter-exploring Policy Gradients.
    """
    def __init__(self,
                 *,
                 solution_length: Integral,
                 popsize: Integral,
                 center_init: Optional[np.ndarray]=None,
                 optimizer: Optional[str]=None,
                 optimizer_config: dict={},
                 popsize_max: Optional[Integral]=None,
                 num_interactions: Optional[Integral]=None,
                 dtype: Union[np.dtype, str]="float32",
                 center_learning_rate: Real=0.15,
                 stdev_learning_rate: Real=0.1,
                 stdev_init: Union[Real, Iterable[Real]]=0.1,
                 stdev_max_change: Optional[Real]=0.2,
                 solution_ranking: bool=True,
                 seed: Optional[Integral]=None):
        """``__init__(...)``: Initialize the PGPE instance.

        Args:
            solution_length: The length of a solution vector.
            popsize: The population size.
            center_init: The starting point in the search space.
            optimizer: Possible values: None, 'clipup' or 'adam'.
            optimizer_config: The configuration dictionary for the optimizer.
                For 'clipup', this dictionary is expected to contain
                the keys: 'momentum' and 'max_speed'.
                For 'adam', this dictionary is expected to contain
                the keys: 'epsilon', 'beta1' and 'beta2'.
            popsize_max: Expected as an integer, or as None.
                If given as an integer, and if the adaptive population
                is activated (see the argument 'num_interactions'),
                imposes an upper bound on the population size.
            num_interactions: Expected as an integer, or as None.
                If given as an integer, activates the adaptive
                population size for PGPE, meaning that
                the population size gets increased
                (by popsize each time), until this specified
                number of interactions (with the simulator)
                is reached in total.
                When the adaptive population size is activated,
                please note that:
                (i) a single population/generation may require
                multiple ask(...) and tell(...) calls;
                (ii) a generation is declared finished and the center point
                is updated only when the num_interactions is reached or
                popsize_max is reached (in which case tell(...) returns True);
                and (iii) the tell(...) method will expect the number of
                simulator interactions as the second argument.
            dtype: The numpy datatype information for the search space.
                Expected as a string (e.g. 'float32') or as a numpy
                dtype object (e.g. np.dtype('float32')).
            center_learning_rate: The learning rate (or the step size)
                for when updating the center of the search distribution.
            stdev_learning_rate: The learning rate for when updating the
                wideness of the search distribution.
            stdev_init: The initial standard deviation.
                Can be provided as a single real number, or as an iterable
                of real numbers.
            stdev_max_change: The maximum change allowed for when updating the
                standard deviation array of the search distribution.
                The default is 0.2, meaning that the standard deviation of the
                search distribution is not allowed to change more than the
                20% of its original value (further changes are clipped).
            solution_ranking: Whether to do a zero-centered solution ranking
                or not, as done in OpenAI's evolution strategy implementation
                reported in Salimans et al (2017).
            seed: Expected as an integer or as None.
                Provide here an integer for explicitly setting the random seed
                for the stochastic operators of PGPE.
        """
        
        self._length = positive_int(solution_length)
        self._popsize = positive_int(popsize)        
        if (self._popsize % 2) == 1:
            raise ValueError(
                "This PGPE implementation uses symmetric populations."
                + " Therefore, an even population size is needed."
                + " But the received population size is: "
                + repr(self._popsize)
            )
        self._num_directions = self._popsize // 2
        self._popsize_max = positive_int_or_none(popsize_max)
        self._num_interactions = positive_int_or_none(num_interactions)
        
        if isinstance(dtype, str):
            self._dtype = np.dtype(dtype)
        else:
            self._dtype = dtype
        
        self._center_learning_rate = positive_float(center_learning_rate)
        self._stdev_learning_rate = non_negative_float(stdev_learning_rate)
        self._stdev = make_vector(stdev_init, self._length, self._dtype)
        self._stdev_max_change = positive_float_or_none(stdev_max_change)
        self._solution_ranking = bool(solution_ranking)
        
        if optimizer is None or optimizer == "":
            self._optimizer = None
        else:
            if optimizer == "clipup":
                optimizer_cls = ClipUp
            elif optimizer == "adam":
                optimizer_cls = Adam
            else:
                raise ValueError("Unknown optimizer:" + repr(optimizer))
            self._optimizer = optimizer_cls(
                stepsize=self._center_learning_rate,
                solution_length=self._length,
                dtype=self._dtype,
                **optimizer_config
            )
        
        self._center = make_vector(0, self._length, self._dtype)
        if center_init is not None:
            self._center[:] = center_init

        if seed is None:
            self._rndgen = RandomState()
        else:
            self._rndgen = RandomState(seed)
        
        self._total_interactions: int = 0
        self._solutions: List[np.ndarray] = []
        self._noises: List[np.ndarray] = []
        self._scaled_noises: List[np.ndarray] = []
        self._fitnesses: List[Real] = []
        self._interactions: List[Real] = []
        self._clear_when_asked = True
            
        self._iteration = 0

    def _clear_population(self):
        self._total_interactions = 0
        self._solutions = []
        self._noises = []
        self._scaled_noises = []
        self._fitnesses = []
        self._interactions = []
            
    def _increase_population(self):
        new_ones = []

        for i in range(self._num_directions):
            noise = self._rndgen.randn(self._length).astype(self._dtype)
            scaled_noise = self._stdev * noise
            solution = self._center + scaled_noise
            mirror = self._center - scaled_noise
            
            self._noises.append(noise)
            self._scaled_noises.append(scaled_noise)
            
            self._solutions.append(solution)
            self._solutions.append(mirror)
            
            new_ones.append(readonly_view(solution))
            new_ones.append(readonly_view(mirror))
            
        return new_ones
               
    def ask(self) -> List[np.ndarray]:
        """Produce and return a batch of solutions for evaluating.

        Returns:
            A list of numpy arrays, each array representing
            a new solution.
        """
        if self._clear_when_asked:
            self._clear_population()
        return self._increase_population()
    
    def tell(self,
             fitnesses: Iterable[Real],
             interactions: Optional[Iterable[Integral]]=None) -> bool:
        """Tell the PGPE instance the fitnesses of the solutions
        received via ask(...).

        Args:
            fitnesses: The fitnesses as an iterable of real numbers,
                where the i-th real number represents the fitness value
                of the i-th solution of the most recent batch produced
                by the method ask(...).
            interactions: To be given as an iterable of integers only if
                PGPE is initialized with its keyword argument
                'num_interactions' set as a positive integer;
                otherwise to be given as None.
                In this interactions array, the i-th
                element represents the number of simulator interactions
                made by the i-th solution during its episode.
        Returns:
            True if the generation is complete and the center of the
            search distribution is updated; False otherwise.
        """
        
        iter_incremented = False
        self._clear_when_asked = False
        
        self._fitnesses.extend(fitnesses)
        
        if self._num_interactions is not None:
            if interactions is None:
                raise ValueError(
                    "A certain number of interactions was declared"
                    + "as required via the initialization arguments:"
                    + " num_interactions="
                    + repr(self._num_interactions)
                    + ". "
                    + "However, the argument 'interactions' was not"
                    + " provided when calling the method 'tell'."
                )
            self._interactions.extend(interactions)
            self._total_interactions += np.sum(interactions)
        else:
            if interactions is not None:
                raise ValueError(
                    " Received the argument 'interactions' unexpectedly."
                    " Please provide the argument 'interactions'"
                    " only when the PGPE solver was initialized"
                    " with num_interactions=A_POSITIVE_INTEGER"
                )
        
        iteration_is_done = (
            (self._num_interactions is None)
            or (self._total_interactions >= self._num_interactions)
            or (
                (self._popsize_max is not None)
                and (len(self._solutions) >= self._popsize_max)
            )
        )
        
        if iteration_is_done:
            dirs_count = len(self._solutions) // 2
            solution_weights = self._fitnesses
            if self._solution_ranking:
                solution_weights = compute_centered_ranks(solution_weights)
             
            # Compute the gradients
            grad_center, grad_stdev = compute_reinforce_update(
                solutions=self._solutions,
                fitnesses=solution_weights,
                scaled_noises=self._scaled_noises,
                stdev=self._stdev
            )

            # Update the center solution
            if self._optimizer is None:
                self._center = self._center + self._center_learning_rate * grad_center
            else:
                self._center = self._center + self._optimizer.ascent(grad_center)
            
            # Update the stdev of the search distribution
            old_stdev = self._stdev
            self._stdev = self._stdev + self._stdev_learning_rate * grad_stdev            
            if self._stdev_max_change is not None and self._stdev_max_change > 0:
                allowed_delta = abs(old_stdev) * self._stdev_max_change
                stdev_min_allowed = old_stdev - allowed_delta
                stdev_max_allowed = old_stdev + allowed_delta
                np.clip(self._stdev, stdev_min_allowed, stdev_max_allowed, out=self._stdev)

            # Update the iterations count
            self._iteration += 1
            iter_incremented = True
            self._clear_when_asked = True
        
        # Return whether the iteration was completed or not
        return iter_incremented

    def __len__(self) -> int:
        return len(self._solutions)

    def __getitem__(self, i: Union[Integral, slice]) -> Union[
        Tuple[np.ndarray, float],
        List[Tuple[np.ndarray, float]]
    ]:
        def fix_index(q, default_value):
            if not (isinstance(q, Integral) or q is None):
                raise TypeError("Invalid index: " + repr(q))

            if q is None:
                return default_value
            elif q < 0:
                return len(self._solutions) + q
            else:
                return q

        if isinstance(i, slice):
            result = []
            for j in range(
                fix_index(i.start, 0),
                fix_index(i.stop, len(self._solutions)),
                i.step
            ):
                result.append(readonly_view(self._solutions[j]), self._fitnesses[j])
        elif isinstance(i, Integral):
            result = readonly_view(self._solutions[i]), self._fitnesses[i]
        else:
            raise TypeError("Invalid index: " + repr(i))

        return result

    def __iter__(self):
        """Iterate over the solutions in the population.
        During the iteration, each element is received
        as (solution, fitness).
        """
        n = len(self._solutions)
        for i in range(n):
            yield self[i]

    @property
    def stdev(self) -> np.ndarray:
        """Get the standard deviation array of the search distribution"""
        return readonly_view(self._stdev)

    @property
    def center(self) -> np.ndarray:
        """Get the center point of the search distribution"""
        return readonly_view(self._center)

    @property
    def num_interactions(self) -> Optional[int]:
        """Get the number of interactions that must be reached
        before declaring a generation complete.
        Can also return 0 if PGPE is initialized with
        its keyword argument 'num_interactions' set as None.
        """
        return self._num_interactions

    @property
    def center_learning_rate(self) -> float:
        """Get the learning rate used for when updating the center
        point of the search distribution.
        """
        return self._center_learning_rate

    @property
    def stdev_learning_rate(self) -> float:
        """Get the learning rate used for when updating the standard
        deviation of the search distribution.
        """
        return self._stdev_learning_rate

    @property
    def stdev_max_change(self) -> Optional[float]:
        """Get the maximum change allowed for when updating
        the standard deviation of the search distribution.
        If, when initializing PGPE, this was left as default,
        0.2 is returned, meaning that 20% change is allowed.
        """
        return self._stdev_max_change

    @property
    def solution_ranking(self) -> bool:
        """Return as True or False, whether or not the solution
        ranking is being used.
        """
        return self._solution_ranking

