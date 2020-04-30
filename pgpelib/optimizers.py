"""
Implementations of momentum-based optimizers.

The algorithms here are to be used within
distribution-based search algorithms, for
following their estimated gradients using
various momentum-based schemes.
"""

from typing import Union, Optional
import numpy as np
from numbers import Real, Integral
from .misc import positive_int, positive_float


# ==========================================================================
# The following section of this source file contains optimizer classes
# copied and adapted from OpenAI's evolution-strategies-starter repository.

# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

# evolution-strategies-starter license:
#
# The MIT License
#
# Copyright (c) 2016 OpenAI (http://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Code copied and adapted from OpenAI's evolution-strategies-starter begins
# here:

#class Optimizer(object):
#    def __init__(self, pi):
#        self.pi = pi
#        self.dim = pi.num_params
#        self.t = 0
#
#    def update(self, globalg):
#        self.t += 1
#        step = self._compute_step(globalg)
#        #theta = self.pi.get_trainable_flat()
#        theta = self.pi.mu
#        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
#        #self.pi.set_trainable_flat(theta + step)
#        self.pi.mu = theta + step
#        return ratio
#
#    def _compute_step(self, globalg):
#        raise NotImplementedError

class Optimizer:
    def __init__(self, *, solution_length: int, dtype: Union[str, np.dtype]):
        self.dim = positive_int(solution_length)
        self.dtype = np.dtype(dtype)
        self.t = 0

    def ascent(self, globalg: np.ndarray) -> np.ndarray:
        globalg = np.asarray(globalg, dtype=self.dtype)

        if globalg.ndim != 1:
            raise ValueError(
                "The argument 'globalg' was expected as a 1-dimensional"
                f" array, but it was received with shape {globalg.shape}"
            )

        if len(globalg) != self.dim:
            raise ValueError(
                "The first 'globalg' to this optimizer was provided"
                f" as an array of length {self.dim},"
                " but the last one has an incompatible length:"
                f" {len(globalg)}"
            )

        self.t += 1

        return np.asarray(self._compute_step(-globalg), dtype=self.dtype)

    def _compute_step(self, globalg):
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self,
                 *,
                 solution_length: int,
                 dtype: Union[str, np.dtype],
                 stepsize: float,
                 beta1: float=0.9,
                 beta2: float=0.999,
                 epsilon: float=1e-08):
        Optimizer.__init__(self, solution_length=solution_length, dtype=dtype)
        self.stepsize = positive_float(stepsize)
        self.beta1 = positive_float(beta1)
        self.beta2 = positive_float(beta2)
        self.epsilon = positive_float(epsilon)
        self.m = np.zeros(self.dim, dtype=self.dtype)
        self.v = np.zeros(self.dim, dtype=self.dtype)

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step

# Code copied and adapted from OpenAI's evolution-strategies-starter ends here.
# ==========================================================================

class ClipUp(Optimizer):
    # This code is based on OpenAI's SGD class.
    # It works like SGD, but also
    # clips the velocity and the step to be taken.

    @staticmethod
    def clip(x, max_length: float):
        length = np.sqrt(np.sum(x * x))
        if length > max_length:
            ratio = max_length / length
            return x * ratio
        else:
            return x

    def __init__(self,
                 *,
                 solution_length: int,
                 dtype: Union[str, np.dtype],
                 stepsize: float,
                 momentum: float=0.9,
                 max_speed: float=0.15,
                 fix_gradient_size: bool=True):
        Optimizer.__init__(self, solution_length=solution_length, dtype=dtype)
        self.v = np.zeros(self.dim, dtype=self.dtype)
        self.stepsize = positive_float(stepsize)
        self.momentum = positive_float(momentum)
        self.max_speed = positive_float(max_speed)
        self.fix_gradient_size = bool(fix_gradient_size)

    def _compute_step(self, globalg):
        if self.fix_gradient_size:
            g_len = np.sqrt(np.sum(globalg * globalg))
            globalg = globalg / g_len

        step = globalg * self.stepsize

        self.v = self.momentum * self.v + step
        self.v = self.clip(self.v, self.max_speed)

        return -self.v
