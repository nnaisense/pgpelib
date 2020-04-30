import numpy as np
from typing import Iterable


###########################################################################
#
# The following section of this file contains ranking functions
# adapted from OpenAI's evolution-strategies-starter repository.
#
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
#
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


def _compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x: Iterable) -> np.ndarray:
    """
    Convert the given fitness array to an array of linear ranks.

    The ranks are centered around 0, the element with the lowest fitness
    is assigned the rank -0.5, and the element with the highest fitness is
    assigned the rank +0.5. This is the ranking method used in
    OpenAI's evolution strategy variant (Salimans et al. (2017)).

    Reference::

        Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor,
        Ilya Sutskever (2017). Evolution Strategies as a Scalable Alternative
        to Reinforcement Learning.

    Args:
        x: A sequence of fitness values

    Returns:
        A numpy array containing the ranks
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=float)
    y = _compute_ranks(x.ravel()).reshape(x.shape).astype(float)
    y /= (x.size - 1)
    y -= .5
    return y

###########################################################################
