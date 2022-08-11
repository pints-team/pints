#
# Defines method for initialising points for sampling and optimising
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#

import pints
import numpy as np


def sample_initial_points(function, n_points, random_sampler=None,
                          boundaries=None, max_tries=50, parallel=False,
                          n_workers=None):
    """
    Samples ``n_points`` parameter values to use as starting points in a
    sampling or optimisation routine on the given ``function``.

    How the initial points are determined depends on the arguments supplied. In
    order of precedence:

    1. If a method ``random_sampler`` is provided then this will be used to
       draw the random samples.
    2. If no sampler method is given but ``function`` is a
       :class:`LogPosterior` then the method ``function.log_prior().sample()``
       will be used.
    3. If no sampler method is supplied and ``function`` is not a
       :class:`LogPosterior` and if ``boundaries`` are provided then the method
       ``boundaries.sample()`` will be used to draw samples.

    A ``ValueError`` is raised if none of the above options are available.

    Each sample ``x`` is tested to ensure that ``function(x)`` returns a finite
    result within ``boundaries`` if these are supplied. If not, a new sample
    will be drawn. This is repeated at most ``max_tries`` times, after which an
    error is raised.

    Parameters
    ----------
    function :
        A :class:`pints.ErrorMeasure` or a :class:`pints.LogPDF` that
        evaluates points in the parameter space. If the latter, it is optional
        that ``function`` be of type :class:`LogPosterior`.
    n_points : int
        The number of initial values to generate.
    random_sampler :
        A function that when called returns draws from a probability
        distribution of the same dimensionality as ``function``. The only
        argument to this function should be an integer specifying the number of
        draws.
    boundaries :
        An optional set of boundaries on the parameter space of class
        :class:`pints.Boundaries`.
    max_tries : int
        Number of attempts to find a finite initial value across all
        ``n_points``. By default this is 50 per point.
    parallel : bool
        Whether to evaluate ``function`` in parallel (defaults to False).
    n_workers : int
        Number of workers on which to run parallel evaluation.
    """
    is_not_logpdf = not isinstance(function, pints.LogPDF)
    is_not_errormeasure = not isinstance(function, pints.ErrorMeasure)

    # Check function
    if is_not_logpdf and is_not_errormeasure:
        raise ValueError(
            'function must be either pints.LogPDF or pints.ErrorMeasure.')

    # Check boundaries
    if boundaries is not None:
        if not isinstance(boundaries, pints.Boundaries):
            raise ValueError('boundaries must be a pints.Boundaries object.')
        elif boundaries.n_parameters() != function.n_parameters():
            raise ValueError('boundaries must match dimension of function.')

    # Check or set random sampler
    if random_sampler is None:
        if isinstance(function, pints.LogPosterior):
            random_sampler = function.log_prior().sample
        elif boundaries is not None:
            random_sampler = boundaries.sample
        else:
            raise ValueError(
                'If function is not a pints.LogPosterior and no boundaries'
                ' are given then a random_sampler must be supplied.')
    elif not callable(random_sampler):
        raise ValueError(
            'random_sampler must be a callable function, if supplied.')

    # Check number of initial points
    if n_points < 1:
        raise ValueError('Number of initial points must be 1 or more.')

    # Set up parallelisation
    if parallel:
        n_workers = min(pints.ParallelEvaluator.cpu_count(), n_points)
        evaluator = pints.ParallelEvaluator(function, n_workers=n_workers)
    else:
        evaluator = pints.SequentialEvaluator(function)

    # Go!
    x0 = []
    n_tries = 0
    while len(x0) < n_points and n_tries < max_tries:
        xs = random_sampler(n_points - len(x0))
        fxs = evaluator.evaluate(xs)
        for i, x in enumerate(xs):
            fx = fxs[i]
            if np.isfinite(fx):
                if boundaries is None or boundaries.check(x):
                    x0.append(x)
        n_tries += 1

    if len(x0) < n_points:
        raise RuntimeError(
            'Initialisation failed since function not finite or within ' +
            'bounds at initial points after ' + str(max_tries) + ' attempts.')
    return x0
