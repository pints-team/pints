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
                          boundaries=None, max_tries=None, parallel=False,
                          n_workers=None):
    """
    Draws parameter values from a given sampling distribution until either
    finite values for each of ``n_points`` have been generated or the total
    number of attempts exceeds ``max_tries``.

    If ``log_pdf`` is of :class:`LogPosterior`, then the
    ``log_pdf.log_prior().sample`` method is used for initialisation, although
    this is overruled by ``random_sampler`` if it is supplied.

    Parameters
    ----------
    function :
        An :class:`pints.ErrorMeasure` or a :class:`pints.LogPDF` that
        evaluates points in the parameter space. If the latter, it is optional
        that ``log_pdf`` be of type :class:`LogPosterior`.
    n_points : int
        The number of initial values to generate.
    random_sampler :
        A function that when called returns draws from a probability
        distribution of the same dimensionality as ``log_pdf``. The only
        argument to this function should be an integer specifying the number of
        draws.
    boundaries :
        An optional set of boundaries on the parameter space of class
        :class:`pints.Boundaries`.
    max_tries : int
        Number of attempts to find a finite initial value across all
        ``n_points``. By default this is 50 x n_points.
    parallel : bool
        Whether to evaluate ``log_pdf`` in parallel (defaults to False).
    n_workers : int
        Number of workers on which to run parallel evaluation.
    """
    if random_sampler is not None and not callable(random_sampler):
        raise ValueError("random_sampler must be a callable function.")

    if random_sampler is None:
        if isinstance(function, pints.LogPosterior):
            random_sampler = function.log_prior().sample
        else:
            raise ValueError("If log_pdf not of class pints.LogPosterior " +
                             "then random_sampler must be supplied.")

    if boundaries is not None:
        if not isinstance(boundaries, pints.Boundaries):
            raise ValueError("Boundaries muse be of class pints.Boundaries.")

    if n_points < 1:
        raise ValueError("Number of initial points must be 1 or more.")

    if max_tries is None:
        max_tries = 50 * n_points

    if parallel:
        n_workers = min(pints.ParallelEvaluator.cpu_count(), n_points)
        evaluator = pints.ParallelEvaluator(function, n_workers=n_workers)
    else:
        evaluator = pints.SequentialEvaluator(function)

    initialised_finite = False
    x0 = []
    n_tries = 0
    while not initialised_finite and n_tries < max_tries:
        xs = random_sampler(n_points)
        fxs = evaluator.evaluate(xs)
        xs_iterator = iter(xs)
        fxs_iterator = iter(fxs)
        for i in range(n_points):
            x = next(xs_iterator)
            fx = next(fxs_iterator)
            if np.isfinite(fx):
                if boundaries is None:
                    x0.append(x)
                else:
                    if boundaries.check(x0):
                        x0.append(x)

            if len(x0) == n_points:
                initialised_finite = True
            n_tries += 1
    if not initialised_finite:
        raise RuntimeError(
            'Initialisation failed since log_pdf not finite at initial ' +
            'points after ' + str(max_tries) + ' attempts.')
    return x0
