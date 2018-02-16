#
# Sub-module containing several optimisation routines
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class Optimiser(object):
    """
    Base class for optimisers implementing an ask-and-tell interface.

    Optimisers are initialised using the arguments:

    ``x0``
        A starting point for searches in the parameter space. This value may be
        used directly (for example as the initial position of a particle in
        :class:`PSO`) or indirectly (for example as the center of a
        distribution in :class:`XNES`).
    ``sigma0=None``
        An optional initial standard deviation around ``x0``. Can be specified
        either as a scalar value (one standard deviation for all coordinates)
        or as an array with one entry per dimension. Not all methods will use
        this information.
    ``boundaries=None``
        An optional set of boundaries on the parameter space.

    """
    def __init__(self, x0, sigma0=None, boundaries=None):

        # Print info to console
        self._verbose = True

        # Get dimension
        self._dimension = len(x0)
        if self._dimension < 1:
            raise ValueError('Problem dimension must be greater than zero.')

        # Store boundaries
        self._boundaries = boundaries
        if self._boundaries:
            if self._boundaries.dimension() != self._dimension:
                raise ValueError(
                    'Boundaries must have same dimension as starting point.')

        # Store initial position
        self._x0 = pints.vector(x0)
        if self._boundaries:
            if not self._boundaries.check(self._x0):
                raise ValueError(
                    'Initial position must lie within given boundaries.')

        # Check initial standard deviation
        if sigma0 is None:
            # Set a standard deviation
            if self._boundaries:
                # Use boundaries to guess
                self._sigma0 = (1 / 6) * self._boundaries.range()
            else:
                # Use initial position to guess at parameter scaling
                self._sigma0 = (1 / 3) * np.abs(self._x0)
                # But add 1 for any initial value that's zero
                self._sigma0 += (self._sigma0 == 0)
            self._sigma0.setflags(write=False)

        elif np.isscalar(sigma0):
            # Single number given, convert to vector
            sigma0 = float(sigma0)
            if sigma0 <= 0:
                raise ValueError(
                    'Initial standard deviation must be greater than zero.')
            self._sigma0 = np.ones(self._dimension) * sigma0
            self._sigma0.setflags(write=False)

        else:
            # Vector given
            self._sigma0 = pints.vector(sigma0)
            if len(self._sigma0) != self._dimension:
                raise ValueError(
                    'Initial standard deviation must be None, scalar, or have'
                    ' dimension ' + str(self._dimension) + '.')
            if np.any(self._sigma0 <= 0):
                raise ValueError(
                    'Initial standard deviations must be greater than zero.')

    def ask(self):
        """
        Returns a list of positions in the search space to evaluate.
        """
        raise NotImplementedError

    def fbest(self):
        """
        Returns the objective function evaluated at the current best position.
        """
        raise NotImplementedError

    def name(self):
        """
        Returns this method's full name.
        """
        raise NotImplementedError

    def tell(self, fx):
        """
        Performs an iteration of the optimiser algorithm, using the evaluations
        ``fx`` of the points ``x`` previously specified by ``ask``.
        """
        raise NotImplementedError

    def xbest(self):
        """
        Returns the current best position.
        """
        raise NotImplementedError


class PopulationBasedOptimiser(Optimiser):
    """
    *Extends:* :class:`PopulationBasedOptimiser`

    Base class for optimisers that work by moving multiple points through the
    search space.
    """
    def population_size(self):
        """
        Returns this optimiser's population size.
        """
        raise NotImplementedError

    def set_population_size(self, population_size=None, parallel=False):
        """
        Sets a population size to use in this optimisation.

        If `population_size` is set to `None` a default value will be set using
        a heuristic (e.g. based on the dimension of the search space).

        If `parallel` is set to `True`, the population size will be adjusted to
        a value suitable for parallel computations (e.g. by rounding up to a
        multiple of the number of reported CPU cores).
        """
        raise NotImplementedError


class Optimisation(object):
    """
    Finds the parameter values that minimise an :class:`ErrorMeasure` or
    maximise a :class:`LogPDF`.

    Arguments:

    ``function``
        An :class:`pints.ErrorMeasure` or a :class:`pints.LogPDF` that
        evaluates points in the parameter space.
    ``x0``
        The starting point for searches in the parameter space. This value may
        be used directly (for example as the initial position of a particle in
        :class:`PSO`) or indirectly (for example as the center of a
        distribution in :class:`XNES`).
    ``sigma0=None``
        An optional initial standard deviation around ``x0``. Can be specified
        either as a scalar value (one standard deviation for all coordinates)
        or as an array with one entry per dimension. Not all methods will use
        this information.
    ``boundaries=None``
        An optional set of boundaries on the parameter space.
    ``method=None``
        The class of :class:`pints.Optimiser` to use for the optimisation.
        If no method is specified, :class:`CMAES` is used.

    """
    def __init__(
            self, function, x0, sigma0=None, boundaries=None, method=None):

        # Check dimension of x0 against function
        if function.dimension() != len(x0):
            raise ValueError(
                'Starting point must have same dimension as function to'
                ' optimise.')

        # Check if minimising or maximising
        self._minimising = not isinstance(function, pints.LogPDF)

        # Store function
        if self._minimising:
            self._function = function
        else:
            self._function = pints.ProbabilityBasedError(function)
        del(function)

        # Create optimiser
        if method is None:
            method = pints.CMAES
        elif not issubclass(method, pints.Optimiser):
            raise ValueError('Method must be subclass of pints.Optimiser.')
        self._optimiser = method(x0, sigma0, boundaries)

        # Print info to console
        self._verbose = True

        # Run parallelised version
        self._parallel = None
        self.set_parallel()

        #
        # Stopping criteria
        #

        # Maximum iterations
        self._max_iterations = None
        self.set_max_iterations()

        # Maximum unchanged iterations
        self._max_unchanged_iterations = None
        self._min_significant_change = 1
        self.set_max_unchanged_iterations()

        # Threshold value
        self._threshold = None

    def max_iterations(self):
        """
        Returns the maximum iterations if this stopping criterion is set, or
        ``None`` if it is not. See :meth:`set_max_iterations`.
        """
        return self._max_iterations

    def max_unchanged_iterations(self):
        """
        Returns a tuple ``(iterations, threshold)`` specifying a maximum
        unchanged iterations stopping criterion, or ``(None, None)`` if no such
        criterion is set. See :meth:`set_max_unchanged_iterations`.
        """
        if self._max_unchanged_iterations is None:
            return (None, None)
        return (self._max_unchanged_iterations, self._min_significant_change)

    def optimiser(self):
        """
        Returns the underlying optimiser object, allowing detailed
        configuration.
        """
        return self._optimiser

    def parallel(self):
        """
        Returns ``True`` if this optimisation runs in parallel.
        """
        return self._parallel

    def run(self):
        """
        Runs the optimisation, returns a tuple ``(xbest, fbest)``.
        """
        # Check stopping criteria
        has_stopping_criterion = False
        has_stopping_criterion |= (self._max_iterations is not None)
        has_stopping_criterion |= (self._max_unchanged_iterations is not None)
        has_stopping_criterion |= (self._threshold is not None)
        if not has_stopping_criterion:
            raise ValueError('At least one stopping criterion must be set.')

        # Iterations
        iteration = 0

        # Unchanged iterations count (used for stopping or just for
        # information)
        unchanged_iterations = 0

        # Create evaluator object
        if self._parallel:
            evaluator = pints.ParallelEvaluator(self._function)
        else:
            evaluator = pints.SequentialEvaluator(self._function)

        # Keep track of best position and score
        fbest = float('inf')

        # Internally we always minimise! Keep a 2nd value to show the user
        fbest_user = fbest if self._minimising else -fbest

        # Set up progress reporting
        next_message = 0
        message_warm_up = 3
        message_interval = 20

        # Print configuration
        if self._verbose:
            # Show direction
            if self._minimising:
                print('Minimising error measure')
            else:
                print('Maximising LogPDF')

            # Show method
            print('using ' + str(self._optimiser.name()))

            # Show parallelisation
            if self._parallel:
                print('Running in parallel mode.')
            else:
                print('Running in sequential mode.')

            # Show population size
            if isinstance(self._optimiser, PopulationBasedOptimiser):
                print('Population size: '
                      + str(self._optimiser.population_size()))

        # Start searching
        running = True
        while running:
            # Get points
            xs = self._optimiser.ask()

            # Calculate scores
            fs = evaluator.evaluate(xs)

            # Perform iteration
            self._optimiser.tell(fs)

            # Check if new best found
            fnew = self._optimiser.fbest()
            if fnew < fbest:
                # Check if this counts as a significant change
                if np.abs(fnew - fbest) < self._min_significant_change:
                    unchanged_iterations += 1
                else:
                    unchanged_iterations = 0

                # Update best
                fbest = fnew

                # Update user value of fbest
                fbest_user = fbest if self._minimising else -fbest
            else:
                unchanged_iterations += 1

            # Show progress in verbose mode:
            if self._verbose and iteration >= next_message:
                print(str(iteration) + ': ' + str(fbest_user))
                if iteration < message_warm_up:
                    next_message = iteration + 1
                else:
                    next_message = message_interval * (
                        1 + iteration // message_interval)

            # Update iteration count
            iteration += 1

            #
            # Check stopping criteria
            #

            # Maximum number of iterations
            if (self._max_iterations is not None and
                    iteration >= self._max_iterations):
                running = False
                if self._verbose:
                    print('Halting: Maximum number of iterations ('
                          + str(iteration) + ') reached.')

            # Maximum number of iterations without significant change
            if (self._max_unchanged_iterations is not None and
                    unchanged_iterations >= self._max_unchanged_iterations):
                running = False
                if self._verbose:
                    print('Halting: No significant change for '
                          + str(unchanged_iterations) + ' iterations.')

            # Threshold value
            if self._threshold is not None and fbest < self._threshold:
                running = False
                if self._verbose:
                    print('Halting: Objective function crossed threshold: '
                          + str(self._threshold) + '.')

        # Show final value
        if self._verbose:
            print(str(iteration) + ': ' + str(fbest_user))

        # Return best position and score
        return self._optimiser.xbest(), fbest_user

    def set_max_iterations(self, iterations=10000):
        """
        Adds a stopping criterion, allowing the routine to halt after the
        given number of `iterations`.

        This criterion is enabled by default. To disable it, use
        `set_max_iterations(None)`.
        """
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError(
                    'Maximum number of iterations cannot be negative.')
        self._max_iterations = iterations

    def set_max_unchanged_iterations(self, iterations=200, threshold=1e-11):
        """
        Adds a stopping criterion, allowing the routine to halt if the
        objective function doesn't change by more than `threshold` for the
        given number of `iterations`.

        This criterion is enabled by default. To disable it, use
        `set_max_unchanged_iterations(None)`.
        """
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError(
                    'Maximum number of iterations cannot be negative.')

        threshold = float(threshold)
        if threshold < 0:
            raise ValueError('Minimum significant change cannot be negative.')

        self._max_unchanged_iterations = iterations
        self._min_significant_change = threshold

    def set_parallel(self, parallel=False, update_population_size=True):
        """
        Enables/disables parallel function evaluation.

        If a :class:`PopulationBasedOptimiser` method is used, this method will
        also update the used population size. To disable this behaviour, use
        `update_population_size=False`.
        """
        self._parallel = bool(parallel)

        if update_population_size:
            if isinstance(self._optimiser, PopulationBasedOptimiser):
                self._optimiser.set_population_size(parallel=parallel)

    def set_threshold(self, threshold):
        """
        Adds a stopping criterion, allowing the routine to halt once the
        objective function goes below a set `threshold`.

        This criterion is disabled by default, but can be enabled by calling
        this method with a valid `threshold`. To disable it, use
        `set_treshold(None)`.
        """
        if threshold is None:
            self._threshold = None
        else:
            self._threshold = float(threshold)

    def set_verbose(self, value):
        """
        Enables or disables verbose mode for this optimiser. In verbose mode,
        lots of output is generated during an optimisation.
        """
        self._verbose = bool(value)

    def threshold(self):
        """
        Returns the threshold stopping criterion, or ``None`` if no threshold
        stopping criterion is set. See :meth:`set_threshold`.
        """
        return self._threshold

    def verbose(self):
        """
        Returns ``True`` if the optimiser is set to run in verbose mode.
        """
        return self._verbose


def optimise(function, x0, sigma0=None, boundaries=None, method=None):
    """
    Finds the parameter values that minimise an :class:`ErrorMeasure` or
    maximise a :class:`LogPDF`.

    Arguments:

    ``function``
        An :class:`pints.ErrorMeasure` or a :class:`pints.LogPDF` that
        evaluates points in the parameter space.
    ``x0``
        The starting point for searches in the parameter space. This value may
        be used directly (for example as the initial position of a particle in
        :class:`PSO`) or indirectly (for example as the center of a
        distribution in :class:`XNES`).
    ``sigma0=None``
        An optional initial standard deviation around ``x0``. Can be specified
        either as a scalar value (one standard deviation for all coordinates)
        or as an array with one entry per dimension. Not all methods will use
        this information.
    ``boundaries=None``
        An optional set of boundaries on the parameter space.
    ``method=None``
        The class of :class:`pints.Optimiser` to use for the optimisation.
        If no method is specified, :class:`CMAES` is used.

    Returns a tuple ``(xbest, fbest)``.
    """
    return Optimisation(function, x0, sigma0, boundaries, method).run()


class TriangleWaveTransform(object):
    """
    Transforms from unbounded to bounded parameter space using a periodic
    triangle-wave transform.

    Note: The transform is applied _inside_ optimisation methods, there is no
    need to wrap this around your own problem or score function.

    This can be applied as a transformation on ``x`` to implement boundaries in
    methods with no natural boundary mechanism. It effectively mirrors the
    search space at every boundary, leading to a continuous (but non-smooth)
    periodic landscape. While this effectively creates an infinite number of
    minima/maxima, each one maps to the same point in parameter space.

    It should work well for that maintain a single search position or a single
    search distribution (e.g. :class:`CMAES`, :class:`xNES`, :class:`SNES`),
    which will end up in one of the many mirror images. However, for methods
    that use independent search particles (e.g. :class:`PSO`) it could lead to
    a scattered population, with different particles exploring different mirror
    images. Other strategies should be used for such problems.
    """
    def __init__(self, boundaries):
        self._lower = boundaries.lower()
        self._upper = boundaries.upper()
        self._range = self._upper - self._lower
        self._range2 = 2 * self._range

    def __call__(self, x):
        y = np.remainder(x - self._lower, self._range2)
        z = np.remainder(y, self._range)
        return ((self._lower + z) * (y < self._range)
                + (self._upper - z) * (y >= self._range))
