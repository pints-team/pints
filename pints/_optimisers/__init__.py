#
# Sub-module containing several optimisation routines
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class Optimiser(pints.Loggable, pints.TunableMethod):
    """
    Base class for optimisers implementing an ask-and-tell interface.

    This interface provides fine-grained control. Users seeking to simply run
    an optimisation may wish to use the :class:`OptimisationController`
    instead.

    Optimisation using "ask-and-tell" proceed by the user repeatedly "asking"
    the optimiser for points, and then "telling" it the function evaluations at
    those points. This allows a user to have fine-grained control over an
    optimisation, and implement custom parallelisation, logging, stopping
    criteria etc. Users who don't need this functionality can use optimisers
    via the :class:`OptimisationController` class instead.

    All PINTS optimisers are _minimisers_. To maximise a function simply pass
    in the negative of its evaluations to :meth:`tell()` (this is handled
    automatically by the :class:`OptimisationController`).

    All optimisers implement the :class:`pints.Loggable` and
    :class:`pints.TunableMethod` interfaces.

    Parameters
    ----------
    x0
        A starting point for searches in the parameter space. This value may be
        used directly (for example as the initial position of a particle in
        :class:`PSO`) or indirectly (for example as the center of a
        distribution in :class:`XNES`).
    sigma0
        An optional initial standard deviation around ``x0``. Can be specified
        either as a scalar value (one standard deviation for all coordinates)
        or as an array with one entry per dimension. Not all methods will use
        this information.
    boundaries
        An optional set of boundaries on the parameter space.

    Example
    -------
    An optimisation with ask-and-tell, proceeds roughly as follows::

        optimiser = MyOptimiser()
        running = True
        while running:
            # Ask for points to evaluate
            xs = optimiser.ask()

            # Evaluate the score function or pdf at these points
            # At this point, code to parallelise evaluation can be added in
            fs = [f(x) for x in xs]

            # Tell the optimiser the evaluations; allowing it to update its
            # internal state.
            optimiser.tell(fs)

            # Check stopping criteria
            # At this point, custom stopping criteria can be added in
            if optimiser.fbest() < threshold:
                running = False

            # Check for optimiser issues
            if optimiser.stop():
                running = False

            # At this point, code to visualise or benchmark optimiser behaviour
            # could be added in, for example by plotting `xs` in the parameter
            # space.
    """

    def __init__(self, x0, sigma0=None, boundaries=None):

        # Convert and store initial position
        self._x0 = pints.vector(x0)

        # Get dimension
        self._n_parameters = len(self._x0)
        if self._n_parameters < 1:
            raise ValueError('Problem dimension must be greater than zero.')

        # Store boundaries
        self._boundaries = boundaries
        if self._boundaries:
            if self._boundaries.n_parameters() != self._n_parameters:
                raise ValueError(
                    'Boundaries must have same dimension as starting point.')

        # Check initial position is within boundaries
        if self._boundaries:
            if not self._boundaries.check(self._x0):
                raise ValueError(
                    'Initial position must lie within given boundaries.')

        # Check initial standard deviation
        if sigma0 is None:
            # Set a standard deviation

            # Try and use boundaries to guess
            try:
                self._sigma0 = (1 / 6) * self._boundaries.range()
            except AttributeError:
                # No boundaries set, or boundaries don't support range()
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
            self._sigma0 = np.ones(self._n_parameters) * sigma0
            self._sigma0.setflags(write=False)

        else:
            # Vector given
            self._sigma0 = pints.vector(sigma0)
            if len(self._sigma0) != self._n_parameters:
                raise ValueError(
                    'Initial standard deviation must be None, scalar, or have'
                    ' dimension ' + str(self._n_parameters) + '.')
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

    def needs_sensitivities(self):
        """
        Returns ``True`` if this methods needs sensitivities to be passed in to
        ``tell`` along with the evaluated error.
        """
        return False

    def running(self):
        """
        Returns ``True`` if this an optimisation is in progress.
        """
        raise NotImplementedError

    def stop(self):
        """
        Checks if this method has run into trouble and should terminate.
        Returns ``False`` if everything's fine, or a short message (e.g.
        "Ill-conditioned matrix.") if the method should terminate.
        """
        return False

    def tell(self, fx):
        """
        Performs an iteration of the optimiser algorithm, using the evaluations
        ``fx`` of the points ``x`` previously specified by ``ask``.

        For methods that require sensitivities (see
        :meth:`needs_sensitivities`), ``fx`` should be a tuple
        ``(objective, sensitivities)``, containing the values returned by
        :meth:`pints.ErrorMeasure.evaluateS1()`.
        """
        raise NotImplementedError

    def xbest(self):
        """
        Returns the current best position.
        """
        raise NotImplementedError


class PopulationBasedOptimiser(Optimiser):
    """
    Base class for optimisers that work by moving multiple points through the
    search space.

    Extends :class:`Optimiser`.
    """

    def __init__(self, x0, sigma0=None, boundaries=None):
        super(PopulationBasedOptimiser, self).__init__(x0, sigma0, boundaries)

        # Set initial population size using heuristic
        self._population_size = self._suggested_population_size()

    def population_size(self):
        """
        Returns this optimiser's population size.

        If no explicit population size has been set, ``None`` may be returned.
        Once running, the correct value will always be returned.
        """
        return self._population_size

    def set_population_size(self, population_size=None):
        """
        Sets a population size to use in this optimisation.

        If `population_size` is set to ``None``, the population size will be
        set using the heuristic :meth:`suggested_population_size()`.
        """
        if self.running():
            raise Exception('Cannot change population size during run.')

        # Check population size or set using heuristic
        if population_size is not None:
            population_size = int(population_size)
            if population_size < 1:
                raise ValueError('Population size must be at least 1.')

        # Store
        self._population_size = population_size

    def suggested_population_size(self, round_up_to_multiple_of=None):
        """
        Returns a suggested population size for this method, based on the
        dimension of the search space (e.g. the parameter space).

        If the optional argument ``round_up_to_multiple_of`` is set to an
        integer greater than 1, the method will round up the estimate to a
        multiple of that number. This can be useful to obtain a population size
        based on e.g. the number of worker processes used to perform objective
        function evaluations.
        """
        population_size = self._suggested_population_size()

        if round_up_to_multiple_of is not None:
            n = int(round_up_to_multiple_of)
            if n > 1:
                population_size = n * (((population_size - 1) // n) + 1)

        return population_size

    def _suggested_population_size(self):
        """
        Returns a suggested population size for use by
        :meth:`suggested_population_size`.
        """
        raise NotImplementedError

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 1

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[population_size]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_population_size(x[0])


class OptimisationController(object):
    """
    Finds the parameter values that minimise an :class:`ErrorMeasure` or
    maximise a :class:`LogPDF`.

    Parameters
    ----------
    function
        An :class:`pints.ErrorMeasure` or a :class:`pints.LogPDF` that
        evaluates points in the parameter space.
    x0
        The starting point for searches in the parameter space. This value may
        be used directly (for example as the initial position of a particle in
        :class:`PSO`) or indirectly (for example as the center of a
        distribution in :class:`XNES`).
    sigma0
        An optional initial standard deviation around ``x0``. Can be specified
        either as a scalar value (one standard deviation for all coordinates)
        or as an array with one entry per dimension. Not all methods will use
        this information.
    boundaries
        An optional set of boundaries on the parameter space.
    method
        The class of :class:`pints.Optimiser` to use for the optimisation.
        If no method is specified, :class:`CMAES` is used.
    """

    def __init__(
            self, function, x0, sigma0=None, boundaries=None, method=None):

        # Convert x0 to vector
        # This converts e.g. (1, 7) shapes to (7, ), giving users a bit more
        # freedom with the exact shape passed in. For example, to allow the
        # output of LogPrior.sample(1) to be passed in.
        x0 = pints.vector(x0)

        # Check dimension of x0 against function
        if function.n_parameters() != len(x0):
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

        # Check if sensitivities are required
        self._needs_sensitivities = self._optimiser.needs_sensitivities()

        # Logging
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False
        self.set_log_interval()

        # Parallelisation
        self._parallel = False
        self._n_workers = 1
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

        # Post-run statistics
        self._evaluations = None
        self._iterations = None
        self._time = None

    def evaluations(self):
        """
        Returns the number of evaluations performed during the last run, or
        ``None`` if the controller hasn't ran yet.
        """
        return self._evaluations

    def iterations(self):
        """
        Returns the number of iterations performed during the last run, or
        ``None`` if the controller hasn't ran yet.
        """
        return self._iterations

    def max_iterations(self):
        """
        Returns the maximum iterations if this stopping criterion is set, or
        ``None`` if it is not. See :meth:`set_max_iterations()`.
        """
        return self._max_iterations

    def max_unchanged_iterations(self):
        """
        Returns a tuple ``(iterations, threshold)`` specifying a maximum
        unchanged iterations stopping criterion, or ``(None, None)`` if no such
        criterion is set. See :meth:`set_max_unchanged_iterations()`.
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
        Returns the number of parallel worker processes this routine will be
        run on, or ``False`` if parallelisation is disabled.
        """
        return self._n_workers if self._parallel else False

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

        # Iterations and function evaluations
        iteration = 0
        evaluations = 0

        # Unchanged iterations count (used for stopping or just for
        # information)
        unchanged_iterations = 0

        # Choose method to evaluate
        f = self._function
        if self._needs_sensitivities:
            f = f.evaluateS1

        # Create evaluator object
        if self._parallel:
            # Get number of workers
            n_workers = self._n_workers

            # For population based optimisers, don't use more workers than
            # particles!
            if isinstance(self._optimiser, PopulationBasedOptimiser):
                n_workers = min(n_workers, self._optimiser.population_size())
            evaluator = pints.ParallelEvaluator(f, n_workers=n_workers)
        else:
            evaluator = pints.SequentialEvaluator(f)

        # Keep track of best position and score
        fbest = float('inf')

        # Internally we always minimise! Keep a 2nd value to show the user
        fbest_user = fbest if self._minimising else -fbest

        # Set up progress reporting
        next_message = 0

        # Start logging
        logging = self._log_to_screen or self._log_filename
        if logging:
            if self._log_to_screen:
                # Show direction
                if self._minimising:
                    print('Minimising error measure')
                else:
                    print('Maximising LogPDF')

                # Show method
                print('Using ' + str(self._optimiser.name()))

                # Show parallelisation
                if self._parallel:
                    print('Running in parallel with ' + str(n_workers) +
                          ' worker processes.')
                else:
                    print('Running in sequential mode.')

            # Show population size
            pop_size = 1
            if isinstance(self._optimiser, PopulationBasedOptimiser):
                pop_size = self._optimiser.population_size()
                if self._log_to_screen:
                    print('Population size: ' + str(pop_size))

            # Set up logger
            logger = pints.Logger()
            if not self._log_to_screen:
                logger.set_stream(None)
            if self._log_filename:
                logger.set_filename(self._log_filename, csv=self._log_csv)

            # Add fields to log
            max_iter_guess = max(self._max_iterations or 0, 10000)
            max_eval_guess = max_iter_guess * pop_size
            logger.add_counter('Iter.', max_value=max_iter_guess)
            logger.add_counter('Eval.', max_value=max_eval_guess)
            logger.add_float('Best')
            self._optimiser._log_init(logger)
            logger.add_time('Time m:s')

        # Start searching
        timer = pints.Timer()
        running = True
        try:
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

                # Update evaluation count
                evaluations += len(fs)

                # Show progress
                if logging and iteration >= next_message:
                    # Log state
                    logger.log(iteration, evaluations, fbest_user)
                    self._optimiser._log_write(logger)
                    logger.log(timer.time())

                    # Choose next logging point
                    if iteration < self._message_warm_up:
                        next_message = iteration + 1
                    else:
                        next_message = self._message_interval * (
                            1 + iteration // self._message_interval)

                # Update iteration count
                iteration += 1

                #
                # Check stopping criteria
                #

                # Maximum number of iterations
                if (self._max_iterations is not None and
                        iteration >= self._max_iterations):
                    running = False
                    halt_message = ('Halting: Maximum number of iterations ('
                                    + str(iteration) + ') reached.')

                # Maximum number of iterations without significant change
                halt = (self._max_unchanged_iterations is not None and
                        unchanged_iterations >= self._max_unchanged_iterations)
                if halt:
                    running = False
                    halt_message = ('Halting: No significant change for ' +
                                    str(unchanged_iterations) + ' iterations.')

                # Threshold value
                if self._threshold is not None and fbest < self._threshold:
                    running = False
                    halt_message = ('Halting: Objective function crossed'
                                    ' threshold: ' + str(self._threshold) +
                                    '.')

                # Error in optimiser
                error = self._optimiser.stop()
                if error:   # pragma: no cover
                    running = False
                    halt_message = ('Halting: ' + str(error))

        except (Exception, SystemExit, KeyboardInterrupt):  # pragma: no cover
            # Unexpected end!
            # Show last result and exit
            print('\n' + '-' * 40)
            print('Unexpected termination.')
            print('Current best score: ' + str(fbest))
            print('Current best position:')
            for p in self._optimiser.xbest():
                print(pints.strfloat(p))
            print('-' * 40)
            raise

        # Stop timer
        self._time = timer.time()

        # Log final values and show halt message
        if logging:
            logger.log(iteration, evaluations, fbest_user)
            self._optimiser._log_write(logger)
            logger.log(self._time)
            if self._log_to_screen:
                print(halt_message)

        # Save post-run statistics
        self._evaluations = evaluations
        self._iterations = iteration

        # Return best position and score
        return self._optimiser.xbest(), fbest_user

    def set_log_interval(self, iters=20, warm_up=3):
        """
        Changes the frequency with which messages are logged.

        Parameters
        ----------
        ``interval``
            A log message will be shown every ``iters`` iterations.
        ``warm_up``
            A log message will be shown every iteration, for the first
            ``warm_up`` iterations.
        """
        iters = int(iters)
        if iters < 1:
            raise ValueError('Interval must be greater than zero.')
        warm_up = max(0, int(warm_up))

        self._message_interval = iters
        self._message_warm_up = warm_up

    def set_log_to_file(self, filename=None, csv=False):
        """
        Enables logging to file when a filename is passed in, disables it if
        ``filename`` is ``False`` or ``None``.

        The argument ``csv`` can be set to ``True`` to write the file in comma
        separated value (CSV) format. By default, the file contents will be
        similar to the output on screen.
        """
        if filename:
            self._log_filename = str(filename)
            self._log_csv = True if csv else False
        else:
            self._log_filename = None
            self._log_csv = False

    def set_log_to_screen(self, enabled):
        """
        Enables or disables logging to screen.
        """
        self._log_to_screen = True if enabled else False

    def set_max_iterations(self, iterations=10000):
        """
        Adds a stopping criterion, allowing the routine to halt after the
        given number of ``iterations``.

        This criterion is enabled by default. To disable it, use
        ``set_max_iterations(None)``.
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
        objective function doesn't change by more than ``threshold`` for the
        given number of ``iterations``.

        This criterion is enabled by default. To disable it, use
        ``set_max_unchanged_iterations(None)``.
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

    def set_parallel(self, parallel=False):
        """
        Enables/disables parallel evaluation.

        If ``parallel=True``, the method will run using a number of worker
        processes equal to the detected cpu core count. The number of workers
        can be set explicitly by setting ``parallel`` to an integer greater
        than 0.
        Parallelisation can be disabled by setting ``parallel`` to ``0`` or
        ``False``.
        """
        if parallel is True:
            self._parallel = True
            self._n_workers = pints.ParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else:
            self._parallel = False
            self._n_workers = 1

    def set_threshold(self, threshold):
        """
        Adds a stopping criterion, allowing the routine to halt once the
        objective function goes below a set ``threshold``.

        This criterion is disabled by default, but can be enabled by calling
        this method with a valid ``threshold``. To disable it, use
        ``set_treshold(None)``.
        """
        if threshold is None:
            self._threshold = None
        else:
            self._threshold = float(threshold)

    def threshold(self):
        """
        Returns the threshold stopping criterion, or ``None`` if no threshold
        stopping criterion is set. See :meth:`set_threshold()`.
        """
        return self._threshold

    def time(self):
        """
        Returns the time needed for the last run, in seconds, or ``None`` if
        the controller hasn't run yet.
        """
        return self._time


class Optimisation(OptimisationController):
    """ Deprecated alias for :class:`OptimisationController`. """

    def __init__(
            self, function, x0, sigma0=None, boundaries=None, method=None):
        # Deprecated on 2019-02-12
        import logging
        logging.basicConfig()
        log = logging.getLogger(__name__)
        log.warning(
            'The class `pints.Optimisation` is deprecated.'
            ' Please use `pints.OptimisationController` instead.')
        super(Optimisation, self).__init__(
            function, x0, sigma0=None, boundaries=None, method=None)


def optimise(function, x0, sigma0=None, boundaries=None, method=None):
    """
    Finds the parameter values that minimise an :class:`ErrorMeasure` or
    maximise a :class:`LogPDF`.

    Parameters
    ----------
    function
        An :class:`pints.ErrorMeasure` or a :class:`pints.LogPDF` that
        evaluates points in the parameter space.
    x0
        The starting point for searches in the parameter space. This value may
        be used directly (for example as the initial position of a particle in
        :class:`PSO`) or indirectly (for example as the center of a
        distribution in :class:`XNES`).
    sigma0
        An optional initial standard deviation around ``x0``. Can be specified
        either as a scalar value (one standard deviation for all coordinates)
        or as an array with one entry per dimension. Not all methods will use
        this information.
    boundaries
        An optional set of boundaries on the parameter space.
    method
        The class of :class:`pints.Optimiser` to use for the optimisation.
        If no method is specified, :class:`CMAES` is used.

    Returns
    -------
    xbest : numpy array
        The best parameter set obtained
    fbest : float
        The corresponding score.
    """
    return OptimisationController(
        function, x0, sigma0, boundaries, method).run()


class TriangleWaveTransform(object):
    """
    Transforms from unbounded to (rectangular) bounded parameter space using a
    periodic triangle-wave transform.

    Note: The transform is applied _inside_ optimisation methods, there is no
    need to wrap this around your own problem or score function.

    This can be applied as a transformation on ``x`` to implement _rectangular_
    boundaries in methods with no natural boundary mechanism. It effectively
    mirrors the search space at every boundary, leading to a continuous (but
    non-smooth) periodic landscape. While this effectively creates an infinite
    number of minima/maxima, each one maps to the same point in parameter
    space.

    It should work well for methods that maintain a single search position or a
    single search distribution (e.g. :class:`CMAES`, :class:`xNES`,
    :class:`SNES`), which will end up in one of the many mirror images.
    However, for methods that use independent search particles (e.g.
    :class:`PSO`) it could lead to a scattered population, with different
    particles exploring different mirror images. Other strategies should be
    used for such problems.
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


def curve_fit(f, x, y, p0, boundaries=None, threshold=None, max_iter=None,
              max_unchanged=200, verbose=False, parallel=False, method=None):
    """
    Fits a function ``f(x, *p)`` to a dataset ``(x, y)`` by finding the value
    of ``p`` for which ``sum((y - f(x, *p))**2) / n`` is minimised (where ``n``
    is the number of entries in ``y``).

    Returns a tuple ``(xbest, fbest)`` with the best position found, and the
    corresponding value ``fbest = f(xbest)``.

    Parameters
    ----------
    f : callable
        A function or callable class to be minimised.
    x
        The values of an independent variable, at which ``y`` was recorded.
    y
        Measured values ``y = f(x, p) + noise``.
    p0
        An initial guess for the optimal parameters ``p``.
    boundaries
        An optional :class:`pints.Boundaries` object or a tuple
        ``(lower, upper)`` specifying lower and upper boundaries for the
        search. If no boundaries are provided an unbounded search is run.
    threshold
        An optional absolute threshold stopping criterium.
    max_iter
        An optional maximum number of iterations stopping criterium.
    max_unchanged
        A stopping criterion based on the maximum number of successive
        iterations without a signficant change in ``f`` (see
        :meth:`pints.OptimisationController`).
    verbose
        Set to ``True`` to print progress messages to the screen.
    parallel
        Allows parallelisation to be enabled.
        If set to ``True``, the evaluations will happen in parallel using a
        number of worker processes equal to the detected cpu core count. The
        number of workers can be set explicitly by setting ``parallel`` to an
        integer greater than 0.
    method
        The :class:`pints.Optimiser` to use. If no method is specified,
        ``pints.CMAES`` is used.

    Returns
    -------
    xbest : numpy array
        The best parameter set obtained.
    fbest : float
        The corresponding score.

    Example
    -------
    ::

        import numpy as np
        import pints

        def f(x, a, b, c):
            return a + b * x + c * x ** 2

        x = np.linspace(-5, 5, 100)
        y = f(x, 1, 2, 3) + np.random.normal(0, 1)

        p0 = [0, 0, 0]
        popt = pints.curve_fit(f, x, y, p0)

    """
    # Test function
    if not callable(f):
        raise ValueError('The argument `f` must be callable.')

    # Get problem dimension from p0
    d = len(p0)

    # First dimension of x and y must agree
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            'The first dimension of `x` and `y` must be the same.')

    # Check boundaries
    if not (boundaries is None or isinstance(boundaries, pints.Boundaries)):
        lower, upper = boundaries
        boundaries = pints.RectangularBoundaries(lower, upper)

    # Create an error measure
    e = _CurveFitError(f, d, x, y)

    # Set up optimisation
    opt = pints.OptimisationController(
        e, p0, boundaries=boundaries, method=method)

    # Set stopping criteria
    opt.set_threshold(threshold)
    opt.set_max_iterations(max_iter)
    opt.set_max_unchanged_iterations(max_unchanged)

    # Set parallelisation
    opt.set_parallel(parallel)

    # Set output
    opt.set_log_to_screen(True if verbose else False)

    # Run and return
    return opt.run()


class _CurveFitError(pints.ErrorMeasure):
    """ Error measure for :meth:`curve_fit()`. """

    def __init__(self, function, dimension, x, y):
        self.f = function
        self.d = dimension
        self.x = x
        self.y = y
        self.n = 1 / np.product(y.shape)    # Total number of points in data

    def n_parameters(self):
        return self.d

    def __call__(self, p):
        return np.sum((self.y - self.f(self.x, *p))**2) * self.n


def fmin(f, x0, args=None, boundaries=None, threshold=None, max_iter=None,
         max_unchanged=200, verbose=False, parallel=False, method=None):
    """
    Minimises a callable function ``f``, starting from position ``x0``, using a
    :class:`pints.Optimiser`.

    Returns a tuple ``(xbest, fbest)`` with the best position found, and the
    corresponding value ``fbest = f(xbest)``.

    Parameters
    ----------
    f
        A function or callable class to be minimised.
    x0
        The initial point to search at. Must be a 1-dimensional sequence (e.g.
        a list or a numpy array).
    args
        An optional tuple of extra arguments for ``f``.
    boundaries
        An optional :class:`pints.Boundaries` object or a tuple
        ``(lower, upper)`` specifying lower and upper boundaries for the
        search. If no boundaries are provided an unbounded search is run.
    threshold
        An optional absolute threshold stopping criterium.
    max_iter
        An optional maximum number of iterations stopping criterium.
    max_unchanged
        A stopping criterion based on the maximum number of successive
        iterations without a signficant change in ``f`` (see
        :meth:`pints.OptimisationController`).
    verbose
        Set to ``True`` to print progress messages to the screen.
    parallel
        Allows parallelisation to be enabled.
        If set to ``True``, the evaluations will happen in parallel using a
        number of worker processes equal to the detected cpu core count. The
        number of workers can be set explicitly by setting ``parallel`` to an
        integer greater than 0.
    method
        The :class:`pints.Optimiser` to use. If no method is specified,
        ``pints.CMAES`` is used.

    Example
    -------
    ::

        import pints

        def f(x):
            return (x[0] - 3) ** 2 + (x[1] + 5) ** 2

        xopt, fopt = pints.fmin(f, [1, 1])
    """
    # Test function
    if not callable(f):
        raise ValueError('The argument `f` must be callable.')

    # Get problem dimension from x0
    d = len(x0)

    # Test extra arguments
    if args is not None:
        args = tuple(args)

    # Check boundaries
    if not (boundaries is None or isinstance(boundaries, pints.Boundaries)):
        lower, upper = boundaries
        boundaries = pints.RectangularBoundaries(lower, upper)

    # Create an error measure
    e = _FminError(f, d) if args is None else _FminErrorWithArgs(f, d, args)

    # Set up optimisation
    opt = pints.OptimisationController(
        e, x0, boundaries=boundaries, method=method)

    # Set stopping criteria
    opt.set_threshold(threshold)
    opt.set_max_iterations(max_iter)
    opt.set_max_unchanged_iterations(max_unchanged)

    # Set parallelisation
    opt.set_parallel(parallel)

    # Set output
    opt.set_log_to_screen(True if verbose else False)

    # Run and return
    return opt.run()


class _FminError(pints.ErrorMeasure):
    """ Error measure for :meth:`fmin()`. """

    def __init__(self, f, d):
        self.f = f
        self.d = d

    def n_parameters(self):
        return self.d

    def __call__(self, x):
        return self.f(x)


class _FminErrorWithArgs(pints.ErrorMeasure):
    """ Error measure for :meth:`fmin()` for functions with args. """

    def __init__(self, f, d, args):
        self.f = f
        self.d = d
        self.args = args

    def n_parameters(self):
        return self.d

    def __call__(self, x):
        return self.f(x, *self.args)
