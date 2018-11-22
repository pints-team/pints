#
# Sub-module containing MCMC inference routines
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import os
import pints
import numpy as np


class MCMCSampler(pints.Loggable, pints.TunableMethod):
    """
    Abstract base class for (single or multi-chain) MCMC methods.

    All MCMC samplers implement the :class:`pints.Loggable` and
    :class:`pints.TunableMethod` interfaces.
    """

    def name(self):
        """
        Returns this method's full name.
        """
        raise NotImplementedError

    def in_initial_phase(self):
        """
        For methods that need an initial phase (see
        :meth:`needs_initial_phase()`), this method returns ``True`` if the
        method is currently configured to be in its initial phase. For other
        methods a ``NotImplementedError`` is returned.
        """
        raise NotImplementedError

    def needs_initial_phase(self):
        """
        Returns ``True`` if this method needs an initial phase, for example an
        adaptation-free period for adaptive covariance methods, or a warm-up
        phase for DREAM.
        """
        return False

    def set_initial_phase(self, in_initial_phase):
        """
        For methods that need an initial phase (see
        :meth:`needs_initial_phase()`), this method toggles the initial phase
        algorithm. For other methods a ``NotImplementedError`` is returned.
        """
        raise NotImplementedError

    def needs_sensitivities(self):
        """
        Returns ``True`` if this methods needs sensitivities to be passed in to
        ``tell`` along with the evaluated logpdf.
        """
        return False


class SingleChainMCMC(MCMCSampler):
    """
    Abstract base class for MCMC methods that generate a single markov chain,
    via an ask-and-tell interface.

    Arguments:

    ``x0``
        An starting point in the parameter space.
    ``sigma0=None``
        An optional (initial) covariance matrix, i.e., a guess of the
        covariance of the distribution to estimate, around ``x0``.

    *Extends:* :class:`MCMCSampler`
    """

    def __init__(self, x0, sigma0=None):

        # Check initial position
        self._x0 = pints.vector(x0)

        # Get number of parameters
        self._dimension = len(self._x0)

        # Check initial standard deviation
        if sigma0 is None:
            # Get representative parameter value for each parameter
            self._sigma0 = np.abs(self._x0)
            self._sigma0[self._sigma0 == 0] = 1
            # Use to create diagonal matrix
            self._sigma0 = np.diag(0.01 * self._sigma0)
        else:
            self._sigma0 = np.array(sigma0)
            if np.product(self._sigma0.shape) == self._dimension:
                # Convert from 1d array
                self._sigma0 = self._sigma0.reshape((self._dimension,))
                self._sigma0 = np.diag(self._sigma0)
            else:
                # Check if 2d matrix of correct size
                self._sigma0 = self._sigma0.reshape(
                    (self._dimension, self._dimension))

    def ask(self):
        """
        Returns a parameter vector to evaluate the logpdf for.
        """
        raise NotImplementedError

    def tell(self, fx):
        """
        Performs an iteration of the MCMC algorithm, using the logpdf
        evaluation ``fx`` of the point previously specified by ``ask``.

        Returns either the next sample in the chain, or ``None`` to indicate
        that no new sample should be added to the chain (this is used to
        implement methods that require multiple evaluations per iteration).
        Note that, if one chain returns ``None``, all chains should return
        ``None``.

        For methods that require sensitivities (see
        :meth:`MCMCSamper.needs_sensitivities`), ``fx`` should be a tuple
        ``(log_pdf, sensitivities)``, containing the values returned by
        :meth:`pints.LogPdf.evaluateS1()`.
        """
        raise NotImplementedError

    def replace(self, x, fx):
        """
        Replaces the chain's current position by a user-specified point ``x``,
        with log-pdf ``fx``.

        This is an optional method, and may not be implemented by all methods!
        """
        raise NotImplementedError


class MultiChainMCMC(MCMCSampler):
    """
    Abstract base class for MCMC methods that generate multiple markov chains,
    via an ask-and-tell interface.

    Arguments:

    ``chains``
        The number of MCMC chains to generate.
    ``x0``
        A sequence of starting points. Can be a list of lists, a 2-dimensional
        array, or any other structure such that ``x0[i]`` is the starting point
        for chain ``i``.
    ``sigma0=None``
        An optional initial covariance matrix, i.e., a guess of the covariance
        in ``logpdf`` around the points in ``x0`` (the same ``sigma0`` is used
        for each point in ``x0``).
        Can be specified as a ``(d, d)`` matrix (where ``d`` is the dimension
        of the parameterspace) or as a ``(d, )`` vector, in which case
        ``diag(sigma0)`` will be used.

    *Extends:* :class:`MCMCSampler`
    """

    def __init__(self, chains, x0, sigma0=None):

        # Check number of chains
        self._chains = int(chains)
        if self._chains < 1:
            raise ValueError('Number of chains must be at least 1.')

        # Check initial position(s)
        if len(x0) != chains:
            raise ValueError(
                'Number of initial positions must be equal to number of'
                ' chains.')
        self._x0 = np.array([pints.vector(x) for x in x0])
        self._x0.setflags(write=False)

        # Get number of parameters
        self._dimension = len(self._x0[0])

        # Check initial points all have correct dimension
        if not all([len(x) == self._dimension for x in self._x0]):
            raise ValueError('All initial points must have same dimension.')

        # Check initial standard deviation
        if sigma0 is None:
            # Get representative parameter value for each parameter
            self._sigma0 = np.max(np.abs(self._x0), axis=0)
            self._sigma0[self._sigma0 == 0] = 1
            # Use to create diagonal matrix
            self._sigma0 = np.diag(0.01 * self._sigma0)
        else:
            self._sigma0 = np.array(sigma0, copy=True)
            if np.product(self._sigma0.shape) == self._dimension:
                # Convert from 1d array
                self._sigma0 = self._sigma0.reshape((self._dimension,))
                self._sigma0 = np.diag(self._sigma0)
            else:
                # Check if 2d matrix of correct size
                self._sigma0 = self._sigma0.reshape(
                    (self._dimension, self._dimension))

    def ask(self):
        """
        Returns a sequence of parameter vectors to evaluate a LogPDF for.
        """
        raise NotImplementedError

    def tell(self, fxs):
        """
        Performs an iteration of the MCMC algorithm, using the evaluations
        ``fxs`` of the points previously specified by ``ask``.

        Returns either the next sample in the chain, or ``None`` to indicate
        that no new sample should be added to the chain (this is used to
        implement methods that require multiple evaluations per iteration).

        For methods that require sensitivities (see
        :meth:`MCMCSamper.needs_sensitivities`), ``fxs`` should be a tuple
        ``(log_pdfs, sensitivities)``, containing the values returned by
        :meth:`pints.LogPdf.evaluateS1()`.
        """
        raise NotImplementedError


class MCMCSampling(object):
    """
    Samples from a :class:`pints.LogPDF` using a Markov Chain Monte Carlo
    (MCMC) method.

    The method to use (either a :class:`SingleChainMCMC` class or a
    :class:`MultiChainMCMC` class) is specified at runtime. For example::

        mcmc = pints.MCMCSampling(
            log_pdf, 3, x0, method=pints.AdaptiveCovarianceMCMC)

    Properties related to the number if iterations, parallelisation, and
    logging can be set directly on the ``MCMCSampling`` object, e.g.::

        mcmc.set_max_iterations(1000)

    Sampler specific properties must be set on the internal samplers
    themselves, e.g.::

        for sampler in mcmc.samplers():
            sampler.set_target_acceptance_rate(0.2)

    Finally, to run an MCMC routine, call::

        chains = mcmc.run()

    By default, an MCMCSampling run will write regular progress updates to
    screen. This can be disabled using :meth:`set_log_to_screen()`. To write a
    similar progress log to a file, use :meth:`set_log_to_file()`. To store the
    chains and/or evaluations generated by :meth:`run()` to a file, use
    :meth:`set_chain_filename()` and :meth:`set_log_pdf_filename()`.

    Constructor arguments:

    ``log_pdf``
        A :class:`LogPDF` function that evaluates points in the parameter
        space.
    ``chains``
        The number of MCMC chains to generate.
    ``x0``
        A sequence of starting points. Can be a list of lists, a 2-dimensional
        array, or any other structure such that ``x0[i]`` is the starting point
        for chain ``i``.
    ``sigma0=None``
        An optional initial covariance matrix, i.e., a guess of the covariance
        in ``logpdf`` around the points in ``x0`` (the same ``sigma0`` is used
        for each point in ``x0``).
        Can be specified as a ``(d, d)`` matrix (where ``d`` is the dimension
        of the parameterspace) or as a ``(d, )`` vector, in which case
        ``diag(sigma0)`` will be used.
    ``method``
        The class of :class:`MCMCSampler` to use. If no method is specified,
        :class:`AdaptiveCovarianceMCMC` is used.

    """

    def __init__(self, log_pdf, chains, x0, sigma0=None, method=None):

        # Store function
        if not isinstance(log_pdf, pints.LogPDF):
            raise ValueError('Given function must extend pints.LogPDF')
        self._log_pdf = log_pdf

        # Get number of parameters
        self._dimension = self._log_pdf.n_parameters()

        # Check number of chains
        self._chains = int(chains)
        if self._chains < 1:
            raise ValueError('Number of chains must be at least 1.')

        # Check initial position(s): Most checking is done by samplers!
        if len(x0) != chains:
            raise ValueError(
                'Number of initial positions must be equal to number of'
                ' chains.')
        if not all([len(x) == self._dimension for x in x0]):
            raise ValueError(
                'All initial positions must have the same dimension as the'
                ' given LogPDF.')

        # Don't check initial standard deviation: done by samplers!

        # Set default method
        if method is None:
            method = pints.AdaptiveCovarianceMCMC
        else:
            try:
                ok = issubclass(method, pints.MCMCSampler)
            except TypeError:   # Not a class
                ok = False
            if not ok:
                raise ValueError('Given method must extend pints.MCMCSampler.')

        # Using single chain samplers?
        self._single_chain = issubclass(method, pints.SingleChainMCMC)

        # Create sampler(s)
        if self._single_chain:
            # Using n individual samplers (Note that it is possible to have
            # _single_chain=True and _n_samplers=1)
            self._n_samplers = self._chains
            self._samplers = [method(x, sigma0) for x in x0]
        else:
            # Using a single sampler that samples multiple chains
            self._n_samplers = 1
            self._samplers = [method(self._chains, x0, sigma0)]

        # Check if sensitivities are required
        self._needs_sensitivities = self._samplers[0].needs_sensitivities()

        # Initial phase (needed for e.g. adaptive covariance)
        self._initial_phase_iterations = None
        self._needs_initial_phase = self._samplers[0].needs_initial_phase()
        if self._needs_initial_phase:
            self.set_initial_phase_iterations()

        # Logging
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False
        self.set_log_interval()

        # Writing chains and evaluations to disk
        self._chain_files = None
        self._evaluation_files = None

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

        # TODO: Add more stopping criteria

    def initial_phase_iterations(self):
        """
        For methods that require an initial phase (e.g. an adaptation-free
        phase for the adaptive covariance MCMC method), this returns the number
        of iterations that the initial phase will take.

        For methods that do not require an initial phase, a
        ``NotImplementedError`` is raised.
        """
        return self._initial_phase_iterations

    def max_iterations(self):
        """
        Returns the maximum iterations if this stopping criterion is set, or
        ``None`` if it is not. See :meth:`set_max_iterations()`.
        """
        return self._max_iterations

    def method_needs_initial_phase(self):
        """
        Returns true if this sampler has been created with a method that has
        an initial phase (see :meth:`MCMCSampler.needs_initial_phase()`.)
        """
        return self._samplers[0].needs_initial_phase()

    def parallel(self):
        """
        Returns the number of parallel worker processes this routine will be
        run on, or ``False`` if parallelisation is disabled.
        """
        return self._n_workers if self._parallel else False

    def run(self):
        """
        Runs the MCMC sampler(s) and returns a number of markov chains, each
        representing the distribution of the given log-pdf.
        """
        # Check stopping criteria
        has_stopping_criterion = False
        has_stopping_criterion |= (self._max_iterations is not None)
        if not has_stopping_criterion:
            raise ValueError('At least one stopping criterion must be set.')

        # Iteration and evaluation counting
        iteration = 0
        evaluations = 0

        # Choose method to evaluate
        f = self._log_pdf
        if self._needs_sensitivities:
            f = f.evaluateS1

        # Create evaluator object
        if self._parallel:
            # Use at most n_workers workers
            n_workers = min(self._n_workers, self._chains)
            evaluator = pints.ParallelEvaluator(f, n_workers=n_workers)
        else:
            evaluator = pints.SequentialEvaluator(f)

        # Initial phase
        if self._needs_initial_phase:
            for sampler in self._samplers:
                sampler.set_initial_phase(True)

        # Write chains to disk
        chain_loggers = []
        if self._chain_files:
            for filename in self._chain_files:
                cl = pints.Logger()
                cl.set_stream(None)
                cl.set_filename(filename, True)
                for k in range(self._dimension):
                    cl.add_float('p' + str(k))
                chain_loggers.append(cl)

        # Write evaluations to disk
        eval_loggers = []
        if self._evaluation_files:
            # Bayesian inference on a log-posterior? Then separate out the
            # prior so we can calculate the loglikelihood
            prior = None
            if isinstance(self._log_pdf, pints.LogPosterior):
                prior = self._log_pdf.log_prior()

            # Set up loggers
            for filename in self._evaluation_files:
                cl = pints.Logger()
                cl.set_stream(None)
                cl.set_filename(filename, True)
                if prior:
                    # Logposterior in first column, to be consistent with the
                    # non-bayesian case
                    cl.add_float('logposterior')
                    cl.add_float('loglikelihood')
                    cl.add_float('logprior')
                else:
                    cl.add_float('logpdf')
                eval_loggers.append(cl)

            # Store last accepted logpdf, per chain
            current_logpdf = np.zeros(self._chains)
            current_prior = np.zeros(self._chains)

        # Set up progress reporting
        next_message = 0

        # Start logging
        logging = self._log_to_screen or self._log_filename
        if logging:
            if self._log_to_screen:
                print('Using ' + str(self._samplers[0].name()))
                print('Generating ' + str(self._chains) + ' chains.')
                if self._parallel:
                    print('Running in parallel with ' + str(n_workers) +
                          ' worker processess.')
                else:
                    print('Running in sequential mode.')
                if self._chain_files:
                    print(
                        'Writing chains to ' + self._chain_files[0] + ' etc.')
                if self._evaluation_files:
                    print(
                        'Writing evaluations to ' + self._evaluation_files[0]
                        + ' etc.')

            # Set up logger
            logger = pints.Logger()
            if not self._log_to_screen:
                logger.set_stream(None)
            if self._log_filename:
                logger.set_filename(self._log_filename, csv=self._log_csv)

            # Add fields to log
            max_iter_guess = max(self._max_iterations or 0, 10000)
            max_eval_guess = max_iter_guess * self._chains
            logger.add_counter('Iter.', max_value=max_iter_guess)
            logger.add_counter('Eval.', max_value=max_eval_guess)
            for sampler in self._samplers:
                sampler._log_init(logger)
            logger.add_time('Time m:s')

        # Create chains
        # TODO Pre-allocate?
        chains = []

        # Start sampling
        timer = pints.Timer()
        running = True
        while running:
            # Initial phase
            # Note: self._initial_phase_iterations is None when no initial
            # phase is needed
            if iteration == self._initial_phase_iterations:
                for sampler in self._samplers:
                    sampler.set_initial_phase(False)
                if self._log_to_screen:
                    print('Initial phase completed.')

            # Get points
            if self._single_chain:
                xs = [sampler.ask() for sampler in self._samplers]
            else:
                xs = self._samplers[0].ask()

            # Calculate logpdfs
            fxs = evaluator.evaluate(xs)

            # Update evaluation count
            evaluations += len(fxs)

            # Update chains
            intermediate_step = False
            if self._single_chain:
                samples = np.array([
                    s.tell(fxs[i]) for i, s in enumerate(self._samplers)])

                none_found = [x is None for x in samples]
                if any(none_found):
                    # Can't mix None w. samples
                    assert(all(none_found))
                    intermediate_step = True
            else:
                samples = self._samplers[0].tell(fxs)
                intermediate_step = samples is None

            # If no new samples were added, then no MCMC iteration was
            # performed, and so the iteration count shouldn't be updated,
            # logging shouldn't be triggered, and stopping criteria shouldn't
            # be checked
            if intermediate_step:
                continue

            # Add new samples to the chains
            chains.append(samples)

            # Write samples to disk
            for k, chain_logger in enumerate(chain_loggers):
                chain_logger.log(*samples[k])

            # Write evaluations to disk
            if self._evaluation_files:
                for k, eval_logger in enumerate(eval_loggers):
                    if np.all(xs[k] == samples[k]):
                        current_logpdf[k] = fxs[k]
                        if prior is not None:
                            current_prior[k] = prior(xs[k])
                    eval_logger.log(current_logpdf[k])
                    if prior is not None:
                        eval_logger.log(current_logpdf[k] - current_prior[k])
                        eval_logger.log(current_prior[k])

            # Show progress
            if logging and iteration >= next_message:
                # Log state
                logger.log(iteration, evaluations)
                for sampler in self._samplers:
                    sampler._log_write(logger)
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

            # TODO Add more stopping criteria

        # Log final state and show halt message
        if logging:
            logger.log(iteration, evaluations)
            for sampler in self._samplers:
                sampler._log_write(logger)
            logger.log(timer.time())
            if self._log_to_screen:
                print(halt_message)

        # Swap axes in chains, to get indices
        #  [chain, iteration, parameter]
        chains = np.array(chains)
        chains = chains.swapaxes(0, 1)

        # Return generated chains
        return chains

    def sampler(self):
        """
        Returns the underlying :class:`MultiChainMCMC` object, or raises an
        error if :class:`SingleChainMCMC` objects are being used.

        See also: :meth:`samplers()`.
        """
        if self._single_chain:
            raise RuntimeError(
                'The method MCMCSampling.sampler() is only supported when a'
                ' MultiChainMCMC is selected. Please use'
                ' MCMCSampling.samplers() instead, to obtain a list of all'
                ' internal SingleChainMCMC instances.')
        return self._samplers[0]

    def samplers(self):
        """
        Returns a list containing the underlying sampler objects.

        If a :class:`SingleChainMCMC` method was selected, this will be a list
        containing as many :class:`SingleChainMCMC` objects as the number of
        chains. If a :class:`MultiChainMCMC` method was selected, this will be
        a list containing a single :class:`MultiChainMCMC` instance.
        """
        return self._samplers

    def set_chain_filename(self, chain_file):
        """
        Write chains to disk as they are generated.

        If a ``chain_file`` is specified, a CSV file will be created for each
        chain, to which samples will be written as they are accepted. To
        disable logging of chains, set ``chain_file=None``.

        Filenames for each chain file will be derived from ``chain_file``, e.g.
        if ``chain_file='chain.csv'`` and there are 2 chains, then the files
        ``chain_0.csv`` and ``chain_1.csv`` will be created. Each CSV file will
        start with a header (e.g. ``"p0","p1","p2",...``) and contain a sample
        on each subsequent line.
        """

        d = self._chains
        self._chain_files = None
        if chain_file:
            b, e = os.path.splitext(str(chain_file))
            self._chain_files = [b + '_' + str(i) + e for i in range(d)]

    def set_log_pdf_filename(self, log_pdf_file):
        """
        Write :class:`LogPDF` evaluations to disk as they are generated.

        If an ``evaluation_file`` is specified, a CSV file will be created for
        each chain, to which :class:`LogPDF` evaluations will be written for
        every accepted sample. To disable this feature, set
        ``evaluation_file=None``. If the ``LogPDF`` being evaluated is a
        :class:`LogPosterior`, the individual likelihood and prior will also
        be stored.

        Filenames for each evaluation file will be derived from
        ``evaluation_file``, e.g. if ``evaluation_file='evals.csv'`` and there
        are 2 chains, then the files ``evals_0.csv`` and ``evals_1.csv`` will
        be created. Each CSV file will start with a header (e.g.
        ``"logposterior","loglikelihood","logprior"``) and contain the
        evaluations for i-th accepted sample on the i-th subsequent line.
        """

        d = self._chains
        self._evaluation_files = None
        if log_pdf_file:
            b, e = os.path.splitext(str(log_pdf_file))
            self._evaluation_files = [b + '_' + str(i) + e for i in range(d)]

    def set_initial_phase_iterations(self, iterations=200):
        """
        For methods that require an initial phase (e.g. an adaptation-free
        phase for the adaptive covariance MCMC method), this sets the number of
        iterations that the initial phase will take.

        For methods that do not require an initial phase, a
        ``NotImplementedError`` is raised.
        """
        if not self._needs_initial_phase:
            raise NotImplementedError

        # Check input
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError(
                'Number of initial-phase iterations cannot be negative.')
        self._initial_phase_iterations = iterations

    def set_log_interval(self, iters=20, warm_up=3):
        """
        Changes the frequency with which messages are logged.

        Arguments:

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
        Enables progress logging to file when a filename is passed in, disables
        it if ``filename`` is ``False`` or ``None``.

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
        Enables or disables progress logging to screen.
        """
        self._log_to_screen = True if enabled else False

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


def mcmc_sample(log_pdf, chains, x0, sigma0=None, method=None):
    """
    Sample from a :class:`pints.LogPDF` using a Markov Chain Monte Carlo
    (MCMC) method.

    Arguments:

    ``log_pdf``
        A :class:`LogPDF` function that evaluates points in the parameter
        space.
    ``chains``
        The number of MCMC chains to generate.
    ``x0``
        A sequence of starting points. Can be a list of lists, a 2-dimensional
        array, or any other structure such that ``x0[i]`` is the starting point
        for chain ``i``.
    ``sigma0=None``
        An optional initial covariance matrix, i.e., a guess of the covariance
        in ``logpdf`` around the points in ``x0`` (the same ``sigma0`` is used
        for each point in ``x0``).
        Can be specified as a ``(d, d)`` matrix (where ``d`` is the dimension
        of the parameterspace) or as a ``(d, )`` vector, in which case
        ``diag(sigma0)`` will be used.
    ``method``
        The class of :class:`MCMCSampler` to use. If no method is specified,
        :class:`AdaptiveCovarianceMCMC` is used.
    """
    return MCMCSampling(    # pragma: no cover
        log_pdf, chains, x0, sigma0, method).run()
