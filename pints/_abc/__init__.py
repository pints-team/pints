#
# Sub-module containing ABC inference routines
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class ABCSampler(pints.Loggable, pints.TunableMethod):
    """
    Abstract base class for ABC methods.
    All ABC samplers implement the :class:`pints.Loggable` and
    :class:`pints.TunableMethod` interfaces.
    """
    def name(self):
        """
        Returns this method's full name.
        """
        raise NotImplementedError

    def ask(self):
        """
        Returns a parameter vector sampled from a prior
        """
        raise NotImplementedError

    def tell(self, fx):
        """
        Performs an iteration of the ABC-rejection algorithm, using the
        parameters specified by ask.

        Returns the accepted parameter values, or ``None`` to indicate
        that no parameters were accepted (tell allows for multiple evaluations
        per iteration).

        """
        raise NotImplementedError


class ABCController(object):
    """
    Samples from a :class:`pints.LogPrior`.

    Properties related to the number of iterations, parallelisation,
    threshold, and number of parameters to sample can be set directly on the
    ``ABCController`` object, e.g.::

        abc.set_max_iterations(1000)

    Finally, to run an ABC routine, call::

        posterior_estimate = abc.run()

    Constructor arguments:
    ``error_measure``
    An error measure to evaluate on a problem, given a forward model,
    simulated and observed data, and times

    ``log_prior``
    A :class:`LogPrior` function from which parameter values are sampled

    ``method``
    The class of :class:`ABCSampler` to use. If no method is specified,
    :class:`RejectionABC` is used.
    """
    def __init__(self, error_measure, log_prior, method=None):

        # Store function
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError('Given function must extend pints.LogPrior')
        self._log_prior = log_prior

        # Check error_measure
        # if not isinstance(error_measure, pints.ErrorMeasure):
        # raise ValueError('Given error_measure must extend
        # pints.ErrorMeasure')
        self._error_measure = error_measure

        # Check if number of parameters from prior matches that of error
        # measure
        if self._log_prior.n_parameters() != \
                self._error_measure.n_parameters():
            raise ValueError('Number of parameters in prior must match number '
                             'of parameters in model')

        # Get number of parameters
        self._n_parameters = self._log_prior.n_parameters()

        # Set default method
        if method is None:
            method = pints.RejectionABC
        else:
            try:
                ok = issubclass(method, pints.ABCSampler)
            except TypeError:   # Not a class
                ok = False
            if not ok:
                raise ValueError('Given method must extend pints.ABCSampler.')

        # Initialisation
        self._parallel = False
        self._n_workers = 1
        self._max_iterations = 10000
        self._n_target = 500
        self._sampler = method(log_prior)
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False
        self.set_log_interval()
        self._acceptance_rate = 0

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

    def max_iterations(self):
        """
        Returns the maximum iterations if this stopping criterion is set, or
        ``None`` if it is not. See :meth:`set_max_iterations()`.
        """
        return self._max_iterations

    def n_target(self):
        """
        Returns the target number of samples to obtain in the estimated
        posterior.
        """
        return self._n_target

    def parallel(self):
        """
        Returns the number of parallel worker processes this routine will be
        run on, or ``False`` if parallelisation is disabled.
        """
        return self._n_workers if self._parallel else False

    def run(self):
        """
        Runs the ABC sampler.
        """
        # Check stopping criteria
        has_stopping_criterion = False
        has_stopping_criterion |= (self._max_iterations is not None)
        if not has_stopping_criterion:
            raise ValueError('At least one stopping criterion must be set.')

        # Iteration and evaluation counting
        iteration = 0
        evaluations = 0
        accepted_count = 0

        # Choose method to evaluate
        f = self._error_measure

        # Create evaluator object
        if self._parallel:
            # Use at most n_workers workers
            n_workers = self._n_workers
            evaluator = pints.ParallelEvaluator(f, n_workers=n_workers)
        else:
            evaluator = pints.SequentialEvaluator(f)

        # Set up progress reporting
        next_message = 0

        # Start logging
        logging = self._log_to_screen or self._log_filename
        if logging:
            if self._log_to_screen:
                print('Using ' + str(self._sampler.name()))
                if self._parallel:
                    print('Running in parallel with ' + str(n_workers) +
                          ' worker processess.')
                else:
                    print('Running in sequential mode.')

            # Set up logger
            logger = pints.Logger()
            if not self._log_to_screen:
                logger.set_stream(None)
            if self._log_filename:
                logger.set_filename(self._log_filename, csv=self._log_csv)

            # Add fields to log
            max_iter_guess = max(self._max_iterations or 0, 10000)
            max_eval_guess = max_iter_guess
            logger.add_counter('Iter.', max_value=max_iter_guess)
            logger.add_counter('Eval.', max_value=max_eval_guess)
            logger.add_float('Acceptance rate')
            self._sampler._log_init(logger)
            logger.add_time('Time m:s')

        # Start sampling
        timer = pints.Timer()
        running = True

        samples = []
        while running:
            # Sample until a given sample is accepted

            # Get points from prior
            xs = self._sampler.ask(self._n_workers)

            # Simulate and get error
            fxs = evaluator.evaluate(xs)

            evaluations += self._n_workers

            # Tell sampler errors and get list of acceptable parameters back
            accepted_vals = self._sampler.tell(fxs)

            if accepted_vals is not None:
                accepted_count += len(accepted_vals)
            while accepted_vals is None:
                xs = self._sampler.ask(self._n_workers)
                fxs = evaluator.evaluate(xs)
                accepted_vals = self._sampler.tell(fxs)
                evaluations += self._n_workers
                if accepted_vals is not None:
                    accepted_count += len(accepted_vals)
            for val in accepted_vals:
                samples.append(val)
            iteration += 1

            # Show progress
            if logging and iteration >= next_message:
                # Log state
                logger.log(iteration, evaluations, (
                           accepted_count / evaluations))
                self._sampler._log_write(logger)
                logger.log(timer.time())

                # Choose next logging point
                if iteration < self._message_warm_up:
                    next_message = iteration + 1
                else:
                    next_message = self._message_interval * (
                        1 + iteration // self._message_interval)

            # Check requested number of samples
            if (self._max_iterations is not None and
                    iteration >= self._max_iterations):
                running = False
                halt_message = ('Halting: Maximum number of iterations ('
                                + str(iteration) + ') reached.')
            elif accepted_count >= self._n_target:
                running = False
                halt_message = ('Halting: target number of samples ('
                                + str(accepted_count) + ') reached.')

        # Log final state and show halt message
        if logging:
            logger.log(iteration, evaluations)
            self._sampler._log_write(logger)
            logger.log(timer.time())
            if self._log_to_screen:
                print(halt_message)
        samples = np.array(samples)
        return samples

    def log_filename(self):
        """
        Returns log filename.
        """
        return self._log_filename

    def sampler(self):
        """
        Returns the underlying sampler object.
        """
        return self._sampler

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

    def set_n_target(self, n_target=500):
        """
        Sets a target number of samples
        """
        self._n_target = n_target

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


class SequentialABCController(object):
    """
    Samples from a :class:`pints.LogPrior`.

    Properties related to the number of iterations, parallelisation,
    threshold, and number of parameters to sample can be set directly on the
    ``ABCController`` object, e.g.::

        abc.set_max_iterations(1000)

    Finally, to run an ABC routine, call::

        posterior_estimate = abc.run()

    The following pluggable components are used in sequential ABC implementations
    each sampler has a set of default components but they can be swapped out
    to produce new algorithms

    Cooling Schedule:
        - Takes t (and reference to controller)
        - Provides epsilon for sampler

    Acceptance Kernel:
        - Take particle, epsilon (and reference to controller)
        - Return probability of acceptance

    Perturbation Kernel:
        - Takes particle (and reference to controller)
        - Return new particle

    Constructor arguments:
    ``error_measure``
    An error measure to evaluate on a problem, given a forward model,
    simulated and observed data, and times

    ``log_prior``
    A :class:`LogPrior` function from which parameter values are sampled

    ``method``
    The class of :class:`ABCSampler` to use. If no method is specified,
    :class:`RejectionABC` is used.
    """
    def __init__(self, error_measure, log_prior, method=None, perturbation_kernel=None, acceptance_kernel=None):

        # Store function
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError('Given function must extend pints.LogPrior')
        self._log_prior = log_prior

        # Check error_measure
        # if not isinstance(error_measure, pints.ErrorMeasure):
        # raise ValueError('Given error_measure must extend
        # pints.ErrorMeasure')
        self._error_measure = error_measure

        # Check if number of parameters from prior matches that of error
        # measure
        if self._log_prior.n_parameters() != \
                self._error_measure.n_parameters():
            raise ValueError('Number of parameters in prior must match number '
                             'of parameters in model')

        # Get number of parameters
        self._n_parameters = self._log_prior.n_parameters()

        # Set default method
        if method is None:
            method = pints.RejectionABC
        else:
            try:
                ok = issubclass(method, pints.ABCSampler)
            except TypeError:   # Not a class
                ok = False
            if not ok:
                raise ValueError('Given method must extend pints.ABCSampler.')

        # Initialisation
        self._parallel = False
        self._n_workers = 1
        self._n_target = 500
        self._sampler = method(log_prior)
        self._scale_mat = None
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False
        self.set_log_interval()
        self._acceptance_rate = 0
        self._k_min = 0.5
        self._I_max = 10000
        self._cooling_schedule = 0.25
        self._epsilon_null = 4
        self._t = 0
        self._rejection_sampler = self._sampler if method == pints.RejectionABC else pints.RejectionABC(log_prior)
        self._perturbation_kernel = perturbation_kernel or pints.SphericalGaussianKernel(0.003)
        self._acceptance_kernel = acceptance_kernel or (lambda error, epsilon: 1 if error < epsilon else 0)

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

    def max_iterations(self):
        """
        Returns the maximum iterations if this stopping criterion is set, or
        ``None`` if it is not. See :meth:`set_max_iterations()`.
        """
        return self._I_max

    def n_target(self):
        """
        Returns the target number of samples to obtain in the estimated
        posterior.
        """
        return self._n_target

    def parallel(self):
        """
        Returns the number of parallel worker processes this routine will be
        run on, or ``False`` if parallelisation is disabled.
        """
        return self._n_workers if self._parallel else False

    def set_cooling_schedule(self, schedule):
        self._cooling_schedule = schedule

    def set_initial_threshold(self, threshold):
        self._epsilon_null = threshold

    def set_cooling_limit(self, limit):
        self._k_min = limit

    def run(self):
        """
        Runs the ABC sampler.
        """
        # Check stopping criteria
        has_stopping_criterion = False
        has_stopping_criterion |= (self._k_min is not None)
        if not has_stopping_criterion:
            raise ValueError('At least one stopping criterion must be set.')

        # Evaluation counting
        evaluations = 0
        last_evals = 0
        accepted_count = [0]

        # Choose method to evaluate
        f = self._error_measure

        # Create evaluator object
        if self._parallel:
            # Use at most n_workers workers
            n_workers = self._n_workers
            evaluator = pints.ParallelEvaluator(f, n_workers=n_workers)
        else:
            evaluator = pints.SequentialEvaluator(f)

        # Set up progress reporting
        next_message = 0

        # Start logging
        logging = self._log_to_screen or self._log_filename
        if logging:
            if self._log_to_screen:
                print('Using ' + str(self._sampler.name()))
                if self._parallel:
                    print('Running in parallel with ' + str(n_workers) +
                          ' worker processess.')
                else:
                    print('Running in sequential mode.')

            # Set up logger
            logger = pints.Logger()
            if not self._log_to_screen:
                logger.set_stream(None)
            if self._log_filename:
                logger.set_filename(self._log_filename, csv=self._log_csv)

            # Add fields to log
            max_eval_guess = 10000
            logger.add_counter('Eval.', max_value=max_eval_guess)
            logger.add_time('Time m:s')

        # Start sampling
        timer = pints.Timer()
        running = True
        samples = [[]]
        weights = [[]]
        k = self._epsilon_null
        epsilon = [self._epsilon_null]
        self._rejection_sampler.set_threshold(self._epsilon_null)
        # While we are still cooling
        while k > self._k_min:
            iterations = 0
            if self._t == 0:
                # Draw self._n_target points using rejection sampling
                while accepted_count[self._t] < self._n_target:
                    # Get points from prior
                    xs = self._rejection_sampler.ask(self._n_workers)
                    # Simulate and get error
                    fxs = evaluator.evaluate(xs)
                    evaluations += self._n_workers

                    accepted_vals = self._rejection_sampler.tell(fxs)
                    if accepted_vals is not None:
                        accepted_count[self._t] += len(accepted_vals)
                    while accepted_vals is None:
                        xs = self._rejection_sampler.ask(self._n_workers)
                        fxs = evaluator.evaluate(xs)
                        accepted_vals = self._rejection_sampler.tell(fxs)
                        evaluations += self._n_workers
                        if accepted_vals is not None:
                            accepted_count[self._t] += len(accepted_vals)
                    for val in accepted_vals:
                        samples[self._t].append(val)

                # Once we have the sample then create an intermediate distribution with uniform weights
                weights[self._t] = np.full(self._n_target, 1/self._n_target)

            else:
                # Draw self._n_targets points from the intermediate distribution in samples
                while len(samples[self._t]) < self._n_target:
                    for i in range(0, self._n_target):
                        first_time = True
                        while first_time or self._log_prior(theta_star_star) == -np.inf:
                            # weighted sample from previous intermediate
                            # TODO: collect multiple samples for parallel mode
                            try:
                                theta_star = samples[self._t-1][np.random.choice(range(len(samples[self._t-1])),
                                                                                p=weights[self._t-1])]
                            except ValueError:
                                print(f"np.random.choice: a (size {len(range(len(samples[self._t-1])))}) and "
                                      f"p (size {len(weights[self._t-1])}) must be the same size")
                                raise
                            iterations += 1
                            # perturb using _K_t
                            # TODO: Allow perturbation kernel to adapt to intermediate distributions
                            theta_star_star = self._perturbation_kernel.perturb(theta_star)

                            # check if theta_star_star is possible under the prior and sample again if not
                            first_time = False

                        fx = evaluator.evaluate([theta_star_star])
                        evaluations += 1
                        if np.random.binomial(1, self._acceptance_kernel(fx[0], epsilon[self._t])):
                            samples[self._t].append(theta_star_star)
                            break
                        elif iterations > self._I_max:
                            samples[self._t] = []
                            k = self._cooling_schedule*k
                            iterations = 0
                            epsilon[self._t] = epsilon[self._t-1] - k
                            print("Hitting max iterations at epsilon="+str(epsilon[self._t]))
                            break

                # After we have self._n_targets samples
                for i in range(0, self._n_target):
                    # Calculate weights according to the Toni algorithm
                    w = np.exp(self._log_prior(samples[self._t][i])) / \
                        sum([weights[self._t-1][j]*self._perturbation_kernel.p(samples[self._t][i], samples[self._t-1][j]) for j in range(self._n_target)])
                    weights[self._t].append(w)

            # Normalise weights
            normal = sum(weights[self._t])
            weights[self._t] = [w/normal for w in weights[self._t]]

            # Cool the system
            k = min(k, self._cooling_schedule * epsilon[self._t])
            epsilon.append(epsilon[self._t]-k)

            # Advance time
            print(f"Time {self._t}: {len(samples[self._t])} samples found at threshold {epsilon[self._t]} "
                  f"within {evaluations-last_evals} evals")
            last_evals = evaluations
            self._t += 1
            samples.append([])
            weights.append([])

        # Log final state and show halt message
        if logging:
            logger.log(evaluations)
            logger.log(timer.time())
            if self._log_to_screen:
                print("Sampling Complete")
        samples = np.array(samples)
        return samples



        #
        #     # Show progress
        #     if logging and iteration >= next_message:
        #         # Log state
        #         logger.log(iteration, evaluations, (
        #                    accepted_count / evaluations))
        #         self._sampler._log_write(logger)
        #         logger.log(timer.time())
        #
        #         # Choose next logging point
        #         if iteration < self._message_warm_up:
        #             next_message = iteration + 1
        #         else:
        #             next_message = self._message_interval * (
        #                 1 + iteration // self._message_interval)
        #
        #     # Check requested number of samples
        #     if (self._max_iterations is not None and
        #             iteration >= self._max_iterations):
        #         running = False
        #         halt_message = ('Halting: Maximum number of iterations ('
        #                         + str(iteration) + ') reached.')
        #     elif accepted_count >= self._n_target:
        #         running = False
        #         halt_message = ('Halting: target number of samples ('
        #                         + str(accepted_count) + ') reached.')

    def log_filename(self):
        """
        Returns log filename.
        """
        return self._log_filename

    def sampler(self):
        """
        Returns the underlying sampler object.
        """
        return self._rejection_sampler

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
        self._I_max = iterations

    def set_n_target(self, n_target=500):
        """
        Sets a target number of samples
        """
        self._n_target = n_target

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
        raise NotImplementedError("ABC-SMC does not currently support parallel computation")