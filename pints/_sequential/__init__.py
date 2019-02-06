#
# Sub-module containing sequential MC inference routines
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


class SMCSampler(object):
    """
    Abstract base class for Sequential Monte Carlo (SMC) samplers.

    Arguments:

    ``log_pdf``
        A :class:`LogPDF` function that evaluates points in the parameter
        space.
    ``log_prior``
        A :class:`LogPrior` on the same parameter space, used to draw proposals
        from.
    ``x0``
        An initial guess, or starting point for the SMC routine.


    #TODO: CURRENTLY WRITTEN AS A SAMPL(ER+ING)


    #TODO REMAINING ARGUMENTS




    """
    def __init__(self, log_pdf, log_prior, x0, sigma0=None):

        # Store log likelihood
        if not isinstance(log_pdf, pints.LogPDF):
            raise ValueError(
                'Given log_pdf function must extend pints.LogPDF')
        self._log_pdf = log_pdf
        self._n_parameters = log_pdf.n_parameters()

        # Store log prior
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError(
                'Given log_prior function must extend pints.LogPrior')
        if log_prior.n_parameters() != self._n_parameters:
            raise ValueError(
                'Given log_pdf and log_prior must have same number of'
                ' parameters.')
        self._log_prior = log_prior

        # Store initial position
        x0 = pints.vector(x0)
        if len(x0) != self._n_parameters:
            raise ValueError(
                'Given intial point has length ' + str(len(x0))
                + ', while log_pdf and log_prior are expecting '
                + str(self._n_parameters) + ' parameters.')
        self._x0 = x0

        #TODO: SIGMA

        # Check standard deviation, set self._sigma0
        if sigma0 is None:
            # Get representative parameter value for each parameter
            self._sigma0 = np.abs(self._x0)
            self._sigma0[self._sigma0 == 0] = 1
            # Use to create diagonal matrix
            self._sigma0 = np.diag(0.01 * self._sigma0)
        else:
            self._sigma0 = np.array(sigma0)
            if np.product(self._sigma0.shape) == self._n_parameters:
                # Convert from 1d array
                self._sigma0 = self._sigma0.reshape((self._n_parameters,))
                self._sigma0 = np.diag(self._sigma0)
            else:
                # Check if 2d matrix of correct size
                self._sigma0 = self._sigma0.reshape(
                    (self._n_parameters, self._n_parameters))

    def name(self):
        """
        Returns this method's full name.
        """
        raise NotImplementedError

    def run(self):
        """
        Runs SMC, returns samples.
        """
        #TODO: Replace this with an ask-tell interface
        raise NotImplementedError


'''
class SMCSampling(object): #TODO WRITE DOCSTRING. LOGPOSTERIOR ONLY?
    """
    Samples from a :class:`pints.LogPDF` using a Sequential Markov Chain Monte
    Carlo (SMC) method.

    The method to use (either a :class:`SingleChainMCMC` class or a
    :class:`MultiChainMCMC` class) is specified at runtime. For example::

        mcmc = pints.MCMCSampling(
            log_pdf, 3, x0, method=pints.AdaptiveCovarianceMCMC)

    TODO
    """
    def __init__(self, log_posterior, x0, sigma=None, method=None):

        # Store function
        #TODO: IS THIS WHAT WE WANT?
        if not isinstance(log_posterior, pints.LogPosterior):
            raise ValueError('Given function must extend pints.LogPosterior')
        self._log_prior = log_posterior.log_prior()
        self._log_likelihood = log_posterior.log_likelihood()

        # Get number of parameters
        #self._n_parameters = self._log_pdf.n_parameters()

        # Check initial position: Most checking is done by samplers!
        #TODO: Assuming 1 chain for now?
        #if len(x0) != chains:
        #    raise ValueError(
        #        'Number of initial positions must be equal to number of'
        #        ' chains.')
        #if not all([len(x) == self._n_parameters for x in x0]):
        #    raise ValueError(
        #        'All initial positions must have the same dimension as the'
        #        ' given LogPDF.')

        # Don't check initial standard deviation: done by samplers!

        # Set default method
        if method is None:
            method = pints.SMC
        else:
            try:
                ok = issubclass(method, pints.SMCSampler)
            except TypeError:   # Not a class
                ok = False
            if not ok:
                raise ValueError('Given method must extend pints.SMCSampler.')

        # Logging
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False
        self.set_log_interval()

        # Writing to disk
        #TODO?

        # Parallelisation
        self._parallel = False
        self._n_workers = 1
        self.set_parallel()

        # Stopping criteria
        #TODO?


    def parallel(self):
        """
        Returns the number of parallel worker processes this routine will be
        run on, or ``False`` if parallelisation is disabled.
        """
        return self._n_workers if self._parallel else False


    def run(self):

        #TODO: Convert to ask/tell, and move loop into this run method.

        raise NotImplementedError


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

'''
