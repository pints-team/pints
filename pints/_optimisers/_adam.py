#
# Uses the Python `cma` module to runs CMA-ES optimisations.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import logging
import numpy as np
import pints


class AdaptiveMomentEstimation(pints.Optimiser):
    """
    Finds the best parameters using the Adam method, as described in [1]

    Adam stands for Adaptive Moment Estimation, it is a stochastic gradient
    descent algorithm that uses a noisy gradient of the error function

    *Extends:* :class:`Optimiser`

    [1] Diederik, Kingma; Ba, Jimmy (2014).
        "Adam: A method for stochastic optimization". arXiv:1412.6980

    """

    def __init__(self, x0, sigma0=None, boundaries=None):
        super(AdaptiveMomentEstimation, self).__init__(x0, sigma0, boundaries)

        self._ready_for_tell = False

        # default hyper-parameters
        if sigma0 is not None:
            self.set_alpha(0.01*np.min(sigma0))
        else:
            self.set_alpha()
        self.set_beta1()
        self.set_beta2()

        # default behaviour is to use function evaluations
        self._ignore_fbest = False

        # init best function eval
        self._fbest = float('inf')

        # init state variables
        self._m = 0.0
        self._v = 0.0
        self._t = 0

        # small epsilon to prevent divide by zero
        self._epsilon = 1e-8

        # boundaries check flag
        self._outside_boundaries = False

        # Best solution found
        self._xbest = pints.vector(x0)
        self._m_hat = float('inf')

        # Python logger
        self._logger = logging.getLogger(__name__)

    def set_alpha(self, alpha=0.001):
        """Should be non-negative"""
        if alpha < 0:
            raise ValueError("alpha should be non-negative")
        self._alpha = alpha

    def set_beta1(self, beta1=0.9):
        """Should be in the range [0, 1)"""
        if beta1 < 0.0 or beta1 >= 1.0:
            raise ValueError("beta1 should be in the range [0, 1)")
        self._beta1 = beta1

    def set_beta2(self, beta2=0.99):
        """Should be in the range [0, 1)"""
        if beta2 < 0.0 or beta2 >= 1.0:
            raise ValueError("beta2 should be in the range [0, 1)")
        self._beta2 = beta2

    def alpha(self):
        return self._alpha

    def beta1(self):
        return self._beta1

    def beta2(self):
        return self._beta2

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 3

    def set_hyper_parameters(self, x):
        """
        See :meth:`TunableMethod.set_hyper_parameters()`.

        Arguments:
            ``x``: array of [alpha, beta1, beta2]

        """
        self.set_alpha(x[0])
        self.set_beta1(x[1])
        self.set_beta2(x[2])

    def set_ignore_fbest(self, ignore=True):
        self._ignore_fbest = ignore

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """
        # Ready for tell now
        self._ready_for_tell = True
        return [self._xbest]

    def tell(self, fx):
        """ See :meth:`Optimiser.tell()`. """
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        self._t += 1
        self._fbest, gradf = fx[0]
        self._m = self._beta1*self._m + (1-self._beta1)*gradf
        self._v = self._beta2*self._v + (1-self._beta2)*gradf**2
        self._m_hat = self._m / (1 - self._beta1**self._t)
        self._v_hat = self._v / (1 - self._beta2**self._t)
        self._xbest = self._xbest - self._alpha * \
            self._m_hat / (np.sqrt(self._v_hat) + self._epsilon)

        if self._boundaries is not None and not self._boundaries.check(self._xbest):
            self._outside_boundaries = True

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'Adaptive Moment Estimation (Adam)'

    def running(self):
        """ See :meth:`Optimiser.running()`. """
        return True

    def stop(self):
        """ See :meth:`Optimiser.stop()`. """
        return self._outside_boundaries

    def xbest(self):
        """ See :meth:`Optimiser.xbest()`. """
        return np.array(self._xbest, copy=True)

    def fbest(self):
        """
        See :meth:`Optimiser.fbest()`

        use norm(gradient) as a proxy for fbest if the function evaluations are being
        ignored
        """
        if self._ignore_fbest:
            return np.linalg.norm(self._m_hat)
        else:
            return self._fbest

    def needs_sensitivities(self):
        """ See :meth:`Optimiser.needs_sensitivities()`. """
        return True
