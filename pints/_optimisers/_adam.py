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
    Finds the best parameters using the Adam method described in [1]

    Adam stands for Adaptive Moment Estimation, it is a stochastic gradient descent
    algorithm that only uses the gradient of the error function

    *Extends:* :class:`Optimiser`

    [1] Diederik, Kingma; Ba, Jimmy (2014). "Adam: A method for stochastic optimization". arXiv:1412.6980

    """

    def __init__(self, x0, sigma0=None, boundaries=None):
        super(AdaptiveMomentEstimation, self).__init__(x0, sigma0, boundaries)

        # default hyper-parameters
        set_alpha()
        set_beta1()
        set_beta2()

        # init state variables
        self._m0 = 0.0
        self._v0 = 0.0

        # Best solution found
        self._xbest = pints.vector(x0)
        self._fbest = float('inf')

        # Python logger
        self._logger = logging.getLogger(__name__)

    def set_alpha(self, alpha=0.001):
        self._alpha = alpha

    def set_beta1(self, beta1=0.9):
        self._beta1 = beta1

    def set_beta2(self, beta2=0.99):
        self._beta2 = beta2

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """
        self._xbest.setflags(write=False)
        return self._xbest

    def tell(self, fx):
        """ See :meth:`Optimiser.tell()`. """

    def fbest(self):
        """ See :meth:`Optimiser.fbest()`. """
        return self._fbest

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'Adaptive Moment Estimation (Adam)'

    def running(self):
        """ See :meth:`Optimiser.running()`. """
        return True

    def stop(self):
        """ See :meth:`Optimiser.stop()`. """
        return False

    def xbest(self):
        """ See :meth:`Optimiser.xbest()`. """
        return np.array(self._xbest, copy=True)
