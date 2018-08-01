#
# Egg-box log-likelihood/PDF
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import scipy.stats


class MultimodalNormalLogPDF(pints.LogPDF):
    """
    *Extends:* :class:`pints.LogPDF`.
    
    Eggbox distribution defined by log of eqn. (31) in [1],
    
    .. math::
        log(p(x,y)) = 5(2 + cos(x/2)cos(y/2))
    
    where x and y are bounded between 0 and 10pi.
    
    [1] "MULTINEST: an efficient and robust Bayesian inference
    tool for cosmology and particle physics",
    Mon. Not. R. Astron. Soc. 000, 1-14 (2008), F. Feroz,
    M.P. Hobson and M. Bridges.
    """
    def __init__(self):
        self._dimension = 2
        self._mean = 15.707962805119111
        self._sd = 9.205858892151795

    def __call__(self, x):
        if x[0] < 0:
          return -float('inf')
        if x[0] > (10 * np.pi):
          return -float('inf')
        if x[1] < 0:
          return -float('inf')
        if x[1] > (10 * np.pi):
          return -float('inf')

        return 5 * (2 + np.cos(x[0] / 2.0) * np.cos(x[1] / 2.0))

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._dimension

