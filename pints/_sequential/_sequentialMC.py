#
# Sequential Monte Carlo
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints


class SMC(pints.SMCSampler):
    """
    Samples from a density using sequential Monte Carlo sampling [1], although
    allows multiple MCMC steps per temperature, if desired.

    Algorithm 3.1.1 using equation (31) for ``w_tilde``.

    [1] "Sequential Monte Carlo Samplers", Del Moral et al. 2006,
    Journal of the Royal Statistical Society. Series B.
    """
    def __init__(self, log_posterior, x0, sigma0=None):
        super(SMC, self).__init__(log_posterior, x0, sigma0)

    def run(self):
        """ See :meth:`SMCSampler`. """
        super(SMC, self).run()
        return self._samples

    def name(self):
        """
        Returns name of sampler
        """
        return "Sequential Monte Carlo"
