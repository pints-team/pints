#
# Neal's funnel log pdf.
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
import scipy
import scipy.stats


class NealsFunnelLogPDF(pints.LogPDF):
    """
    Toy distribution based on a d-dimensional distribution of the form,

    .. math::

        f(x_1, x_2,...,x_d,\\nu) =
            \\left[\\prod_{i=1}^d\\mathcal{N}(x_i|0,e^{\\nu/2})\\right] \\times
            \\mathcal{N}(\\nu|0,3)

    where ``x`` is a d-dimensional real.

    Arguments:

    ``dimensions``
        The dimensionality of funnel (by default equal to 11) which must
        exceed 1.

    *Extends:* :class:`pints.LogPDF`.
    """
    def __init__(self, dimensions=11):
        if dimensions < 2:
            raise ValueError('Dimensions must exceed 1.')
        self._n_parameters = int(dimensions)

    def __call__(self, x):
        nu = x[-1]
        x_log_pdf = [scipy.stats.norm.logpdf(y, 0, np.exp(nu / 2)) for y in x]
        return np.sum(x_log_pdf) + scipy.stats.norm.logpdf(nu, 0, 3)

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._n_parameters

    def evaluateS1(self, x):
        

    def sample(self, n_samples):
        """ Samples from the underlying distribution. """
        n = self._n_parameters
        samples = np.zeros((n_samples, n))
        for i in range(n_samples):
            nu = np.random.normal(0, 3, 1)[0]
            sd = np.exp(nu / 2)
            x = np.random.normal(0, sd, n - 1)
            samples[i, 0:(n - 1)] = x
            samples[i, n - 1] = nu
        return samples
