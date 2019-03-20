# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import pints

from aboria_wrapper import GaussianProcess2

class GaussianProcessErrorMeasure(pints.ErrorMeasure):
    """
    """

    def __init__(self, gaussian_process):
        super(GaussianProcessErrorMeasure, self).__init__()
        self._gaussian_process = gaussian_process

    def evaluateS1(self, x):
        """
        Evaluates this error measure, and returns the result plus the partial
        derivatives of the result with respect to the parameters.

        The returned data has the shape ``(e, e')`` where ``e`` is a scalar
        value and ``e'`` is a sequence of length ``n_parameters``.

        *This is an optional method that is not always implemented.*
        """
        self._gaussian_process.set_sigma(x[0])
        self._gaussian_process.set_lengthscale(x[1:])
        return -self._gaussian_process.likelihood_gradient()

    def n_parameters(self):
        """ See :meth:`ErrorMeasure.n_parameters()`. """
        return self._gaussian_process.n_parameters()


class GaussianProcess(pints.LogPDF):
    """
    Represents the natural logarithm of a (not necessarily normalised)
    probability density function (PDF), obtained from fitting a gaussian process.

    Arguments:

    ``samples``
        A sequence of samples in parameter space. Is a numpy array with shape
        ``(n, d)`` where ``n`` is the requested number of samples, and ``d`` is
        the number of parameters

    ``pdf_values``
        The value of the PDF at the location of the samples


    *Extends:* :class:`LogPDF`
    """

    def __init__(self, samples, pdf_values):
        super(GaussianProcess, self).__init__()

        # currently only supports dimension 2
        if samples.shape[1] != 2:
            raise NotImplementedError

        self._gaussian_process = GaussianProcess2()
        self._gaussian_process.set_data(samples, pdf_values)

        score = GaussianProcessErrorMeasure(self._gaussian_process)

        sample_range = np.ptp(samples, axis=0)
        value_range = np.ptp(pdf_values)
        hyper_min = np.zeros(self.n_parameters() + 1)
        hyper_max = np.concatinate(([value_range], sample_range))
        boundaries = pints.RectangularBoundaries(hyper_min, hyper_max)

        x0 = 0.5*(hyper_min+hyper_max)
        sigma0 = hyper_min-hyper_max
        found_parameters, found_value = pints.optimise(
            score,
            x0,
            sigma0,
            boundaries,
            method=pints.AdaptiveMomentEstimation
        )


    def __call__(self, x):
        raise NotImplementedError

    def evaluateS1(self, x):
        """
        Evaluates this LogPDF, and returns the result plus the partial
        derivatives of the result with respect to the parameters.

        The returned data is a tuple ``(L, L')`` where ``L`` is a scalar value
        and ``L'`` is a sequence of length ``n_parameters``.

        Note that the derivative returned is of the log-pdf, so
        ``L' = d/dp log(f(p))``, evaluated at ``p=x``.

        *This is an optional method that is not always implemented.*
        """
        raise NotImplementedError

    def n_parameters(self):
        """
        Returns the dimension of the space this :class:`LogPDF` is defined
        over.
        """
        return self._gaussian_process.n_parameters()
