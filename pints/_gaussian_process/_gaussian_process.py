# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import pints

from gaussian_process import GaussianProcess2, GaussianProcess1


class GaussianProcessLogLikelihood(pints.LogPDF):
    """
    """

    def __init__(self, gaussian_process, use_approximate_likelihood=False):
        super(GaussianProcessLogLikelihood, self).__init__()
        self._gaussian_process = gaussian_process
        self._use_approximate_likelihood = use_approximate_likelihood

    def __call__(self, x):
        self._gaussian_process.set_parameters(x)
        if self_use_approximate_likelihood:
            likelihood = self._gaussian_process.likelihood()
        else:
            likelihood = self._gaussian_process.likelihood_exact()
        return likelihood

    def evaluateS1(self, x):
        """
        returns the partial derivatives of the function with respect to the parameters.

        """
        self._gaussian_process.set_parameters(x)
        # result is [gradient, likelihood]
        if self._use_approximate_likelihood:
            result = self._gaussian_process.likelihoodS1()
        else:
            result = self._gaussian_process.likelihoodS1_exact()
        return float('nan'), result[:len(x)]

    def n_parameters(self):
        """ See :meth:`ErrorMeasure.n_parameters()`. """
        return self._gaussian_process.n_parameters()


class GaussianProcess(pints.LogPDF, pints.TunableMethod):
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

        # handle 1d array input
        if len(samples.shape) == 1:
            samples = np.reshape(samples, (-1, 1))

        self._n_parameters = samples.shape[1]

        if self._n_parameters == 1:
            self._gaussian_process = GaussianProcess1()
        elif self._n_parameters == 2:
            self._gaussian_process = GaussianProcess2()
        else:
            raise NotImplementedError(
                'GaussianProcess currently only supports d <= 2'
            )

        pdf_values = pdf_values
        self._gaussian_process.set_data(samples, pdf_values)

        self._samples = samples
        self._values = pdf_values

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[lengthscale0, lengthscale1, ...,
        lengthscaleN, scale, noise]]``, where ``N`` is the number of dimensions
        in the parameter space.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self._gaussian_process.set_parameters(x)

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return self._gaussian_process.n_parameters()

    def n_parameters(self):
        """
        Returns the dimension of the space this :class:`LogPDF` is defined
        over.
        """
        return self._n_parameters

    def optimise_hyper_parameters(self, use_approximate_likelihood=False):
        score = GaussianProcessLogLikelihood(self._gaussian_process,
                                             use_approximate_likelihood)

        sample_range = np.ptp(self._samples, axis=0)
        value_range = np.ptp(self._values)
        hyper_min = np.zeros(self.n_hyper_parameters())
        hyper_max = np.concatenate((sample_range, [value_range], [value_range]))
        boundaries = pints.RectangularBoundaries(hyper_min, hyper_max)

        x0 = 0.5*(hyper_min + hyper_max)
        print(x0)
        sigma0 = 0.9*(hyper_max - hyper_min)

        opt = pints.OptimisationController(
            score,
            x0,
            sigma0,
            boundaries,
            # method=pints.PSO
            method=pints.AdaptiveMomentEstimation
        )
        opt.optimiser().set_ignore_fbest()

        found_parameters, found_value = opt.run()
        print('found parameters ', found_parameters)
        print(hyper_max)
        print(hyper_min)
        self.set_hyper_parameters(found_parameters)

    def __call__(self, x):
        return self._gaussian_process.predict(x)

    def predict(self, x):
        mean, variance = self._gaussian_process.predict_var(x)
        return mean, variance

    def n_parameters(self):
        """
        Returns the dimension of the space this :class:`LogPDF` is defined
        over.
        """
        return self._n_parameters
