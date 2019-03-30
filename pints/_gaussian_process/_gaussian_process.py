# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import pints
import scipy

from gaussian_process import (
        GaussianProcessMatrixFree1,
        GaussianProcessMatrixFree2,
        GaussianProcessDenseMatrix1,
        GaussianProcessDenseMatrix2,
        )


class matern_kernel:
    def __init__(self, dim):
        self.set_sigma(1.0)
        self.set_lengthscales(np.ones(dim))
        self._dim = dim

    def set_sigma(self, sigma):
        self._sigma = sigma
        self._sigma2 = sigma**2

    def set_lengthscales(self, lengthscales):
        self._inv_lengthscales = 1.0/np.array(lengthscales)

    def __call__(self, a, b):
        r = np.linalg.norm((b - a) * self._inv_lengthscales)
        return self._sigma2 * (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)

    def gradient_by(self, a, b, i):
        dx2 = ((b - a) * self._inv_lengthscales)**2
        r = np.sqrt(np.sum(dx2))
        exp_term = np.exp(-np.sqrt(3.0) * r)
        if i == self._dim:
            return 2 * self._sigma * (1.0 + np.sqrt(3.0) * r) * exp_term
        else:
            factor = 3 * self._sigma2 * exp_term
            return self._inv_lengthscales[i] * dx2[i] * factor


class grad_kernel:
    def __init__(self, kernel, dim):
        self._kernel = kernel
        self._dim = dim

    def __call__(self, a, b):
        return self._kernel.gradient_by(a, b, self._dim)


class GaussianProcessLogLikelihood(pints.LogPDF):
    """
    """

    def __init__(self, gaussian_process):
        super(GaussianProcessLogLikelihood, self).__init__()
        self._gaussian_process = gaussian_process

    def __call__(self, x):
        self._gaussian_process.set_hyper_parameters(x)
        likelihood = self._gaussian_process.likelihood()
        return likelihood

    def evaluateS1(self, x):
        """
        returns the partial derivatives of the function with respect to the parameters.

        """
        self._gaussian_process.set_hyper_parameters(x)
        # result is [gradient, likelihood]
        result = self._gaussian_process.grad_likelihood()
        return float('nan'), result

    def n_parameters(self):
        """ See :meth:`ErrorMeasure.n_parameters()`. """
        return self._gaussian_process.n_hyper_parameters()


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

    def __init__(self, samples, pdf_values, matrix_free=False, dense_matrix=False, hierarchical_matrix=False):
        super(GaussianProcess, self).__init__()

        # handle 1d array input
        if len(samples.shape) == 1:
            samples = np.reshape(samples, (-1, 1))

        self._n_parameters = samples.shape[1]
        self._use_dense_matrix = not (matrix_free or hierarchical_matrix or
                                      dense_matrix)

        if not self._use_dense_matrix:
            if self._n_parameters > 2:
                raise NotImplementedError(
                    'GaussianProcess with matrix_free or hierarchical_matrix '
                    'currently only supports d <= 2'
                )
            if dense_matrix:
                if self._n_parameters == 1:
                    self._gaussian_process = GaussianProcessDenseMatrix1()
                elif self._n_parameters == 2:
                    self._gaussian_process = GaussianProcessDenseMatrix2()
            elif matrix_free:
                if self._n_parameters == 1:
                    self._gaussian_process = GaussianProcessMatrixFree1()
                elif self._n_parameters == 2:
                    self._gaussian_process = GaussianProcessMatrixFree2()

            self._gaussian_process.set_data(samples, pdf_values)

        self._samples = samples
        self._values = pdf_values
        self._kernel = matern_kernel(self._n_parameters)
        self._uninitialised = True
        self._lambda = 1e-5

    def _create_matrix(self, kernel, diagonal):
        n = self._values.size
        matrix = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = kernel(self._samples[i, :], self._samples[j, :])

        diag = np.diag_indices(n)
        matrix[diag] += diagonal

        return matrix

    def initialise(self):
        self._K = self._create_matrix(self._kernel, self._lambda**2)
        self._gradK = [
            self._create_matrix(grad_kernel(self._kernel, i), np.sqrt(1e-5))
            for i in range(self.n_parameters()+1)
        ]
        n = self._values.size
        self._gradK.append(2*self._lambda*np.identity(n))

        self._cholesky_L = scipy.linalg.cho_factor(self._K)
        self._invKy = scipy.linalg.cho_solve(self._cholesky_L, self._values)

        self._uninitialised = False

    def _calc_likelihood(self):

        n = self._values.size
        if n == 0:
            return 0

        if self._uninitialised:
            self.initialise()

        half_logdet_K = np.sum(np.log(np.diag(self._cholesky_L[0])))

        return -half_logdet_K - 0.5*np.dot(self._values, self._invKy)

    def likelihood(self):
        if self._use_dense_matrix:
            return self._calc_likelihood()
        else:
            return self._gaussian_process.likelihood()

    def _calc_grad_likelihood(self):
        n = self._values.size
        if n == 0:
            return np.zeros(self.n_hyper_parameters())

        if self._uninitialised:
            self.initialise()

        trace_term = np.empty(len(self._gradK))
        for i, gradK in enumerate(self._gradK):
            trace_term[i] = np.trace(scipy.linalg.cho_solve(self._cholesky_L, gradK))

        second_term = np.empty(len(self._gradK))
        for i, gradK in enumerate(self._gradK):
            second_term[i] = np.dot(self._invKy, gradK @ self._invKy)

        gradient = -0.5*trace_term + 0.5*second_term

        return gradient

    def grad_likelihood(self):
        if self._use_dense_matrix:
            return self._calc_grad_likelihood()
        else:
            return self._gaussian_process.grad_likelihood()

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[lengthscale0, lengthscale1, ...,
        lengthscaleN, scale, noise]]``, where ``N`` is the number of dimensions
        in the parameter space.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        if self._use_dense_matrix:
            self._kernel.set_lengthscales(x[:self.n_parameters()])
            self._kernel.set_sigma(x[-2])
            self._lambda = x[-1]
        else:
            self._gaussian_process.set_parameters(x)

        self._uninitialised = True

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        if self._use_dense_matrix:
            return self._n_parameters + 2
        else:
            return self._gaussian_process.n_parameters()

    def n_parameters(self):
        """
        Returns the dimension of the space this :class:`LogPDF` is defined
        over.
        """
        return self._n_parameters

    def optimise_hyper_parameters(self, use_approximate_likelihood=False):
        score = GaussianProcessLogLikelihood(self)

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
