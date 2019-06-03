#
# Constant model with multiple outputs.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import pints


class Approx_Grad_Model(pints.ForwardModelS1):
    """
    This model takes another model as input and can compute the approximated gradient of the input model

    For an `n`-dimensional model, evaluated with parameters
    ``p = [p_1, p_2, ..., p_n]`` we will compute the approximated partial derivative for each
    parameter to form the approximated gradient at each time step t

    .. math::
        f(p_1,..., p_i,..., p_n)/dp_i

    Arguments:

    ``model``
        The model whose gradient we want to approximate
    ``repetition``
        The number different values we will use to approximate each partial derivative.
        The higher the number of repetation the more accurate is the approximation.

    Example::

        model = toy.LogisticModel()
        SGD = Approx_Grad_Model(model, 10)

        real_parameters = [0.015, 500]
        times = np.linspace(0, 1000, 1000)

        y, dy = SGD.simulateS1(real_parameters, times)

    In this example, the returned output is
            ``[[gradient p_1 at t_1, ... gradient p_1 at t_n,], [...], [gradient p_n at t_1, ... gradient p_n at t_n,]]`` .

    *Extends:* :class:`pints.ForwardModelS1`.
    """

    def __init__(self, model, repetition):
        super(Approx_Grad_Model, self).__init__()

        n = model.n_parameters()
        self._n = n
        self._model = model

        # The number of samples we want from the noisy params to estimate the gradient
        self._repet = repetition

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        return self._n

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return self._n

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        return self._model.simulate(parameters, times)

    def simulateS1(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulateS1()`. """

        self._params = np.array([]).reshape(0, self._n)

        # Basically adding a N(0,1) times the values noise
        # We might want to change this

        zs = np.array([np.random.normal(1, 1) for _ in range(self._repet)])
        for i in range(self._n):
            # Modify only for one parameter
            temp = np.full((self._repet, self._n), parameters)
            temp[:, i] *= zs
            self._params = np.vstack([self._params, temp])

        self._ys = np.array([]).reshape(0, len(times))
        for x in self._params:
            self._ys = np.concatenate((self._ys, [self.simulate(x, times)]))

        dy = np.zeros((self._n, len(times)))
        p = self._repet
        for i in range(self._n):
            for j in range(p):
                for k in range(p):
                    if j == k:
                        continue
                    dy[i] += (self._ys[j + i * p] - self._ys[k + i * p]) / (
                                self._params[j + i * p][i] - self._params[k + i * p][i])

        return self._ys, dy
