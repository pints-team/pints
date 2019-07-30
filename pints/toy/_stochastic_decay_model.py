#
# Logistic toy model.
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

from . import ToyModel


class StochasticDecayModel(pints.ForwardModelS1, ToyModel):

    """
    Stochastic decay model of a single chemical reaction [1].

    .. math::


    Has one parameter: Rate constant :math:`k`.
    The initial concentration :math:`A(0) = n_0` can be set using the
    (optional) named constructor arg ``initial_concentration``

    [1] Erban et al., 2007

    *Extends:* :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.
    """

    def __init__(self, initial_concentration=20):
        super(StochasticDecayModel, self).__init__()
        self._n0 = float(initial_concentration)
        if self._n0 <= 0:
            raise ValueError('Initial concentration must be positive.')

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 1

    def simulate(self, parameter):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        return self._simulate(parameter)

    def _simulate(self, parameter):
        if parameter <= 0:
            raise ValueError('rate constant must be postive')

        A = self._n0
        k = float(parameter)

        t = 0
        mol_conc = []
        time = []

        while A > 0:
            r = np.random.uniform(0, 1)
            tao = (1 / (A * k)) * np.log(1 / r)
            t += tao
            time.append(t)
            A = A - 1
            mol_conc.append(A)


        return mol_conc, time

    def suggested_parameter(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """

        return 0.1

