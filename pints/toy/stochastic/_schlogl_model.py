#
# Stochastic Schlogl toy model.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
from scipy.interpolate import interp1d
import pints

from . import MarkovJumpModel


class SchloglModel(MarkovJumpModel):
    r"""
     Simulates the Schlögl System using Gillespie.
     
     This system of reaction involved 4 chemical species with
     inital counts x_0, and reactions:
        - A + 2X -> 3X, rate k1
        - 3X -> A+2X, rate k2
        - B -> X, rate k3
        - X -> B, rate k4
    """
    def __init__(self, x_0):
        self.a_count = 1e5
        self.b_count = 2e5
        # We are only concered with the change in X concentration
        mat = [[ 1],
               [-1],
               [ 1],
               [-1]]
        super(SchloglModel, self).__init__([x_0], mat, self._propensities)

    def simulate(self, parameters, times, approx=None, approx_tau=None):
        return super(SchloglModel, self).simulate(parameters, times, approx, approx_tau)[:,0]

    r"""
     Calculate the rates of reaction based on molecular availability
     xs is of the form [A, B, X]
    """
    def _propensities(self, xs, ks):
        return [
            self.a_count*xs[0]*(xs[0]-1)*ks[0],
            xs[0]*(xs[0]-1)*(xs[0]-2)*ks[1],
            self.b_count*ks[2],
            xs[0]*ks[3]
        ]

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 4
