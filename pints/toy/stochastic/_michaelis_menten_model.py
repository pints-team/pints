#
# Stochastic degradation toy model.
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


class MichaelisMentenModel(MarkovJumpModel):
    r"""
     Simulates the Michaelis Menten Dynamics using Gillespie.
     
     This system of reaction involved 4 chemical species with
     inital counts x_0, and reactions:
        - X1+X2 -> X3 with rate k1
        - X3 -> X1+X2 with rate k2
        - X3 -> X2+X4 with rate k3
    """
    def __init__(self, x_0):
        mat =  [[-1, -1,  1,  0],
                [ 1,  1, -1,  0],
                [ 0,  1, -1,  1]]
        super(MichaelisMentenModel, self).__init__(x_0, mat, self._propensities)

    @staticmethod
    def _propensities(xs,ks):
        return [
            xs[0]*xs[1]*ks[0],
            xs[2]*ks[1],
            xs[2]*ks[2]
        ]

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 3
