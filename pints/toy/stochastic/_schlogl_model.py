#
# Schlogl's stochastic toy model.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from . import MarkovJumpModel

import numpy as np


class SchloglModel(MarkovJumpModel):
    r"""
    Schlogl's system of chemical reactions has a single type of molecules and
    starts with an initial count :math:`A(0)`. The evolution of the molecule
    count is defined through the rates :math:`k_1`, :math:`k_2`, :math:`k_3`
    and :math:`k_4` and the following equations:

    ..math::
        2A \xrightarrow{k_1} 3A
        3A \xrightarrow{k_2} 2A
        0 \xrightarrow{k_3} A
        A \xrightarrow{k_4} 0

    Extends :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.

    Parameters
    ----------
    initial_molecule_count
        The initial molecule count :math:`A(0)`.
    """
    def __init__(self, initial_molecule_count=20):
        V = [[1], [-1], [1], [-1]]
        init_list = [initial_molecule_count]
        super(SchloglModel, self).__init__(init_list,
                                           V, self._propensities)

    @staticmethod
    def _propensities(xs, ks):
        return [
            xs[0] * (xs[0] - 1) * ks[0],
            xs[0] * (xs[0] - 1) * (xs[0] - 2) * ks[1],
            ks[2],
            xs[0] * ks[3]
        ]

    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        return np.array([0.18, 0.00025, 2200, 37.5])

    def suggested_times(self):
        """ See "meth:`pints.toy.ToyModel.suggested_times()`."""
        return np.linspace(0, 100, 101)
