#
# Stochastic production and degradation toy model.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from . import MarkovJumpModel

import numpy as np


class ProductionDegradationModel(MarkovJumpModel):
    r"""
    Stochastic production and degradation model of two separate chemical
    reactions reaction starting from an initial molecule count :math:`A(0)`
    and degrading to 0 with a fixed rate.
    :math:`k`:

    .. math::
        A \xrightarrow{k1} 0, 0 \xrightarrow{k2} A

    Extends :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.

    Parameters
    ----------
    initial_molecule_count
        The initial molecule count :math:`A(0)`.
    """
    def __init__(self, initial_molecule_count=20):
        V = [[-1], [1]]
        init_list = [initial_molecule_count]
        super(ProductionDegradationModel, self).__init__(init_list,
                                                         V, self._propensities)

    @staticmethod
    def _propensities(xs, ks):
        return [
            xs[0] * ks[0],
            ks[1]
        ]

    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        return np.array([0.1, 0.2])

    def suggested_times(self):
        """ See "meth:`pints.toy.ToyModel.suggested_times()`."""
        return np.linspace(0, 100, 101)
