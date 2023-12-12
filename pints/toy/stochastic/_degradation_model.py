#
# Stochastic degradation toy model.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from . import MarkovJumpModel

import numpy as np


class DegradationModel(MarkovJumpModel):
    r"""
    Stochastic degradation model of a single chemical reaction starting from
    an initial molecule count :math:`A(0)` and degrading to 0 with a fixed rate
    :math:`k`:

    .. math::
        A \xrightarrow{k} 0

    Extends :class:`pints.MarkovJumpModel`.

    Parameters
    ----------
    initial_molecule_count
        The initial molecule count :math:`A(0)`.
    """
    def __init__(self, initial_molecule_count=20):
        V = [[-1]]
        init_list = [initial_molecule_count]
        super(DegradationModel, self).__init__(
            init_list, V, self._propensities)

    @staticmethod
    def _propensities(xs, ks):
        return [xs[0] * ks[0]]

    def mean(self, parameters, times):
        r"""
        Returns the deterministic mean of infinitely many stochastic
        simulations, which follows :math:`A(0) \exp(-kt)`.
        """
        parameters = np.asarray(parameters)
        if len(parameters) != self.n_parameters():
            raise ValueError('This model should have only 1 parameter.')
        k = parameters[0]

        if k <= 0:
            raise ValueError('Rate constant must be positive.')

        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')

        mean = self._x0 * np.exp(-k * times)
        return mean

    def variance(self, parameters, times):
        r"""
        Returns the deterministic variance of infinitely many stochastic
        simulations, which follows :math:`\exp(-2kt)(-1 + \exp(kt))A(0)`.
        """
        parameters = np.asarray(parameters)
        if len(parameters) != self.n_parameters():
            raise ValueError('This model should have only 1 parameter.')
        k = parameters[0]

        if k <= 0:
            raise ValueError('Rate constant must be positive.')

        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')

        variance = np.exp(-2 * k * times) * (-1 + np.exp(k * times)) * self._x0
        return variance

    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        return np.array([0.1])

    def suggested_times(self):
        """ See "meth:`pints.toy.ToyModel.suggested_times()`."""
        return np.linspace(0, 100, 101)
