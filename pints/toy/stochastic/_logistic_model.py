#
# Stochastic logistic toy model.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from . import MarkovJumpModel

import numpy as np


class LogisticModel(MarkovJumpModel):
    r"""
    This model describes the growth of a population of individuals, where the
    birth rate per capita, initially :math:`b_0`, decreases to :math:`0` as the
    population size, :math:`\mathcal{C}(t)`, starting from an initial
    population size, :math:`n_0`, approaches a carrying capacity, :math:`k`.
    This process follows a rate according to [1]_

    .. math::
       A \xrightarrow{b_0(1-\frac{\mathcal{C}(t)}{k})} 2A.

    The model is simulated using the Gillespie stochastic simulation algorithm
    [2]_, [3]_.

    *Extends:* :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.

    Parameters
    ----------
    initial_molecule_count : float
        Sets the initial population size :math:`n_0`.

    References
    ----------
    .. [1] Simpson, M. et al. 2019. Process noise distinguishes between
           indistinguishable population dynamics. bioRxiv.
           https://doi.org/10.1101/533182
    .. [2] Gillespie, D. 1976. A General Method for Numerically Simulating the
           Stochastic Time Evolution of Coupled Chemical Reactions.
           Journal of Computational Physics. 22 (4): 403-434.
           https://doi.org/10.1016/0021-9991(76)90041-3
    .. [3] Erban R. et al. 2007. A practical guide to stochastic simulations
           of reaction-diffusion processes. arXiv.
           https://arxiv.org/abs/0704.1908v2
    """
    def __init__(self, initial_molecule_count=50):
        V = [[1]]
        init_list = [initial_molecule_count]
        super(LogisticModel, self).__init__(init_list,
                                            V, self._propensities)

    def n_parameters(self):
        """
        Default value must be overwritten because the number of parameters
        does not correspond with the number of equations.
        """
        return 2

    @staticmethod
    def _propensities(xs, ks):
        return [
            ks[0] * (1 - xs[0] / ks[1]) * xs[0],
        ]

    def mean(self, parameters, times):
        r"""
        Computes the deterministic mean of infinitely many stochastic
        simulations with times :math:`t` and parameters (:math:`b`, :math:`k`),
        which follows:
        :math:`\frac{kC(0)}{C(0) + (k - C(0)) \exp(-bt)}`.

        Returns an array with the same length as `times`.
        """
        parameters = np.asarray(parameters)
        if len(parameters) != self.n_parameters():
            raise ValueError('This model should have only 2 parameters.')

        b = parameters[0]
        if b <= 0:
            raise ValueError('Rate constant must be positive.')

        k = parameters[1]
        if k <= 0:
            raise ValueError("Carrying capacity must be positive")

        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        c0 = self._x0
        return (c0 * k) / (c0 + np.exp(-b * times) * (k - c0))

    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        return np.array([0.1, 500])

    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`."""
        return np.linspace(0, 100, 101)
