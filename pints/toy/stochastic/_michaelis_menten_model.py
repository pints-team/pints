#
# Stochastic michaelis-menten toy model.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from . import MarkovJumpModel


class MichaelisMentenModel(MarkovJumpModel):
    r"""
    Simulates the Michaelis Menten Dynamics using Gillespie.

    This system of reaction involves 4 chemical species with
    inital counts ``initial_molecule_count``, and reactions:

        - X1+X2 -> X3 with rate k1
        - X3 -> X1+X2 with rate k2
        - X3 -> X2+X4 with rate k3

    Parameters
    ----------
    initial_molecule_count : Array of size 3 of integers
        Sets the initial molecule count.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Michaelis-Menten_kinetics
    """
    def __init__(self, initial_molecule_count):
        V = [[-1, -1, 1, 0],
             [1, 1, -1, 0],
             [0, 1, -1, 1]]
        super(MichaelisMentenModel, self).__init__(initial_molecule_count,
                                                   V, self._propensities)

    @staticmethod
    def _propensities(xs, ks):
        return [
            xs[0] * xs[1] * ks[0],
            xs[2] * ks[1],
            xs[2] * ks[2]
        ]

    def n_outputs(self):
        return 4
