#
# Stochastic degradation toy model.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from . import MarkovJumpModel


class DegradationModel(MarkovJumpModel):
    r"""
    Stochastic degradation model of a single chemical reaction starting from
    an initial molecule count :math:`A(0)` and degrading to 0 with a fixed rate
    :math:`k`:

    .. math::
        A \xrightarrow{k} 0

    Extends :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.

    Parameters
    ----------
    initial_molecule_count
        The initial molecule count :math:`A(0)`.
    """
    def __init__(self, initial_molecule_count=20):
        V = [[-1]]
        init_list = [initial_molecule_count]
        super(DegradationModel, self).__init__(init_list,
                                               V, self._propensities)

    @staticmethod
    def _propensities(xs, ks):
        return [
            xs[0] * ks[0],
        ]
