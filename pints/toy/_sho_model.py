#
# simple harmonic oscillator toy model.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import pints

from . import ToyModel


class SimpleHarmonicOscillatorModel(pints.ForwardModelS1, ToyModel):

    r"""
    Simple harmonic oscillator model for a particle that experiences a force
    in proportion to its displacement from an equilibrium position,
    and, in addition, a friction force. The system's behaviour is determined by
    a second order ordinary differential equation (from Newton's second law):

    .. math::
        \frac{d^2y}{dt^2} = -y(t) - \theta \frac{dy(t)}{dt}

    Here it has been assumed that the particle has unit mass and that the
    restoring force has constant of proportionality equal to 1.

    The model has three parameters: the initial position of the particle,
    ``y(0)``, its initial momentum, ``dy/dt(0)`` and the magnitude of the
    friction force, ``theta``.

    Extends :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Simple_harmonic_motion
    """
    def __init__(self):
        super(SimpleHarmonicOscillatorModel, self).__init__()

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 3

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        return self._simulate(parameters, times, False)

    def simulateS1(self, parameters, times):
        """ See :meth:`pints.ForwardModelS1.simulateS1()`. """
        return self._simulate(parameters, times, True)

    def _simulate(self, parameters, times, sensitivities):
        y1_0, y2_0, theta = [float(x) for x in parameters]
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')

        if theta != 2:  # i.e. not critically damped
            sqrt = np.sqrt(theta**2 - 4 + 0j)
            exp1 = np.exp(0.5 * times * (-theta - sqrt))
            exp2 = np.exp(0.5 * times * (-theta + sqrt))

            values = 1 / (2 * sqrt) * (
                2 * y2_0 * (exp2 - exp1) + y1_0 * theta * (exp2 - exp1) +
                y1_0 * sqrt * (exp1 + exp2)
            )
            values = np.real(values)

            if sensitivities:
                exp3 = np.exp(-0.5 * theta * times)
                exp4 = np.exp(times * sqrt)
                sqrt1 = 0.5 * times * sqrt
                sinh_sqrt1 = np.sinh(sqrt1)
                first = 1 / (2 * sqrt**3) * (
                    np.exp(-0.5 * times * (theta + sqrt)))
                second = -4 * times * y2_0 + 2 * y1_0 * (
                    2 + times * sqrt + exp4 * (
                        times * sqrt - 2
                    )
                )
                third = y2_0 * (theta * (2 + times * (theta + sqrt)) + exp4 * (
                    -2 * theta + times * (4 + theta * (-theta + sqrt)))
                )
                dvalues_dp = np.empty((len(times), len(parameters)),
                                      dtype=complex)
                dvalues_dp[:, 0] = exp3 * (
                    (np.cosh(sqrt1) + 1 / sqrt * theta * sinh_sqrt1))
                dvalues_dp[:, 1] = 2 * exp3 * sinh_sqrt1 * 1 / sqrt
                dvalues_dp[:, 2] = first * (second + third)
                dvalues_dp = np.real(dvalues_dp)
        else:
            exp5 = np.exp(-times)
            values = exp5 * (y1_0 + times * (y1_0 + y2_0))
            if sensitivities:
                dvalues_dp = np.empty((len(times), len(parameters)),
                                      dtype=complex)
                dvalues_dp[:, 0] = exp5 * (1 + times)
                dvalues_dp[:, 1] = exp5 * times
                dvalues_dp[:, 2] = np.zeros(len(times))
        if sensitivities:
            return values, dvalues_dp
        else:
            return values

    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        return np.array([1, 0, 0.15])

    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`. """
        return np.linspace(0, 50, 100)
