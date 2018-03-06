#!/usr/bin/env python
#
# Logistic model.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import pints
from scipy.integrate import odeint


class HodgkinHuxleyIKModel(pints.ForwardModel):
    """
    Toy model based on the potassium current experiments used for Hodgkin and
    Huxley's 1952 model of the action potential of a squid's giant axon.

    A voltage-step protocol is created and applied to an axon, and the elicited
    potassium current is given as model output.

    The protocol is applied in the interval ``t = [0, 1200]``, so sampling
    outside this interval will not provide much new information.

    Example usage::

        model = HodgkinHuxleyIKModel()

        p0 = model.suggested_parameters()
        times = model.suggested_times()
        values = model.simulate(p0, times)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(times, values)

    Alternatively, the data can be displayed using the :meth:`fold` method::

        plt.figure()
        for t, v in model.fold(times, values):
            plt.plot(t, v)
        plt.show()

    References:

    [1] A quantitative description of membrane currents and its application to
    conduction and excitation in nerve
    Hodgkin, Huxley (1952d) Journal of Physiology
    """
    def __init__(self, initial_condition=0.3):
        super(HodgkinHuxleyIKModel, self).__init__()

        # Initial conditions
        self._n0 = float(initial_condition)
        if self._n0 <= 0 or self._n0 >= 1:
            raise ValueError('Initial condition must be > 0 and < 1.')

        # Reversal potential, in mV
        self._E_k = -88

        # Maximum conductance, in mS/cm^2
        self._g_max = 36

        # Voltage step protocol
        self._t_hold = 90         # 90ms at v_hold
        self._t_step = 10         # 10ms at v_step
        self._t_both = self._t_hold + self._t_step
        self._v_hold = -(0 + 75)
        self._v_step = np.array([
            -(-6 + 75),
            -(-11 + 75),
            -(-19 + 75),
            -(-26 + 75),
            -(-32 + 75),
            -(-38 + 75),
            -(-51 + 75),
            -(-63 + 75),
            -(-76 + 75),
            -(-88 + 75),
            -(-100 + 75),
            -(-109 + 75),
        ])
        self._n_steps = len(self._v_step)

        # Protocol duration
        self._duration = len(self._v_step) * (self._t_hold + self._t_step)

    def _protocol(self, time):
        """
        Returns the voltage at the given time, according to the embedded
        voltage step protocol.
        """
        i = int(time / self._t_both)
        if i < 0 or i >= self._n_steps:
            return self._v_hold
        if time - i * self._t_both >= self._t_hold:
            return self._v_step[i]
        return self._v_hold

    def dimension(self):
        """ See :meth:`pints.ForwardModel.dimension()`. """
        return 5

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """

        # Unpack parameters
        p1, p2, p3, p4, p5 = parameters

        # Derivative
        def dndt(n, t):
            v = self._protocol(t)
            a = p1 * (-(v + 75) + p2) / (np.exp((-(v + 75) + p2) / p3) - 1)
            b = p4 * np.exp((-v - 75) / p5)
            return a * (1 - n) - b * n

        # Integrate
        ns = odeint(dndt, self._n0, times).reshape(times.shape)

        # Voltage over time
        voltage = np.array([self._protocol(t) for t in times])

        # Calculate and return current
        return self._g_max * ns**4 * (voltage - self._E_k)

    def fold(self, times, values):
        """
        Takes a set of times and values as return by this model, and "folds"
        the individual currents over each other, to create a very common plot
        in electrophysiology.

        Returns a list of tuples ``(times, values)`` for each different voltage
        step.
        """
        # Get modulus of times
        times = np.mod(times, self._t_both)
        # Remove all points during t_hold
        selection = times >= self._t_hold
        times = times[selection]
        values = values[selection]
        # Use the start of the step as t=0
        times -= self._t_hold
        # Find points to split arrays
        split = 1 + np.argwhere(times[1:] < times[:-1])
        split = split.reshape((len(split),))
        # Split arrays
        traces = []
        i = 0
        for j in split:
            traces.append((times[i:j], values[i:j]))
            i = j
        traces.append((times[i:], values[i:]))
        return traces

    def suggested_duration(self):
        """
        Returns the duration of the experimental protocol modeled in this toy
        model.
        """
        return self._duration

    def suggested_parameters(self):
        """
        Returns an array with the original model parameters used by Hodgkin
        and Huxley.
        """
        p1 = 0.01
        p2 = 10
        p3 = 10
        p4 = 0.125
        p5 = 80
        return p1, p2, p3, p4, p5

    def suggested_times(self):
        """
        Returns a suggested set of sampling times.
        """
        fs = 10
        return np.arange(self._duration * fs) / fs

