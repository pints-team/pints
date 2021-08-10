#
# Potassium current (IK) toy model based on the model by Hodgkin & Huxley (HH).
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
#
import numpy as np
import pints

from . import ToyModel


class HodgkinHuxleyIKModel(pints.ForwardModel, ToyModel):
    r"""
    Toy model based on the potassium current experiments used for Hodgkin and
    Huxley's 1952 model of the action potential of a squid's giant axon [1]_.

    A voltage-step protocol is created and applied to an axon, and the elicited
    potassium current (:math:`I_\text{K}`) is given as model output.

    The model equations are

    .. math::

        \alpha &= p_1 \frac{-V - 75 + p_2}{\exp[(-V - 75 + p_2) / p_3] - 1} \\
        \beta &= p_4 \exp[(-V - 75) / p_5] \\
        \frac{dn}{dt} &= \alpha \cdot (1 - n) - \beta \cdot n \\
        E_\text{K} &= -88 \\
        g_\text{max} &= 36 \\
        I_\text{K} &= g_\text{max} \cdot n^4 \cdot (V - E_\text{K})

    Where :math:`p_1, p_2, ..., p_5` are the parameters varied in this toy
    model.

    During simulation, the membrane potential :math:`V` is varied by holding it
    at -75mV for 90ms, then at a "step potential" for 10ms. The step potentials
    are based on the values used in the original paper, and are -69, -64, -56,
    -49, -43, -37, -24, -12, 1, 13, 25, and 34mV.
    The protocol is applied in the interval :math:`t = [0, 1200]`, so sampling
    outside this interval will not provide new information.

    With the parameter values from :meth:`suggested_parameters`, simulation
    results will match those in [1]_.

    Extends :class:`pints.ForwardModel`, :class:`pints.toy.ToyModel`.

    Parameters
    ----------
    initial_condition : float
        The initial value of the state variable :math:`n`.

    References
    ----------
    .. [1] A quantitative description of membrane currents and its application
           to conduction and excitation in nerve.
           Hodgkin, Huxley (1952d) Journal of Physiology.
           https://doi.org/10.1113/jphysiol.1964.sp007378

    Example usage::

        model = HodgkinHuxleyIKModel()

        p0 = model.suggested_parameters()
        times = model.suggested_times()
        values = model.simulate(p0, times)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(times, values)

    Alternatively, the data can be displayed using the :meth:`fold()` method::

        plt.figure()
        for t, v in model.fold(times, values):
            plt.plot(t, v)
        plt.show()

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
        self._prepare_protocol()

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

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 5

    def _prepare_protocol(self):
        """
        Sets up a voltage step protocol for use with this model.

        The protocol consists of multiple steps, each starting with 90ms at a
        fixed holding potential, followed by 10ms at a varying step potential.
        """
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

        # Create list of times when V changes (not including t=0)
        self._events = np.concatenate((
            self._t_both * (1 + np.arange(self._n_steps)),
            self._t_both * np.arange(self._n_steps) + self._t_hold))
        self._events.sort()

        # List of voltages (not including V(t=0))
        self._voltages = np.repeat(self._v_step, 2)
        self._voltages[1::2] = self._v_hold

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """

        if times[0] < 0:
            raise ValueError('All times must be positive.')
        times = np.asarray(times)

        # Unpack parameters
        p1, p2, p3, p4, p5 = parameters

        # Analytically calculate n, during a fixed-voltage step
        def calculate_n(v, n0, t0, times):
            a = p1 * (-(v + 75) + p2) / (np.exp((-(v + 75) + p2) / p3) - 1)
            b = p4 * np.exp((-v - 75) / p5)
            tau = 1 / (a + b)
            inf = a * tau
            return inf - (inf - n0) * np.exp(-(times - t0) / tau)

        # Output arrays
        ns = np.zeros(times.shape)
        vs = np.zeros(times.shape)

        # Iterate over the step, fill in the output arrays
        v = self._v_hold
        t_last = 0
        n_last = self._n0
        for i, t_next in enumerate(self._events):
            index = (t_last <= times) * (times < t_next)
            vs[index] = v
            ns[index] = calculate_n(v, n_last, t_last, times[index])
            n_last = calculate_n(v, n_last, t_last, t_next)
            t_last = t_next
            v = self._voltages[i]
        index = times >= t_next
        vs[index] = v
        ns[index] = calculate_n(v, n_last, t_last, times[index])
        n_last = calculate_n(v, n_last, t_last, t_next)

        # Calculate and return current
        return self._g_max * ns**4 * (vs - self._E_k)

    def suggested_duration(self):
        """
        Returns the duration of the experimental protocol modeled in this toy
        model.
        """
        return self._duration

    def suggested_parameters(self):
        """
        See :meth:`pints.toy.ToyModel.suggested_parameters()`.

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
        """ See :meth:`pints.toy.ToyModel.suggested_times()`. """
        fs = 4
        return np.arange(self._duration * fs) / fs

