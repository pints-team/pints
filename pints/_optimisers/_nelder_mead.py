#
# Nelder-Mead simplex method.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import pints
import warnings


# Flow
#          Start
#            |
#         EVAL xr
#            |       yes
#          expand? ------> EVAL xe --> accept xe or xr
#            |
#            |no
#            |       yes
#         reflect? ------> accept xr
#            |
#            |no
#            |
#    contract in or out?
#    |                 |
#    |in               |out
#    |                 |
# EVAL xi           EVAL xo
#    |                 |
# better? ----+---- better?
#    |        |       |
#    |yes     |no      |yes
#    |        |        |
# accept xi   |     accept x
#             |
#          shrink
#         EVAL n xs
#
# This becomes:
#
# Ask:
#  if no xs set
#    Set xs, ask for f(xs)
#  elif shrink
#    Ask for shrink points
#  elif no xr set
#    Order, set mean, set xr, ask for f(xr)
#  else
#    Ask for f(xp) (could be xi, xo, or xe)
#
# Tell:
#  if no fs set
#    Set fs, return (ask will set xr)
#  if shrink:
#    Shrink, unset xr, return (ask will set xr)
#
#  if no fr set
#    set fr
#  if expand
#    if xp set
#      accept xp or xr, unset xp&xr, return (ask will set xr)
#    else
#      set xp, return (ask will ask f(xp))
#  elif contract
#    if xp set
#      if accept xp
#        unset xp&xr, return (ask will set xr)
#      else
#        set shrink mode (ask will ask shrink points)
#    else
#      set xp, return (ask will ask f(xp))
#  else
#    accept xr, unset xr, return (ask will set xr)


class NelderMead(pints.Optimiser):
    """
    Nelder-Mead downhill simplex method.

    Implementation of the classical algorithm by [1]_, following the
    presentation in Algorithm 8.1 of [2]_.

    This is a deterministic local optimiser. In most update steps it performs
    either 1 evaluation, or 2 sequential evaluations, so that it will not
    typically benefit from parallelisation.

    Generates a "simplex" of ``n + 1`` samples around a given starting point,
    and evaluates their scores. Next, each iteration consists of a sequence of
    operations, typically the worst sample ``y_worst`` is replaced with a new
    point::

        y_new = mu + delta * (mu - y_worst)
        mu = (1 / n) * sum(y), y != y_worst

    where ``delta`` has one of four values, depending on the type of operation:

    - Reflection (``delta = 1``)
    - Expansion (``delta = 2``)
    - Inside contraction (``delta = -0.5``)
    - Outside contraction (``delta = 0.5``)

    Note that the ``delta`` values here are common choices, but not the only
    valid choices.

    A fifth type of iteration called a "shrink" is occasionally performed, in
    which all samples except the best sample ``y_best`` are replaced::

        y_i_new = y_best + ys * (y_i - y_best)

    where ys is a parameter (typically ys = 0.5).

    The initialisation of the initial simplex was copied from [3]_.

    References
    ----------
    .. [1] A simplex method for function minimization
           Nelder, Mead 1965, Computer Journal
           https://doi.org/10.1093/comjnl/7.4.308

    .. [2] Introduction to derivative-free optimization
           Andrew R. Conn, Katya Scheinberg, Luis N. Vicente
           2009, First edition. ISBN 978-0-098716-68-9
           https://doi.org/10.1137/1.9780898718768

    .. [3] SciPy on GitHub
           https://github.com/scipy/scipy/
    """

    def __init__(self, x0, sigma0=None, boundaries=None):
        super(NelderMead, self).__init__(x0, sigma0, boundaries)

        if self._boundaries is not None:
            warnings.warn(
                'Nelder-Mead optimisation does not support boundaries.')

        # Parameters for reflection, expansion, and contraction
        # Other choices can be made, as long as -1 < di < 0 < do < dr < de
        self._di = -0.5   # Inside contraction
        self._do = 0.5    # Outside contraction
        self._dr = 1      # Reflection
        self._de = 2      # Expansion

        # Parameter for shrinking
        # Other choices are ok, as long as 0 < ys < 1
        self._ys = 0.5

        # Points and scores
        self._xs = None
        self._fs = None

        self._xm = None             # Mean of n best
        self._xr = self._fr = None  # Reflected point and score
        self._xp = None             # Proposed expansion/contraction point
        self._shrink = False        # Shrink step required

        # Status
        self._running = False
        self._ready_for_tell = False

    def ask(self):
        """ See: :meth:`pints.Optimiser.ask()`. """

        # Check ask/tell
        if self._ready_for_tell:
            raise Exception('ask() called twice')
        self._ready_for_tell = True

        # Initialise
        if self._xs is None:
            self._running = True

            # Initialise simplex, reticulate splines
            # TODO: Use sigma here?
            n = self._n_parameters
            self._xs = np.zeros((n + 1, n))
            self._xs[0] = self._x0
            x_grow = 1.05       # From scipy implementation
            x_zero = 0.00025    # From scipy implementation
            for i in range(n):
                self._xs[1 + i] = self._x0
                if self._xs[1 + i][i] == 0:
                    self._xs[1 + i][i] = x_zero
                else:
                    self._xs[1 + i][i] *= x_grow

            # Ask for initial points
            return np.array(self._xs, copy=True)

        # Shrink operation
        if self._shrink:
            for i in range(self._n_parameters):
                self._xs[1 + i] = \
                    self._xs[0] + self._ys * (self._xs[1 + i] - self._xs[0])

            return np.array(self._xs[1:], copy=True)

        # Start of normal iteration, ask for reflection point
        if self._xr is None:

            # Order, and calculate mean of all except worst
            ifs = np.argsort(self._fs)
            self._xs = self._xs[ifs]
            self._fs = self._fs[ifs]
            self._xm = np.mean(self._xs[:-1], axis=0)

            # Calculate reflection point and return
            self._xr = self._xm + self._dr * (self._xm - self._xs[-1])

            # Ask for f(xr)
            return np.array([self._xr])

        # Extended iteration: ask for expansion or contraction point
        return np.array([self._xp])

    def f_best(self):
        """ See: :meth:`pints.Optimiser.f_best()`. """
        return self._fs[0] if self._running else np.inf

    def name(self):
        """ See: :meth:`pints.Optimiser.name()`. """
        return 'Nelder-Mead'

    def running(self):
        """ See: :meth:`pints.Optimiser.running()`. """
        return self._running

    def stop(self):
        """ See: :meth:`pints.Optimiser.stop()`. """
        return False

    def tell(self, fx):
        """ See: :meth:`pints.Optimiser.tell()`. """

        # Check ask/tell sequence
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        # Initialise
        if self._fs is None:
            fx = np.array(fx, copy=True)
            if np.prod(fx.shape) != self._n_parameters + 1:
                raise ValueError(
                    'Expecting a vector of length (1 + n_parameters).')
            self._fs = fx.reshape((1 + self._n_parameters, ))

            # Return: ask will set xr, get f(xr)
            return

        # Shrink
        if self._shrink:
            fx = np.array(fx, copy=False)
            if np.prod(fx.shape) != self._n_parameters:
                raise ValueError(
                    'Expecting a vector of length n_parameters.')
            self._fs[1:] = fx

            # Reset and return: ask will set xr, get f(xr)
            self._shrink = False
            self._xp = self._xr = self._fr = None
            return

        # Reflection point or contraction/expansion point returned
        if len(fx) != 1:
            raise ValueError('Expecting only a single evaluation')
        fx = fx[0]

        # Set reflection point
        if self._fr is None:
            self._fr = fx

        # Determine operation
        expand = self._fr < self._fs[0]
        contract = self._fr >= self._fs[-2]

        # Option 1: Reflection
        if not (expand or contract):
            self._xs[-1] = self._xr
            self._fs[-1] = self._fr

            # Reset and return: ask will set xr, get f(xr)
            self._xr = self._fr = None
            return

        # Option 2: Expansion
        if expand:
            # Propose expansion
            if self._xp is None:
                self._xp = self._xm + self._de * (self._xm - self._xs[-1])
                return

            # Accept or reject expansion
            if fx <= self._fr:
                self._xs[-1] = self._xp
                self._fs[-1] = fx
            else:
                self._xs[-1] = self._xr
                self._fs[-1] = self._fr

            # Reset and return: ask will set xr, get f(xr)
            self._xp = self._xr = self._fr = None
            return

        # Option 3/4: Outside/Inside contraction
        outside = self._fr < self._fs[-1]

        # Propose contraction
        if self._xp is None:
            dc = self._do if outside else self._di
            self._xp = self._xm + dc * (self._xm - self._xs[-1])
            return

        # Accept contraction, or shrink
        if ((fx <= self._fr) if outside else (fx < self._fs[-1])):
            self._xs[-1] = self._xp
            self._fs[-1] = fx

            # Reset and return: ask will set xr, get f(xr)
            self._xp = self._xr = self._fr = None
            return

        # Option 5: Shrink
        self._shrink = True

    def x_best(self):
        """ See: :meth:`pints.Optimiser.x_best()`. """
        return pints.vector(self._xs[0] if self._running else self._x0)

