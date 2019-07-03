#
# Slice Sampling with Stepout MCMC Method
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class SliceStepoutMCMC(pints.SingleChainMCMC):
    """
    *Extends:* :class:`SingleChainMCMC`

    Implements Slice Sampling with Stepout, as described in [1].

    Generates samples by sampling uniformly from the volume laying underneath the 
    un-normalised posterior (``f``). It does so by introducing an auxiliary variable (``y``)
    and by definying a Markov chain.
    
    If the distribution is univariate, the Markov chain will have the following steps:

    1) Calculate the pdf (``f(x0)``) of the current sample (``x0``).
    2) Draw a real value (``y``) uniformly from (0, f(x0)), thereby defining a
    horizontal “slice”: S = {x: y < f (x)}. Note that ``x0`` is always within S.
    3) Find an interval (``I = (L, R)``) around ``x0`` that contains all, or much, of the
    slice.
    4) Draw the new point (``x1``) from the part of the slice within this interval.

    If the distribution is multivariate and we need to sample for ``x = (x1,...,xn)``, we
    apply the univariate algorithm to each variable in turn. We hereby define ``x = (x1,...,xn)``
    as the sample, and ``(x1,...,xn)`` as the parameters of the sample.

    The algorithm we implement uses the ``Stepout`` method to define the interval ``I = (L, R)``,
    as described in [1] Fig. 3. pp.715: we define the interval ``I`` by expanding the initial 
    interval by a width ``w`` in each direction until both edges fall outside the slice, or until
    a pre-determined limit is reached.
    
    The presented implementation uses the ``Shrinkage`` procedure to shrink ``I = (L, R)`` 
    after rejecting a trial point, as defined in [1] Fig. 5. pp.716: we initially uniformly 
    sample a trial point from the interval ``I``, and subsequently shrink the interval each 
    time a trial point is rejected.

    Note that to include the a new trial point, we implement 1 check procedure, as described
    in [1] Fig. 5. pp.716:

    1) We check whether y < f(x1). We will refer to this check as the ``Threshold Check``.
    
    To avoid floating-point underflow, we implement the suggestion advanced in [1] pp.712: 
    we use the log pdf of the un-normalised posterior (``g(x) = log(f(x))``) instead of 
    ``f(x)``, and we define the slice as S = {x : z < g(x)}, where:

        z = log(y) = g(x0) − e

    and e is exponentially distributed with mean 1.

    [1] Neal, R.M., 2003. Slice sampling. The annals of statistics, 31(3), pp.705-767.
    """

    def __init__(self, x0, sigma0=None):
        super(SliceStepoutMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._x0 = np.asarray(x0, dtype=float)
        self._running = False
        self._ready_for_tell = False

        # Iterations monitoring
        self._mcmc_iteration = 0 

        # Current sample, log_pdf of the current sample 
        self._current = None
        self._current_log_pdf = None

        # Current log_y used to define the slice
        self._current_log_y = None

        # Current proposed sample
        self._proposed = None

        # Default initial interval width w used in the Stepout procedure 
        # to expand the interval
        self._w = 1

        # Default integer limiting the size of the interval to ``m*w```
        self._m = 50

        # Flag to initialise the expansion of the interval ``I=(L,R)``
        self._first_expansion = False

        # Flag indicating whether the interval expansion is concluded
        self._interval_found = False

        # Number of steps used for expanding the interval ``I=(L,R)``
        self._j = None
        self._k = None

        # Edges of the interval ``I=(L,R)``
        self._l = None
        self._r = None

        # Multi-dimensional points used to calculate the log_pdf of the edges ``l,r```
        self._temp_l = None
        self._temp_r = None

        # Log_pdf of interval edge points ``l,r```
        self._fx_l = None
        self._fx_r = None

        # Flags to indicate the interval edge to update
        self._set_l = False
        self._set_r  = False

        # Index of parameter "xi" we are updating of the sample "x = (x1,...,xn)"
        self._i = 0

        # Stochastic variables as instance variables for testing
        self._u = 0
        self._v = 0
        self._e = 0



    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """

        # Check ask/tell pattern
        if self._ready_for_tell:
            raise RuntimeError('Ask() called when expecting call to tell().')
        
        # Initialise on first call
        if not self._running:
            self._running = True

        # Very first iteration
        if self._current is None:

            # Ask for the log pdf of x0
            self._ready_for_tell = True
            return np.array(self._x0, copy=True)


        # If the flag is True, we initialise the expansion of interval ``I=(l,r)`` 
        if self._first_expansion == True:

            # Set initial values for l and r
            self._u = np.random.uniform()
            self._l = self._proposed[self._i] - self._w*self._u
            self._r = self._l + self._w

            # Set maximum number of steps for expansion to the left (j) and right (k)
            self._v = np.random.uniform()
            self._j = np.floor(self._m*self._v)
            self._k = (self._m-1) - self._j
            
            # Initialise arrays used for calculating the log_pdf of the edges l,r
            self._temp_l = np.array(self._proposed, copy=True)
            self._temp_r = np.array(self._proposed, copy=True)
            self._temp_l[self._i] = self._l
            self._temp_r[self._i] = self._r

            # We have initialised the expansion, so we set the flag to false
            self._first_expansion = False

            # Ask for log pdf of initial interval ``I``` edges
            self._ready_for_tell = True

            return np.array(self._temp_l, copy=True), np.array(self._temp_r, copy=True)
        
        # Expand the interval ``I``` until edges ``l,r`` are outside the slice or we have reached
        # limit of expansion steps
        
        # Check whether we can expand to the left
        if self._j > 0 and self._current_log_y < self._fx_l:
            
            # Set flag to indicate that we are updating the left edge
            self._set_l = True

            # If left edge of the interval is inside the slice, keep expanding
            self._l -= self._w
            self._temp_l[self._i] = self._l
            self._j -= 1 

            # Ask for log pdf of the updated left edge
            self._ready_for_tell = True

            return np.array(self._temp_l, copy=True)

        # Reset flag now that we have finished updating the left edge
        self._set_l = False

        # Check whether we can expand to the right
        if self._k > 0 and self._current_log_y < self._fx_r:

            # Set flag to indicate that we are updating the right edge
            self._set_r = True

            # If right edge of the interval is inside the slice, keep expanding
            self._r += self._w
            self._temp_r[self._i] = self._r
            self._k -= 1

            # Ask for log pdf of the updated right edge
            self._ready_for_tell = True

            return np.array(self._temp_r, copy=True)

        # Reset flag now that we have finished updating the right edge
        self._set_r = False

        # Now that we have expanded the interval, set flag
        self._interval_found = True

        # Sample new trial point by sampling uniformly from the interval ``I=(l,r)``
        self._u = np.random.uniform()
        self._proposed[self._i] = self._l + self._u*(self._r - self._l)

        # Send trial point for checks
        self._ready_for_tell = True
        return np.array(self._proposed, copy=True)


    def tell(self, reply):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """

        # Check ask/tell pattern
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False

        # Unpack reply
        fx = np.asarray(reply, dtype=float)

        # Very first call
        if self._current is None:

            # Check first point is somewhere sensible
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Set current sample, log pdf of current sample and initialise proposed 
            # sample for next iteration
            self._current = np.array(self._x0, copy=True)
            self._current_log_pdf = fx
            self._proposed = np.array(self._current, copy=True)

            # Sample height of the slice log_y for next iteration
            self._e = np.random.exponential(1)
            self._current_log_y = self._current_log_pdf - self._e

            # Increase number of samples found
            self._mcmc_iteration += 1

            # Set flag to true as we need to initialise the interval expansion for
            # next iteration
            self._first_expansion = True

            # Return first point in chain, which is x0
            return np.array(self._current, copy=True)

        # While we expand the interval ``I=(l,r)``, we return None
        if self._interval_found == False:
            
            # Set the log_pdf of the interval edge that we are expanding
            if self._set_l == True:
                self._fx_l = fx
            elif self._set_r == True:
                self._fx_r = fx
            else:
                self._fx_l = fx[0]
                self._fx_r = fx[1]

            return None

        # Do ``Threshold Check`` to check if the proposed point is within the slice
        if self._current_log_y < fx:

            # Reset flag to true as we need to initialise the interval for the update of the
            # next parameter 
            self._first_expansion = True

            # Reset flag to false since we still need to estimate the interval for the next parameter
            self._interval_found = False


            # If we have updated all the parameters of the sample, start constructing next sample
            if self._i == len(self._proposed)-1:

                # Reset index to 0, so that we start from updating parameter x0 of the next sample
                self._i = 0

                # The accepted sample becomes the new current sample
                self._current = np.array(self._proposed, copy=True)

                # The log_pdf of the accepted sample is used to construct the new slice
                self._current_log_pdf = fx

                # Update number of mcmc iterations
                self._mcmc_iteration += 1

                # Sample new log_y used to define the next slice
                self._e = np.random.exponential(1)
                self._current_log_y = self._current_log_pdf - self._e

                # Return the accepted sample
                return np.array(self._proposed, copy=True)

            # If there are still parameters to update to generate the sample, move to next parameter
            else:
                self._i += 1
                return None

        # If the trail point is rejected in the ``Threshold Check``, shrink the interval
        if self._proposed[self._i] < self._current[self._i]:
            self._l = self._proposed[self._i]
            self._temp_l[self._i] = self._l
        else:
            self._r = self._proposed[self._i]
            self._temp_r[self._i] = self._r

        return None        


    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Slice Sampling'

    def set_w(self, w):
        """
        Sets width w for generating the interval
        """
        w = float(w)
        if w <= 0:
            raise ValueError('Width w must be positive for expanding the interval.')
        self._w = w

    def set_m(self, m):
        """
        Set integer m for limiting interval expansion
        """
        m = int(m)
        if m <= 0:
            raise ValueError('Integer m must be positive to limit the interval size to "m*w".')
        self._m = m

    def w(self):
        """
        Returns width w used for generating the interval
        """
        return self._w

    def m(self):
        """
        Returns integer m used for limiting interval expansion
        """
        return self._m  





