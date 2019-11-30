#
# ABC Rejection method
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
import random


class ABCSMC(pints.ABCSampler):
    """
    ABC-SMC algorithm  Crudely implemented mostly as a POC
    """
    def __init__(self, log_prior, n_samples=100):

        self._log_prior = log_prior
        self._initial_threshold = 5
        self._n_samples = n_samples
        self._xs = None
        self._intermediate = []
        self._samples_so_far = []
        self._t = 0
        self._max_t = 4
        self._ready_for_tell = False

    def name(self):
        """ See :meth:`pints.ABCSampler.name()`. """
        return 'Rejection ABC'

    def ask(self, n_samples):
        """ See :meth:`ABCSampler.ask()`. """
        if self._ready_for_tell:
            raise RuntimeError('ask called before tell.')
        if self._t == 0:
            self._xs = self._log_prior.sample(n_samples)
        else:
            self._xs = np.array(random.sample(self._intermediate, n_samples)) # You can add probability distribution for weights here

            # QUESTION, how do i add this noise (covariance matrix etc)
            # Scipy multivariate normal
            self._xs += np.random.normal(0, 0.02, self._xs.shape)

            # TODO: Verify points still within initial search space
            if not all([p>0 for p in self._xs]):
                print("Generated dodgy point, rerolling")
                return self.ask(n_samples)

        self._ready_for_tell = True
        return self._xs

    def tell(self, fx):
        """ See :meth:`ABCSampler.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('tell called before ask.')
        self._ready_for_tell = False
        if isinstance(fx, list):
            fx = fx[0]
        if fx < self._threshold:
            if self._t == self._max_t:
                return self._xs
            self._samples_so_far.append(self._xs[0])
            if len(self._samples_so_far) == self._n_samples:
                self._intermediate = self._samples_so_far
                print(f"Pass {self._t} at threshold {self._current_threshold()} complete")
                self._samples_so_far = []
                self._t += 1

        return None

    def _current_threshold(self):
        return np.linspace(self._initial_threshold, self._threshold, self._max_t)[self._t]

    def threshold(self):
        """
        Returns threshold error distance that determines if a sample is
        accepted (is error < threshold).
        """
        return self._threshold

    def set_threshold(self, threshold):
        """
        Sets threshold error distance that determines if a sample is accepted]
        (if error < threshold).
        """
        x = float(threshold)
        if x <= 0:
            raise ValueError('Threshold must be positive.')
        self._threshold = threshold

    def set_initial_threshold(self, initial_threshold):
        x = float(initial_threshold)
        if x <= 0:
            raise ValueError('Threshold must be positive')
        self._initial_threshold = initial_threshold

    def set_number_of_iterations(self, num_iterations):
        self._max_t = num_iterations
