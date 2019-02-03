#
# Nested MCMC sampler implementation.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class NestedMCMCSampler(pints.NestedSampler):
    """
    Creates a Nested MCMC sampler as introduced in [1] which uses an MCMC
    kernel to sample from the prior subject to the likelihood constraint.

    *Extends:* :class:`NestedSampler`

    [1] F. Feroz and M. P. Hobson, 2008,
        "Multimodal nested sampling: an efficient and robust alternative
        to Markov chain Monte Carlo methods for astronomical data analysis",
        Mon. Not. R. Astron. Soc. 384 (449-463).
    """
    def __init__(self, log_prior):
        super(NestedMCMCSampler, self).__init__(log_prior)
        self._needs_sensitivities = False
        self._first_ask = True
        self._current_prior = None
        self._current = None
        self._step_size = np.ones(self._dimension)

    def step_size(self):
        """
        Returns step size used in MCMC kernel
        """
        return self._step_size

    def set_step_size(self, step_size):
        """
        Sets step size of MCMC kernel
        """
        if len(step_size[step_size <= 0]) > 0:
            raise ValueError('Step size must be positive')
        if not len(step_size) == self._dimension:
            raise ValueError('Length of step size must match ' +
                             'dimensions of problem')
        self._step_size = step_size

    def ask(self):
        """
        Proposes a new point by sampling from an MCMC kernel
        """
        if self._first_ask:
            self._current = self._m_active[self._min_index, :-1]
        z = np.random.normal(self._current, 1, self._dimension)
        self._proposed = z * self._step_size
        return self._proposed

    def tell(self, fx):
        """
        Accepts or rejects point according to eq. (16) in [1]

        [1] F. Feroz and M. P. Hobson, 2008,
            "Multimodal nested sampling: an efficient and robust alternative
            to Markov chain Monte Carlo methods for astronomical
            data analysis",
            Mon. Not. R. Astron. Soc. 384 (449-463)
        """
        if self._first_ask:
            self._current_prior = (
                self._log_prior(self._current)
            )
            self._first_ask = False
        if np.isnan(fx) or fx < self._running_log_likelihood:
            return None
        else:
            proposed_prior = self._log_prior(self._proposed)
            if proposed_prior > self._current_prior:
                self._first_ask = True
                return self._proposed
            else:
                a = proposed_prior / self._accept_count
                if a > np.random.rand():
                    self._first_ask = True
                    return self._proposed
                else:
                    return None

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 1

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[# active points]``

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_n_active_points(x[0])

    def name(self):
        """ See :meth:`pints.NestedSampler.name()`. """
        return 'Nested MCMC sampler'
