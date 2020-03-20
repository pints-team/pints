#
# Interface for Stan models
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
from collections import Counter
import pints

from . import InterfaceLogPDF


class StanLogPDF(InterfaceLogPDF):
    def __init__(self, stanfit, rectangular_boundaries=None):
        if not stanfit.__class__.__name__ == 'StanFit4Model':
            raise ValueError('Stan fit object must be of class StanFit4Model.')
        self._fit = stanfit
        self._log_prob = stanfit.log_prob
        self._grad_log_prob = stanfit.grad_log_prob
        self._u_to_c = stanfit.unconstrain_pars
        names = stanfit.unconstrained_param_names()
        self._n_parameters = len(names)
        self._names, self._index = self._initialise_dict_index(names)
        self._long_names = names
        self._counter = Counter(self._index)
        self._dict = {self._names[i]: [] for i in range(len(self._names))}
        self._boundaries = None
        self._lower = None
        self._upper = None
        if rectangular_boundaries is not None:
            if not isinstance(rectangular_boundaries,
                              pints.RectangularBoundaries):
                raise ValueError('Any boundaries must be of class ' +
                                 '`pints.RectangularBoundaries`.')
            if rectangular_boundaries.n_parameters() != self._n_parameters:
                raise ValueError('Dimensionality of boundaries must equal ' +
                                 'that of Stan fit.')
            self._boundaries = rectangular_boundaries
            self._lower = rectangular_boundaries.lower()
            self._upper = rectangular_boundaries.upper()

    def __call__(self, x):
        lower = self._lower
        upper = self._upper
        if lower is not None:
            for i, xs in enumerate(x):
                if xs < lower[i] or xs > upper[i]:
                    return -np.inf
        dict = self._dict_update(x)
        return self._log_prob(self._u_to_c(dict), adjust_transform=True)

    def _dict_update(self, x):
        """ Updates dictionary object with parameter values. """
        names = self._names
        k = 0
        for i, name in enumerate(names):
            count = self._counter[i]
            if count == 1:
                self._dict[name] = x[k]
                k += 1
            else:
                vals = []
                for j in range(count):
                    vals.append(x[k])
                    k += 1
                self._dict[name] = vals
        return self._dict

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        lower = self._lower
        upper = self._upper
        if lower is not None:
            for i, xs in enumerate(x):
                j = self._index[i]
                if xs < lower[j] or xs > upper[j]:
                    return -np.inf, self._boundaries.sample()
        dict = self._dict_update(x)
        uncons = self._u_to_c(dict)
        return (self._log_prob(uncons, adjust_transform=True),
                self._grad_log_prob(uncons, adjust_transform=True))

    def _initialise_dict_index(self, names):
        """ Initialises dictionary and index of names. """
        names_short = []
        for name in names:
            num = name.find(".")
            if num < 0:
                names_short.append(name)
            else:
                names_short.append(name[:num])
        names_long = list(names_short)
        names_short = list(dict.fromkeys(names_short))
        index = [names_short.index(name) for name in names_long]
        return names_short, index

    def names(self):
        """ Returns names of Stan parameters. """
        return self._long_names

    def n_parameters(self):
        """ See `InterfaceLogPDF.n_parameters`. """
        return self._n_parameters
