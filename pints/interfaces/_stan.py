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
import pystan
import os
import pickle

from . import InterfaceLogPDF


class StanLogPDF(InterfaceLogPDF):
    def __init__(self, stan_code, stan_data, pickle_filename=None):

        if pickle_filename:
            if os.path.isfile(pickle_filename):
                sm = pickle.load(open(pickle_filename, 'rb'))
            else:
                sm = pystan.StanModel(model_code=stan_code)
                pickle.dump(sm, open(pickle_filename, 'wb'))
        else:
            sm = pystan.StanModel(model_code=stan_code)

        stanfit = sm.sampling(data=stan_data, iter=1, chains=1,
                              verbose=False, refresh=10,
                              control={'adapt_engaged': False})
        print("Stan model compiled and runs ok...ignore various warnings.")
        self._compiled_stan = sm
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

    def __call__(self, x):
        try:
            return self._log_prob(x, adjust_transform=True)
        # if Pints proposes a value outside of Stan's parameter bounds
        except RuntimeError:
            return -np.inf

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
        val = self._log_prob(x, adjust_transform=True)
        dp = self._grad_log_prob(x, adjust_transform=True)
        return val, dp.reshape(-1)

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

    def pickled_form(self):
        """ Returns form of model ready for pickling. """
        outputs = {'model': self._compiled_stan,
                   'fit': self._fit,
                   'StanLogPDF': self}
        return outputs

    def names(self):
        """ Returns names of Stan parameters. """
        return self._long_names

    def n_parameters(self):
        """ See `InterfaceLogPDF.n_parameters`. """
        return self._n_parameters
