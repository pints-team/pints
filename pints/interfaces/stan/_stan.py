#
# LogPDF that uses Stan models.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np
import pystan
import pints
import warnings


class StanLogPDF(pints.LogPDF):
    def __init__(self, stan_code, stan_data=None):
        """
        Creates a :class:`pints.LogPDF` object from Stan code and data, which
        can then be used in sampling, optimisation etc.

        This class uses Pystan to interface with Stan, which compiles the Stan
        model code (see [1]_). This can take some time (typically minutes).

        Note that the interface assumes that the parameters are on the
        unconstrained scale (according to Stan's "constraint transforms" [1]_).
        So, for example, if a variable is declared to have a lower bound of
        zero, sampling happens on the log-transformed space. The interface
        takes care of Jacobian transformations, so a user only needs to
        transform the variable back to the constrained space (in the example,
        using a ``exp`` transform) to obtain appropriate samples.

        Parameters are ordered as in the stan model. Vector and matrix
        parameters are "flattened" into a sequence by Stan, use
        :meth:`StanLogPDF.names()` to see the result.

        Extends :class:`pints.LogPDF`.

        Parameters
        ----------
        stan_code
            Stan code describing the model.
        stan_data
            Data in Python dictionary format as required by PyStan. Defaults to
            None in which case ``update_data`` must be called to create a valid
            Stan model fit object before calling.

        References
        ----------
        .. [1] "Stan: a probabilistic programming language".
               B Carpenter et al., (2017), Journal of Statistical Software
        """
        self._fit = None
        self._log_prob = None
        self._grad_log_prob = None
        self._n_parameters = None
        self._names = None

        # compile stan
        self._compiled_stan = pystan.StanModel(model_code=stan_code)

        # only create stanfit if data supplied
        if stan_data is not None:
            self.update_data(stan_data)

    def __call__(self, x):
        if self._fit is None:
            raise RuntimeError(
                'No data supplied to create Stan model fit object. '
                'Run `update_data` first.')
        vals = x
        try:
            return self._log_prob(vals, adjust_transform=True)
        # if Pints proposes a value outside of Stan's parameter bounds
        except (RuntimeError, ValueError) as e:
            warnings.warn('RuntimeError or ValueError encountered when '
                          'calling `pints.LogPDF`: ' + str(e))
            return -np.inf

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        if self._fit is None:
            raise RuntimeError(
                'No data supplied to create Stan model fit object. '
                'Run `update_data` first.')
        try:
            val = self._log_prob(x, adjust_transform=True)
            dp = self._grad_log_prob(x, adjust_transform=True)
            return val, dp.reshape(-1)
        except (RuntimeError, ValueError) as e:
            warnings.warn('RuntimeError or ValueError encountered when '
                          'calling `pints.LogPDF`: ' + str(e))
            return -np.inf, np.ones(self._n_parameters)

    def names(self):
        """ Returns names of Stan parameters. """
        return self._names

    def n_parameters(self):
        """ See `pints.LogPDF.n_parameters`. """
        return self._n_parameters

    def update_data(self, stan_data):
        """
        Updates data passed to the underlying Stan model.

        Parameters
        ----------
        stan_data
            Data in Python dictionary format as required by PyStan.
        """
        self._fit = self._compiled_stan.sampling(
            data=stan_data,
            iter=1,
            chains=1,
            verbose=False,
            refresh=0,
            control={'adapt_engaged': False})

        self._log_prob = self._fit.log_prob
        self._grad_log_prob = self._fit.grad_log_prob
        self._names = self._fit.unconstrained_param_names()
        self._n_parameters = len(self._names)

