#
# LogPDF that uses Stan models.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import warnings

import httpstan
import numpy as np
import stan

import pints


class StanLogPDF(pints.LogPDF):
    """
    A :class:`pints.LogPDF` based on Stan code and data, which can be used in
    sampling and optimisation.

    This class uses PyStan to interface with Stan, which compiles the Stan
    model code (see [1]_). This can take some time (typically minutes).

    Note that the interface assumes that the parameters are on the
    unconstrained scale (according to Stan's "constraint transforms" [1,2]_).
    So, for example, if a variable is declared to have a lower bound of zero,
    sampling happens on the log-transformed space. The interface takes care of
    Jacobian transformations, so a user only needs to transform the variable
    back to the constrained space (in the example, using an ``exp`` transform)
    to obtain appropriate samples.

    Parameters are ordered as in the stan model. Vector and matrix parameters
    are "flattened" into a sequence by Stan, use :meth:`StanLogPDF.names()` to
    see the result.

    Extends :class:`pints.LogPDF`.

    Parameters
    ----------
    stan_code
        Stan code describing the model.
    stan_data
        Data in Python dictionary format as required by PyStan.

    References
    ----------
    .. [1] "Stan: a probabilistic programming language".
           B Carpenter et al., (2017), Journal of Statistical Software
    .. [2] https://github.com/stan-dev/pystan/
    """

    def __init__(self, stan_code, stan_data):

        # Store data
        self.update_data(stan_data)

        # Build stan model
        posterior = stan.build(stan_code, data=stan_data)

        # Use httpstan to get access to the compiled module
        module = httpstan.models.import_services_extension_module(
            posterior.model_name)

        self._log_prob = module.log_prob
        self._log_prob_grad = module.log_prob_grad
        self._names = module.get_param_names(self._data)
        self._n_parameters = len(self._names)

    def __call__(self, x):
        try:
            return self._log_prob(self._data, x, adjust_transform=True)
        # if Pints proposes a value outside of Stan's parameter bounds
        except (RuntimeError, ValueError) as e:
            warnings.warn(
                'Error encountered when evaluating Stan LogPDF: ' + str(e))
            return -np.inf

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        try:
            val = self._log_prob(self._data, x, adjust_transform=True)
            dp = self._log_prob_grad(self._data, x, adjust_transform=True)
            return val, dp
        except (RuntimeError, ValueError) as e:
            warnings.warn(
                'Error encountered when evaluating Stan LogPDF: ' + str(e))
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
        # TODO: Clone data so that it can't be changed anymore?
        self._data = stan_data

