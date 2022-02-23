#
# LogPDF that uses Stan models.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import re
import warnings

import httpstan
import numpy as np
import stan

import pints


class StanLogPDF(pints.LogPDF):
    """
    A :class:`pints.LogPDF` based on Stan code and data, which can be used in
    sampling and optimisation.

    This class interfaces with Stan ([1]_) using PyStan ([2]_) and httpspan
    ([3]_), which compiles the Stan model code. This can take some time
    (typically minutes).

    Note that the interface assumes that the parameters are on the
    unconstrained scale (according to Stan's "constraint transforms" [1]_).
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
    code
        Stan code describing the model.
    data
        Data in Python dictionary format as required by PyStan.

    References
    ----------
    .. [1] "Stan: a probabilistic programming language".
           B Carpenter et al., (2017), Journal of Statistical Software
    .. [2] https://github.com/stan-dev/pystan/
    .. [3] https://github.com/stan-dev/httpstan

    """

    def __init__(self, code, data):

        # Get number of parameters (where vector parameters count as 1)
        n_params = StanLogPDF._count_parameters(code)

        # Store data
        # TODO: Deep clone data so that it can't be changed anymore?
        self._data = dict(data)

        # Build stan model
        posterior = stan.build(code, data=data)

        # Use httpstan to get access to the compiled module
        module = httpstan.models.import_services_extension_module(
            posterior.model_name)

        # Get flattened parameter names
        names = posterior.param_names[:n_params]
        dims = posterior.dims[:n_params]
        self._names = []
        for name, dims in zip(names, dims):
            if dims:
                assert len(dims) == 1
                self._names.extend(
                    [name + '_' + str(i) for i in range(dims[0])])
            else:
                self._names.append(name)

        # Get flattened parameter count
        self._n_parameters = len(self._names)

        # Get PDF and PDFS1 methods
        self._log_prob = module.log_prob
        self._log_prob_grad = module.log_prob_grad

    def __call__(self, x):
        try:
            return self._log_prob(self._data, x, adjust_transform=True)
        # if Pints proposes a value outside of Stan's parameter bounds
        except (RuntimeError, ValueError) as e:
            warnings.warn(
                'Error encountered when evaluating Stan LogPDF: ' + str(e))
            return -np.inf

    @staticmethod
    def _count_parameters(code):
        """
        Counts the parameters -- not including transformed parameters -- that
        are declared in a piece of Stan code.

        Notes:
        - Supported comments are // and /* */:
          https://mc-stan.org/docs/2_29/reference-manual/comments.html
        - The { and } characters appear in statements, but never in
          declarations:
          https://mc-stan.org/docs/2_29/reference-manual/variable-declaration.html
        - Each block in Stan must start with declarations, followed by
          statements: https://mc-stan.org/docs/2_29/reference-manual/overview-of-stans-program-blocks.html
        - The parameters block only contains declarations:
          https://mc-stan.org/docs/2_29/reference-manual/program-block-parameters.html
        """ # noqa

        # Strip // and /* */ comments
        pattern = re.compile(r'//.*?$|/\*.*?\*/', re.DOTALL | re.MULTILINE)
        code = pattern.sub('', code)

        # Extract parameters block code
        pattern = re.compile(
            r'parameters\s+\{([^}]*)\}', re.DOTALL | re.MULTILINE)
        block = pattern.search(code)
        code = block.group(1)

        # Count statements
        return len(code.split(';')) - 1

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        try:
            val = self._log_prob(self._data, x, adjust_transform=True)
            dp = self._log_prob_grad(self._data, x, adjust_transform=True)
            return val, np.asarray(dp)
        except (RuntimeError, ValueError) as e:
            warnings.warn(
                'Error encountered when evaluating Stan LogPDF: ' + str(e))
            return -np.inf, np.ones(self._n_parameters)

    def names(self):
        """
        Returns list comprising names of Stan parameters. Note that the length
        of this list will not generally equal ``n_parameters`` since each
        name may correspond to a vector.
        """
        return self._names

    def n_parameters(self):
        """ See `pints.LogPDF.n_parameters`. """
        return self._n_parameters

