#
# Interfaces base classes.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints


class InterfaceLogPDF(pints.LogPDF):
    """
    Abstract base class for interface distributions.

    Extends :class:`pints.LogPDF`.
    """
    def evaluateS1(self, n_samples):
        """ See `pints.LogPDF.evaluateS1`. """
        raise NotImplementedError

    def n_parameters(self):
        """ Returns number of model parameters. """
        raise NotImplementedError
