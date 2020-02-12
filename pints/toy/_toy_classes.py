#
# Toy base classes.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints


class ToyLogPDF(pints.LogPDF):
    """
    Abstract base class for toy distributions.

    Extends :class:`pints.LogPDF`.
    """

    def distance(self, samples):
        """
        Calculates a measure of distance from ``samples`` to some
        characteristic of the underlying distribution.
        """
        raise NotImplementedError

    def sample(self, n_samples):
        """
        Generates independent samples from the underlying distribution.
        """
        raise NotImplementedError

    def suggested_bounds(self):
        """
        Returns suggested boundaries for prior.
        """
        raise NotImplementedError


class ToyModel(object):
    """
    Defines an interface for toy problems.

    Note that toy models should extend both ``ToyModel`` and one of the forward
    model classes, e.g. :class:`pints.ForwardModel`.
    """

    def suggested_parameters(self):
        """
        Returns an numpy array of the parameter values that are representative
        of the model. For example, these parameters might reproduce a
        particular result that the model is famous for.
        """
        raise NotImplementedError

    def suggested_times(self):
        """
        Returns an numpy array of time points that is representative of the
        model
        """
        raise NotImplementedError

