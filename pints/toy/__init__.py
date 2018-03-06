#
# Root of the toy module.
# Provides a number of toy models and logpdfs for tests of Pints' functions.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from ._logistic import LogisticModel                        # noqa
from ._hh_ik import HodgkinHuxleyIKModel                    # noqa
from ._rosenbrock import RosenbrockError, RosenbrockLogPDF  # noqa
from ._multimodal_normal import MultimodalNormalLogPDF      # noqa
from ._high_dimensional_normal import HighDimensionalNormalLogPDF   # noqa
from ._distributions import ( # noqa
    TwistedGaussianLogPDF,
)
