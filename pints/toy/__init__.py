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

from ._beeler_reuter_model import ActionPotentialModel              # noqa
from ._constant_model import ConstantModel                          # noqa
from ._fitzhugh_nagumo_model import FitzhughNagumoModel             # noqa
from ._goodwin_oscillator_model import GoodwinOscillatorModel       # noqa
from ._hes1_michaelis_menten import Hes1Model                       # noqa
from ._hh_ik_model import HodgkinHuxleyIKModel                      # noqa
from ._high_dimensional_normal import HighDimensionalNormalLogPDF   # noqa
from ._logistic_model import LogisticModel                          # noqa
from ._lotka_volterra_model import LotkaVolterraModel               # noqa
from ._multimodal_normal import MultimodalNormalLogPDF              # noqa
from ._normal import NormalLogPDF                                   # noqa
from ._parabola import ParabolicError                               # noqa
from ._repressilator_model import RepressilatorModel                # noqa
from ._rosenbrock import RosenbrockError, RosenbrockLogPDF          # noqa
from ._simple_egg_box import SimpleEggBoxLogPDF                     # noqa
from ._sir_model import SIRModel                                    # noqa
from ._twisted_gaussian_banana import TwistedGaussianLogPDF         # noqa
from ._cone import ConeLogPDF                                       # noqa
from ._annulus import AnnulusLogPDF                                 # noqa
