#
# Root of the toy module.
# Provides a number of toy models and logpdfs for tests of Pints' functions.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from ._toy_classes import ToyLogPDF, ToyModel, ToyODEModel

from ._annulus import AnnulusLogPDF
from ._beeler_reuter_model import ActionPotentialModel
from ._cone import ConeLogPDF
from ._constant_model import ConstantModel
from ._eight_schools import EightSchoolsLogPDF
from ._fitzhugh_nagumo_model import FitzhughNagumoModel
from ._gaussian import GaussianLogPDF
from ._german_credit import GermanCreditLogPDF
from ._german_credit_hierarchical import GermanCreditHierarchicalLogPDF
from ._goodwin_oscillator_model import GoodwinOscillatorModel
from ._hes1_michaelis_menten import Hes1Model
from ._hh_ik_model import HodgkinHuxleyIKModel
from ._high_dimensional_gaussian import HighDimensionalGaussianLogPDF
from ._logistic_model import LogisticModel
from ._lotka_volterra_model import LotkaVolterraModel
from ._multimodal_gaussian import MultimodalGaussianLogPDF
from ._neals_funnel import NealsFunnelLogPDF
from ._parabola import ParabolicError
from ._repressilator_model import RepressilatorModel
from ._rosenbrock import RosenbrockError, RosenbrockLogPDF
from ._sho_model import SimpleHarmonicOscillatorModel
from ._simple_egg_box import SimpleEggBoxLogPDF
from ._sir_model import SIRModel
from ._twisted_gaussian_banana import TwistedGaussianLogPDF
