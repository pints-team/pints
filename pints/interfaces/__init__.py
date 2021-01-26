#
# Root of the interfaces module.
# Provides a number of toy models and logpdfs for tests of Pints' functions.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

from ._interface_classes import InterfaceLogPDF                           # noqa

from ._stan import StanLogPDF                                       # noqa
