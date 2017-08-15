#
# Provides C++ support
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
import myokit.formats.ansic
# Importers
# Exporters
# Expression writers
from _ewriter import CppExpressionWriter
_ewriters = {
    'cpp' : CppExpressionWriter,
    }
def ewriters():
    """
    Returns a dict of all expression writers available in this module.
    """
    return dict(_ewriters)
# Language keywords
keywords = list(myokit.formats.ansic.keywords)
#TODO Append more keywords
