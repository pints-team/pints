#
# Provides Matlab/Octave support
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
# Importers
#
# Exporters
from _exporter import MatlabExporter
_exporters = {
    'matlab' : MatlabExporter,
    }
def exporters():
    """
    Returns a dict of all exporters available in this module.
    """
    return dict(_exporters)
# Expression writers
from _ewriter import MatlabExpressionWriter
_ewriters = {
    'matlab' : MatlabExpressionWriter,
    }
def ewriters():
    """
    Returns a dict of all expression writers available in this module.
    """
    return dict(_ewriters)
# Language keywords
keywords = [
    'i',
    'e',
    'pi'
    ]
