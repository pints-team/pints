#
# Provides CellML support
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
from _importer import CellMLImporter, CellMLError
_importers = {
    'cellml' : CellMLImporter,
    }
def importers():
    """
    Returns a dict of all importers available in this module.
    """
    return dict(_importers)
# Exporters
from _exporter import CellMLExporter
_exporters = {
    'cellml' : CellMLExporter,
    }
def exporters():
    """
    Returns a dict of all exporters available in this module.
    """
    return dict(_exporters)
# Expression writers
from _ewriter import CellMLExpressionWriter
_ewriters = {
    'cellml' : CellMLExpressionWriter,
    }
def ewriters():
    """
    Returns a dict of all expression writers available in this module.
    """
    return dict(_ewriters)
#
# Language keywords
#
