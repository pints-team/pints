#
# Provides latex export
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
# Exporters
from _exporter import PdfExporter, PosterExporter
_exporters = {
    'latex-article' : PdfExporter,
    'latex-poster' : PosterExporter,
    }
def exporters():
    """
    Returns a dict of all exporters available in this module.
    """
    return dict(_exporters)
# Expression writers
from _ewriter import LatexExpressionWriter
_ewriters = {
    'latex' : LatexExpressionWriter,
    }
def ewriters():
    """
    Returns a dict of all expression writers available in this module.
    """
    return dict(_ewriters)
#
# Language keywords
#
# None!
