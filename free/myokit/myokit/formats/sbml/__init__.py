#
# Provides SBML support
#
# This file is part of Myokit
#  Copyright 2011-2016 Michael Clerx, Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Importers
from _importer import SBMLImporter, SBMLError
_importers = {
    'sbml' : SBMLImporter,
    }
def importers():
    """
    Returns a dict of all importers available in this module.
    """
    return dict(_importers)
# Exporters
#
# Expression writers
#
