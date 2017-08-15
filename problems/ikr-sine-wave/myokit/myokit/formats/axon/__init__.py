#
# Provides support for working with data in formats used by Axon Instruments.
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
# Abf classes
from _abf import AbfFile, Sweep, Channel
# Atf classes
from _atf import AtfFile, load_atf, save_atf
# Importers
from _importer import AbfImporter
_importers = {
    'abf' : AbfImporter,
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
