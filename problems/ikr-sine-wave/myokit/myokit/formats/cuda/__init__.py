#
# Provides an export to a CUDA kernel
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
from _exporter import CudaKernelExporter
_exporters = {
    'cuda-kernel' : CudaKernelExporter,
    }
def exporters():
    """
    Returns a list of all exporters available in this module.
    """
    return dict(_exporters)
# Expression writers
from _ewriter import CudaExpressionWriter
_ewriters = {
    'cuda' : CudaExpressionWriter,
    }
def ewriters():
    """
    Returns a list of all expression writers available in this module.
    """
    return dict(_ewriters)
# Keywords
from myokit.formats import ansic
keywords = list(ansic.keywords)
#TODO: Append CUDA keywords
