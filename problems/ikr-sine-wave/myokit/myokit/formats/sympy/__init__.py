#
# Provides interaction with the computer algebra system Sympy
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
from _ereader import SymPyExpressionReader, SympyExpressionReader
from _ewriter import SymPyExpressionWriter, SympyExpressionWriter
# Importers
# Exporters
# Expression writers
_ewriters = {
    'sympy' : SymPyExpressionWriter,
    }
def ewriters():
    """
    Returns a dict of all expression writers available in this module.
    """
    return dict(_ewriters)
# Shared expression reader and writer
_sympyreader_ = None
def _sympyreader(model=None):
    """
    Returns a globally shared SymPyExpressionReader, set to resolve variable
    names using the given model.
    """
    global _sympyreader_
    if _sympyreader_ is None:
        _sympyreader_ = SymPyExpressionReader(model)
    else:
        _sympyreader_.set_model(model)
    return _sympyreader_
_sympywriter_ = None
def _sympywriter():
    """
    Returns a globally shared SymPyExpessionWriter.
    """
    global _sympywriter_
    if _sympywriter_ is None:
        _sympywriter_ = SymPyExpressionWriter()
    return _sympywriter_
# Simple write and read methods
def write(e):
    """
    Converts the given Myokit expression to a SymPy expression.
    """
    w = _sympywriter()
    return w.ex(e)
def read(e, model=None):
    """
    Converts the given Sympy expression to a myokit one. Any variable names
    will be resolved against the given model.
    """
    r = _sympyreader(model)
    return r.ex(e)
#
# Language keywords
#
# None!
