#
# Provides Python support
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
from _exporter import PythonExporter
_exporters = {
    'python' : PythonExporter,
    }
def exporters():
    """
    Returns a dict of all exporters available in this module.
    """
    return dict(_exporters)
# Expression writers
from _ewriter import PythonExpressionWriter, NumpyExpressionWriter
_ewriters = {
    'python' : PythonExpressionWriter,
    'numpy' : NumpyExpressionWriter,
    }
def ewriters():
    """
    Returns a dict of all expression writers available in this module.
    """
    return dict(_ewriters)
# Language keywords
keywords = [
    'and',
    'del',
    'from',
    'not',
    'while',
    'as',
    'elif',
    'global',
    'or',
    'with', 'assert', 'else', 'if', 'pass', 'yield', 'break', 'except',
    'import', 'print', 'class', 'exec', 'in', 'raise', 'continue',
    'finally',
    'is',
    'return',
    'def',
    'for',
    'lambda',
    'try',
    ]
