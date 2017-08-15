#
# CellML expression writer
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
from myokit.formats.mathml import MathMLExpressionWriter
class CellMLExpressionWriter(MathMLExpressionWriter):
    """
    Writes equations for variables using CellML's version of MathML.
    
    Differences from normal MathML:
    
     1. Only content MathML is supported
     2. Variable names are always written as unames.
     3. Every number defines an attribute cellml:units
    
    The expression writer requires a single argument ``units``. This should be
    a mapping of Myokit units to string unit names.
    """
    def __init__(self, units):
        super(CellMLExpressionWriter, self).__init__()
        super(CellMLExpressionWriter, self).set_mode(presentation=False)
        self._units = units
    def _ex_number(self, e, t):
        x = self._et.SubElement(t, 'cn')
        x.text = self._fnum(e)
        u = e.unit()
        x.attrib['cellml:units'] = self._units[u] if u else 'dimensionless'
    def _ex_name(self, e, t):
        x = self._et.SubElement(t, 'ci')
        x.text = e.var().uname()
    def set_mode(self, presentation=False):
        """
        This expression writer only supports content MathML, so this method
        does nothing.
        """
        pass
    def set_lhs_function(self, f):
        """
        This expression writer always uses unames, setting an LHS function is
        not supported.
        """
        pass
