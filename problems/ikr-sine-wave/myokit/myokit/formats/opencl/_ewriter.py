#
# OpenCL expression writer
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
import myokit
from myokit.formats.python import PythonExpressionWriter
class OpenCLExpressionWriter(PythonExpressionWriter):
    """
    This :class:`ExpressionWriter <myokit.formats.ExpressionWriter>` translates
    Myokit :class:`expressions <myokit.Expression>` to OpenCL syntax.
    """
    def __init__(self, precision=myokit.SINGLE_PRECISION, native_math=True):
        super(OpenCLExpressionWriter, self).__init__()
        self.function_prefix = ''
        self.sp = (precision == myokit.SINGLE_PRECISION)
        self.nm = (native_math == True)
    #def _ex_name(self, e):
    #def _ex_derivative(self, e):
    def _ex_number(self, e):
        return myokit.strfloat(e) + 'f' if self.sp else myokit.strfloat(e)
    #def _ex_prefix_plus(self, e):
    #def _ex_prefix_minus(self, e):
    #def _ex_plus(self, e):
    #def _ex_minus(self, e):
    #def _ex_multiply(self, e):
    def _ex_divide(self, e):
        # Native divide seemed to cause some issues
        #if self.nm:
        #    #return 'native_divide(' + self.ex(e[0]) +', '+ self.ex(e[1]) + ')'
        #else:
        return self._ex_infix(e, '/')
    def _ex_quotient(self, e):
        return 'floor(' + self._ex_infix(e, '/') + ')'
    def _ex_remainder(self, e):
        return 'fmod(' + self.ex(e[0]) + ', ' + self.ex(e[1]) + ')'
    def _ex_power(self, e):
        if e[1] == 2:
            #TODO: This can be optimised with native functions
            if e.bracket(e[0]):
                out = '(' + self.ex(e[0]) + ') * '
            else:
                out = self.ex(e[0]) + ' * '
            if e.bracket(e[1]):
                return out + '(' + self.ex(e[1]) + ')'
            else:
                return out + self.ex(e[1])
        else:
            return 'pow(' + self.ex(e[0]) + ', ' + self.ex(e[1]) + ')'
    def _ex_sqrt(self, e):
        f = 'native_sqrt' if self.nm else 'sqrt'
        return self._ex_function(e, f)
    def _ex_sin(self, e):
        f = 'native_sin' if self.nm else 'sin'
        return self._ex_function(e, f)
    def _ex_cos(self, e):
        f = 'native_cos' if self.nm else 'cos'
        return self._ex_function(e, f)
    def _ex_tan(self, e):
        f = 'native_tan' if self.nm else 'tan'
        return self._ex_function(e, f)
    def _ex_asin(self, e):
        return self._ex_function(e, 'asin')
    def _ex_acos(self, e):
        return self._ex_function(e, 'acos')
    def _ex_atan(self, e):
        return self._ex_function(e, 'atan')
    def _ex_exp(self, e):
        f = 'native_exp' if self.nm else 'exp'
        return self._ex_function(e, f)
    def _ex_log(self, e):
        f = 'native_log' if self.nm else 'log'
        if len(e) == 1:
            return self._ex_function(e, f)
        return '('+f+'('+self.ex(e[0])+') / '+f+'('+self.ex(e[1])+'))'
    def _ex_log10(self, e):
        f = 'native_log10' if self.nm else 'log10'
        return self._ex_function(e, f)
    def _ex_floor(self, e):
        return self._ex_function(e, 'floor')
    def _ex_ceil(self, e):
        return self._ex_function(e, 'ceil')
    def _ex_abs(self, e):
        return self._ex_function(e, 'fabs')
    def _ex_not(self, e):
        return '!(' + self.ex(e[0]) + ')'
    #def _ex_equal(self, e):
    #def _ex_not_equal(self, e):
    #def _ex_more(self, e):
    #def _ex_less(self, e):
    #def _ex_more_equal(self, e):
    #def _ex_less_equal(self, e):
    def _ex_and(self, e):
        return self._ex_infix_condition(e, '&&')
    def _ex_or(self, e):
        return self._ex_infix_condition(e, '||')
    def _ex_if(self, e):
        return '(%s ? %s : %s)' % (self.ex(e._i), self.ex(e._t), self.ex(e._e))
    def _ex_piecewise(self, e):
        s = []
        n = len(e._i)
        for i in xrange(0, n):
            s.append('(')
            s.append(self.ex(e._i[i]))
            s.append(' ? ')
            s.append(self.ex(e._e[i]))
            s.append(' : ')
        s.append(self.ex(e._e[n]))
        s.append(')'*n)
        return ''.join(s)
    #def _ex_opiecewise(self, e):
    #def _ex_polynomial(self, e):
    #def _ex_spline(self, e):
