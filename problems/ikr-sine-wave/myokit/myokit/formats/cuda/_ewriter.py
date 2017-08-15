#
# CUDA expression writer
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
class CudaExpressionWriter(PythonExpressionWriter):
    """
    This :class:`ExpressionWriter <myokit.formats.ExpressionWriter>` translates
    Myokit :class:`expressions <myokit.Expression>` to their CUDA equivalents.
    """
    def __init__(self, precision=myokit.SINGLE_PRECISION):
        super(CudaExpressionWriter, self).__init__()
        self.function_prefix = ''
        self.sp = (precision == myokit.SINGLE_PRECISION)
    #def _ex_name(self, e):
    #def _ex_derivative(self, e):
    def _ex_number(self, e):
        return myokit.strfloat(e) + 'f' if self.sp else myokit.strfloat(e)
    #def _ex_prefix_plus(self, e):
    def _ex_prefix_minus(self, e):
        if e.bracket():
            return '-(' + self.ex(e[0]) + ')'
        else:
            return '-' + self.ex(e[0])
    #def _ex_plus(self, e):
    #def _ex_minus(self, e):
    #def _ex_multiply(self, e):
    #def _ex_divide(self, e):
    def _ex_quotient(self, e):
        f = 'floorf' if self.sp else 'floor'
        return f + '(' + self._ex_infix(e, '/') + ')'
    def _ex_remainder(self, e):
        f = 'fmodf' if self.sp else 'fmod'
        return f + '(' + self.ex(e[0]) + ', ' + self.ex(e[1]) + ')'
    def _ex_power(self, e):
        if e[1] == 2:
            if e.bracket(e[0]):
                out = '(' + self.ex(e[0]) + ') * '
            else:
                out = self.ex(e[0]) + ' * '
            if e.bracket(e[1]):
                return out + '(' + self.ex(e[1]) + ')'
            else:
                return out + self.ex(e[1])
        else:
            f = 'powf' if self.sp else 'pow'
            return f + '(' + self.ex(e[0]) + ', ' + self.ex(e[1]) + ')'
    def _ex_sqrt(self, e):
        f = 'sqrtf' if self.sp else 'sqrt'
        return self._ex_function(e, f)
    def _ex_sin(self, e):
        f = 'sinf' if self.sp else 'sin'
        return self._ex_function(e, f)
    def _ex_cos(self, e):
        f = 'cosf' if self.sp else 'cos'
        return self._ex_function(e, f)
    def _ex_tan(self, e):
        f = 'tanf' if self.sp else 'tan'
        return self._ex_function(e, f)
    def _ex_asin(self, e):
        f = 'asinf' if self.sp else 'asin'
        return self._ex_function(e, f)
    def _ex_acos(self, e):
        f = 'acosf' if self.sp else 'acos'
        return self._ex_function(e, f)
    def _ex_atan(self, e):
        f = 'atanf' if self.sp else 'atan'
        return self._ex_function(e, f)
    def _ex_exp(self, e):
        f = 'expf' if self.sp else 'exp'
        return self._ex_function(e, f)
    def _ex_log(self, e):
        f = 'logf' if self.sp else 'log'
        if len(e) == 1:
            return self._ex_function(e, f)
        return '('+f+'('+self.ex(e[0])+') / '+f+'('+self.ex(e[1])+'))'
    def _ex_log10(self, e):
        f = 'log10f' if self.sp else 'log10'
        return self._ex_function(e, f)
    def _ex_floor(self, e):
        f = 'floorf' if self.sp else 'floor'
        return self._ex_function(e, f)
    def _ex_ceil(self, e):
        f = 'ceilf' if self.sp else 'ceil'
        return self._ex_function(e, f)
    def _ex_abs(self, e):
        f = 'fabsf' if self.sp else 'fabs'
        return self._ex_function(e, f)
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
