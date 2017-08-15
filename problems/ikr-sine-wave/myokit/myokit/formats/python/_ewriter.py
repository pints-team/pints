#
# Python expression writer and keywords
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
import myokit.formats
class PythonExpressionWriter(myokit.formats.ExpressionWriter):
    """
    This :class:`ExpressionWriter <myokit.formats.ExpressionWriter>` translates
    Myokit :class:`expressions <myokit.Expression>` to their Python
    equivalent.
    """
    def __init__(self):
        super(PythonExpressionWriter, self).__init__()
        self.flhs = lambda v: str(v)
        self.function_prefix = 'math.'
    def set_lhs_function(self, f):
        """
        Sets a naming function, will be called to get the variable name from a
         ``myokit.LhsExpression`` object.

        The argument ``f`` should be a function that takes an ``LhsExpression``
        as input and returns a string.
        """
        self.flhs = f
    def _ex_infix(self, e, op):
        """
        Handles ex() for infix operators
        """
        if e.bracket(e[0]):
            out = '(' + self.ex(e[0]) + ') ' + op
        else:
            out = self.ex(e[0]) + ' ' + op
        if e.bracket(e[1]):
            return out + ' (' + self.ex(e[1]) + ')'
        else:
            return out + ' ' + self.ex(e[1])
    def _ex_function(self, e, func):
        """
        Handles ex() for function operators
        """
        return self.function_prefix + func \
            + '(' + ', '.join([self.ex(x) for x in e]) + ')'
    def _ex_infix_condition(self, e, op):
        """
        Handles ex() for infix condition operators
        """
        return '(' + self.ex(e[0]) + ' ' + op + ' ' + self.ex(e[1]) + ')'
    def _ex_name(self, e):
        return self.flhs(e)
    def _ex_derivative(self, e):
        return self.flhs(e)
    def _ex_number(self, e):
        return myokit.strfloat(e)
    def _ex_prefix_plus(self, e):
        return self.ex(e[0])
    def _ex_prefix_minus(self, e):
        return '-('+ self.ex(e[0]) +')' if e.bracket() else '-' + self.ex(e[0])            
    def _ex_plus(self, e):
        return self._ex_infix(e, '+')
    def _ex_minus(self, e):
        return self._ex_infix(e, '-')
    def _ex_multiply(self, e):
        return self._ex_infix(e, '*')
    def _ex_divide(self, e):
        return self._ex_infix(e, '/')
    def _ex_quotient(self, e):
        return self._ex_infix(e, '//')
    def _ex_remainder(self, e):
        return self._ex_infix(e, '%')
    def _ex_power(self, e):
        return self._ex_infix(e, '**')
    def _ex_sqrt(self, e):
        return self._ex_function(e, 'sqrt')
    def _ex_sin(self, e):
        return self._ex_function(e, 'sin')
    def _ex_cos(self, e):
        return self._ex_function(e, 'cos')
    def _ex_tan(self, e):
        return self._ex_function(e, 'tan')
    def _ex_asin(self, e):
        return self._ex_function(e, 'asin')
    def _ex_acos(self, e):
        return self._ex_function(e, 'acos')
    def _ex_atan(self, e):
        return self._ex_function(e, 'atan')
    def _ex_exp(self, e):
        return self._ex_function(e, 'exp')
    def _ex_log(self, e):
        return self._ex_function(e, 'log')
    def _ex_log10(self, e):
        return self._ex_function(e, 'log10')
    def _ex_floor(self, e):
        return self._ex_function(e, 'floor')
    def _ex_ceil(self, e):
        return self._ex_function(e, 'ceil')
    def _ex_abs(self, e):
        return 'abs(' + self.ex(e[0]) + ')'
    def _ex_not(self, e):
        return 'not(' + self.ex(e[0]) + ')'
    def _ex_equal(self, e):
        return self._ex_infix_condition(e, '==')
    def _ex_not_equal(self, e):
        return self._ex_infix_condition(e, '!=')
    def _ex_more(self, e):
        return self._ex_infix_condition(e, '>')
    def _ex_less(self, e):
        return self._ex_infix_condition(e, '<')
    def _ex_more_equal(self, e):
        return self._ex_infix_condition(e, '>=')
    def _ex_less_equal(self, e):
        return self._ex_infix_condition(e, '<=')
    def _ex_and(self, e):
        return self._ex_infix_condition(e, 'and')
    def _ex_or(self, e):
        return self._ex_infix_condition(e, 'or')
    def _ex_if(self, e):
        return '(' + self.ex(e._t) + ' if ' + self.ex(e._i) + ' else ' \
                   + self.ex(e._e) + ')'
    def _ex_piecewise(self, e):
        s = ''
        n = len(e) // 2
        for i in xrange(0, n):
            s += '('+self.ex(e._e[i])+' if '+self.ex(e._i[i])+' else '
        s += self.ex(e._e[n])
        s += ')'*n
        return s
    def _ex_opiecewise(self, e):
        return self._ex_if(e.if_tree())
    def _ex_polynomial(self, e):
        return self.ex(e.tree())
    def _ex_spline(self, e):
        return self._ex_opiecewise(e)
class NumpyExpressionWriter(PythonExpressionWriter):
    """
    This :class:`ExpressionWriter <myokit.formats.ExpressionWriter>` translates
    Myokit :class:`expressions <myokit.Expression>` to Python expressions
    intended for use in NumPy arrays.
    """
    def __init__(self):
        super(NumpyExpressionWriter, self).__init__()
        self.function_prefix = 'numpy.'
    #def _ex_name(self, e):
    #def _ex_derivative(self, e):
    #def _ex_number(self, e):
    #def _ex_prefix_plus(self, e):
    #def _ex_prefix_minus(self, e):
    #def _ex_plus(self, e):
    #def _ex_minus(self, e):
    #def _ex_multiply(self, e):
    #def _ex_divide(self, e):
    #def _ex_quotient(self, e):
    #def _ex_remainder(self, e):
    #def _ex_power(self, e):
    #def _ex_sqrt(self, e):
    #def _ex_sin(self, e):
    #def _ex_cos(self, e):
    #def _ex_tan(self, e):
    def _ex_asin(self, e):
        return self._ex_function(e, 'arcsin')
    def _ex_acos(self, e):
        return self._ex_function(e, 'arccos')
    def _ex_atan(self, e):
        return self._ex_function(e, 'arctan')
    #def _ex_exp(self, e):
    #def _ex_log(self, e):
    #def _ex_log10(self, e):
    #def _ex_floor(self, e):
    #def _ex_ceil(self, e):
    #def _ex_abs(self, e):
    #def _ex_not(self, e):
    #def _ex_equal(self, e):
    #def _ex_not_equal(self, e):
    #def _ex_more(self, e):
    #def _ex_less(self, e):
    #def _ex_more_equal(self, e):
    #def _ex_less_equal(self, e):
    #def _ex_and(self, e):
    #def _ex_or(self, e):
    def _ex_if(self, e):
        return self.function_prefix + 'select([' + self.ex(e._i) + '], [' \
               + self.ex(e._t) + '], ' + self.ex(e._e) + ')'
    def _ex_piecewise(self, e):
        n = len(e._i)
        s = [self.function_prefix, 'select([']
        for i in xrange(0, n):
            s.append(self.ex(e._i[i]))
            s.append(', ')
        s.append('], [')
        for i in xrange(0, n):
            s.append(self.ex(e._e[i]))
            s.append(', ')
        s.append('], ')
        s.append(self.ex(e._e[n]))
        s.append(')')
        return ''.join(s)
    def _ex_opiecewise(self, e):
        return self._ex_piecewise(e.piecewise())
    #def _ex_polynomial(self, e):
    #def _ex_spline(self, e):
