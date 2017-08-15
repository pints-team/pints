#
# Latex expression writer
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
class LatexExpressionWriter(myokit.formats.ExpressionWriter):
    """
    This :class:`ExpressionWriter <myokit.formats.ExpressionWriter>` translates
    Myokit :class:`expressions <myokit.Expression>` to their Tex equivalent.
    """
    def __init__(self):
        super(LatexExpressionWriter, self).__init__()
        # Default time variable
        self._time = 't'
        # Default lhs function
        def fhls(lhs):
            var = self._prepare_name(lhs.var().uname())
            var = '\\text{' + var + '}' # Depends on amsmath package!
            if isinstance(lhs, myokit.Derivative):
                #var = '\\dot{' + var + '}'
                var = '\\frac{d}{d\\text{t}}' + var
            return var
        self._flhs = fhls
    def set_lhs_function(self, f):
        """
        Sets a naming function, will be called to get the variable name from a
         ``myokit.LhsExpression`` object.

        The argument ``f`` should be a function that takes an ``LhsExpression``
        as input and returns a string.
        """
        self._flhs = f
    def set_time_variable_name(self, name='t'):
        """
        Sets a name to use for the time variable in derivatives
        """
        self._time = self._prepare_name(name)
    def eq(self, eq):
        """
        Converts an equation to a string.
        """
        return self.ex(eq.lhs) + ' = ' +  self.ex(eq.rhs)
    def ex(self, e):
        """
        Converts an expression to a string.
        """
        b = []
        self._ex(e, b)
        return ''.join(b)
    def _prepare_name(self, text):
        """
        Prepares a name for use in latex
        """
        text = str(text)
        return text.replace('_', '\_')
    def _ex(self, e, b):
        try:
            action = self._op_map[type(e)]
        except KeyError:
            raise Exception('Unsupported type: ' + str(type(e)))
        action(e, b)
    def _ex_infix(self, e, b, op):
        """
        Handles _ex() for infix operators
        """
        if e.bracket(e[0]):
            b.append('\\left(')
        self._ex(e[0], b)
        if e.bracket(e[0]):
            b.append('\\right)')
        b.append(op)
        if e.bracket(e[1]):
            b.append('\\left(')
        self._ex(e[1], b)
        if e.bracket(e[1]):
            b.append('\\right)')
    def _ex_function(self, e, b, func):
        """
        Handles _ex() for function operators
        """
        b.append(func)
        b.append('\\left(')
        b.append(', '.join([self.ex(x) for x in e]))
        b.append('\\right)')
    def _ex_infix_condition(self, e, b, op):
        """
        Handles _ex() for infix condition operators
        """
        b.append('\\left(')
        self._ex(e[0], b)
        b.append(op)
        self._ex(e[1], b)
        b.append('\\right)')
    def _ex_name(self, e, b):
        b.append(self._flhs(e))
    def _ex_derivative(self, e, b):
        b.append(self._flhs(e))
    def _ex_number(self, e, b):
        b.append(myokit.strfloat(e))
    def _ex_prefix_plus(self, e, b):
        self._ex(e[0], b)
    def _ex_prefix_minus(self, e, b):
        b.append('-')
        if e.bracket():
            b.append('\\left(')
        self._ex(e[0], b)
        if e.bracket():
            b.append('\\right)')
    def _ex_plus(self, e, b):
        self._ex_infix(e, b, '+')
    def _ex_minus(self, e, b):
        self._ex_infix(e, b, '-')
    def _ex_multiply(self, e, b):
        self._ex_infix(e, b, '*')
    def _ex_divide(self, e, b):
        b.append('\\frac{')
        self._ex(e[0], b)
        b.append('}{')
        self._ex(e[1], b)
        b.append('}')
    def _ex_quotient(self, e, b):
        self._ex_function(e, b, 'quotient')
    def _ex_remainder(self, e, b):
        self._ex_function(e, b, 'remainder')
    def _ex_power(self, e, b):
        if e.bracket(e[0]):
            b.append('\\left(')
        self._ex(e[0], b)
        if e.bracket(e[0]):
            b.append('\\right)')
        b.append('^{')
        self._ex(e[1], b)
        b.append('}')
    def _ex_sqrt(self, e, b):
        b.append('\\sqrt{')
        self._ex(e[0], b)
        b.append('}')
    def _ex_sin(self, e, b):
        self._ex_function(e, b, '\\sin')
    def _ex_cos(self, e, b):
        self._ex_function(e, b, '\\cos')
    def _ex_tan(self, e, b):
        self._ex_function(e, b, '\\tan')
    def _ex_asin(self, e, b):
        self._ex_function(e, b, '\\arcsin')
    def _ex_acos(self, e, b):
        self._ex_function(e, b, '\\arccos')
    def _ex_atan(self, e, b):
        self._ex_function(e, b, '\\arctan')
    def _ex_exp(self, e, b):
        self._ex_function(e, b, '\\exp')
    def _ex_log(self, e, b):
        b.append('\\log')
        if len(e) > 1:
            b.append('_{')       
            self._ex(e[1], b)
            b.append('}')
        b.append('\\left(')
        self._ex(e[0], b)
        b.append('\\right)')
    def _ex_log10(self, e, b):
        return self._ex_function(e, 'log_{10}')
    def _ex_floor(self, e, b):
        b.append('\\left\\lfloor')
        self._ex(e[0], b)
        b.append('\\right\\rfloor')
    def _ex_ceil(self, e, b):
        b.append('\\left\\lceil')
        self._ex(e[0], b)
        b.append('\\right\\rceil')
    def _ex_abs(self, e, b):
        b.append('\\lvert')
        self._ex(e[0], b)
        b.append('\\rvert')
    def _ex_not(self, e, b):
        self._ex_function(e, b, '\\not')
    def _ex_equal(self, e, b):
        self._ex_infix_condition(e, b, '==')
    def _ex_not_equal(self, e, b):
        self._ex_infix_condition(e, b, '!=')
    def _ex_more(self, e, b):
        self._ex_infix_condition(e, b, '>')
    def _ex_less(self, e, b):
        self._ex_infix_condition(e, b, '<')
    def _ex_more_equal(self, e, b):
        self._ex_infix_condition(e, b, '>=')
    def _ex_less_equal(self, e, b):
        self._ex_infix_condition(e, b, '<=')
    def _ex_and(self, e, b):
        self._ex_infix_condition(e, b, '\\and')
    def _ex_or(self, e, b):
        self._ex_infix_condition(e, b, '\\or')
    def _ex_if(self, e, b):
        self._ex_function(e, b, 'if') # No slashes!
    def _ex_piecewise(self, e, b):
        self._ex_function(e, b, '\\piecewise')
