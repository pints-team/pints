#
# SymPy expression writer
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
try:
    import sympy as sp
except ImportError:
    pass
class SymPyExpressionWriter(myokit.formats.ExpressionWriter):
    """
    This :class:`ExpressionWriter <myokit.formats.ExpressionWriter>` converts
    Myokit :class:`expressions <myokit.Expression>` to SymPy expressions.
    
    The returned type is a SymPy object, not a string!
    """
    def __init__(self):
        super(SymPyExpressionWriter, self).__init__()
        self._flhs = lambda lhs : str(lhs)
        try:
            test = sp
        except NameError:
            raise Exception('This class requires SymPy to be installed.')
    def set_lhs_function(self, f):
        """
        Sets a naming function, will be called to get the variable name from a
         ``myokit.LhsExpression`` object.

        The argument ``f`` should be a function that takes an ``LhsExpression``
        as input and returns a string.
        """
        self._lhs = f
    def eq(self, eq):
        """
        Converts an equation to a sympy ``Eq`` object.
        """
        return sp.Eq(self.ex(eq.lhs), self.ex(eq.rhs), evaluate=False)
    def ex(self, e):
        """
        Converts an expression to a sympy ``Eq`` object.
        """
        try:
            action = self._op_map[type(e)]
        except KeyError:
            raise Exception('Unsupported type: ' + str(type(e)))
        return action(e)
    def _ex_name(self, e):
        return sp.Symbol(self._flhs(e))
    def _ex_derivative(self, e):
        # Derivatives are treated as symbols, not as derivatives!
        return sp.Symbol(self._flhs(e))
    def _ex_number(self, e):
        return sp.Float(e.eval())
    def _ex_prefix_plus(self, e):
        return self.ex(e[0])
    def _ex_prefix_minus(self, e):
        return sp.Mul(-1, self.ex(e[0]))
    def _ex_plus(self, e):
        return sp.Add(self.ex(e[0]), self.ex(e[1]))
    def _ex_minus(self, e):
        return sp.Add(self.ex(e[0]), sp.Mul(-1, self.ex(e[1])))
    def _ex_multiply(self, e):
        return sp.Mul(self.ex(e[0]), self.ex(e[1]))
    def _ex_divide(self, e):
        return sp.Mul(self.ex(e[0]), sp.Pow(self.ex(e[1]), -1))
    def _ex_quotient(self, e):
        # Integer division a // b == (a - a % b) / b
        a = self.ex(e[0])
        b = self.ex(e[1])
        return sp.Mul(sp.Add(a, sp.Mul(-1, sp.Mod(a, b))), sp.Pow(b, -1))
    def _ex_remainder(self, e):
        return sp.Mod(self.ex(e[0]), self.ex(e[1]))
    def _ex_power(self, e):
        return sp.Pow(self.ex(e[0]), self.ex(e[1]))
    def _ex_sqrt(self, e):
        return sp.Pow(self.ex(e[0]), sp.Rational(1,2))
    def _ex_sin(self, e):
        return sp.sin(self.ex(e[0]))
    def _ex_cos(self, e):
        return sp.cos(self.ex(e[0]))
    def _ex_tan(self, e):
        return sp.tan(self.ex(e[0]))
    def _ex_asin(self, e):
        return sp.asin(self.ex(e[0]))
    def _ex_acos(self, e):
        return sp.acos(self.ex(e[0]))
    def _ex_atan(self, e):
        return sp.atan(self.ex(e[0]))
    def _ex_exp(self, e):
        return sp.exp(self.ex(e[0]))
    def _ex_log(self, e):
        if len(e) == 1:
            return sp.log(self.ex(e[0]))
        else:
            return sp.log(self.ex(e[0]), self.ex(e[1]))
    def _ex_log10(self, e):
        return sp.log(self.ex(e[0]), 10)
    def _ex_floor(self, e):
        return sp.floor(self.ex(e[0]))
    def _ex_ceil(self, e):
        return sp.ceiling(self.ex(e[0]))
    def _ex_abs(self, e):
        return sp.Abs(self.ex(e[0]))
    def _ex_not(self, e):
        return sp.Not(self.ex(e[0]))
    def _ex_equal(self, e):
        return sp.Eq(self.ex(e[0]), self.ex(e[1]))
    def _ex_not_equal(self, e):
        return sp.Ne(self.ex(e[0]), self.ex(e[1]))
    def _ex_more(self, e):
        return sp.Gt(self.ex(e[0]), self.ex(e[1]))
    def _ex_less(self, e):
        return sp.Lt(self.ex(e[0]), self.ex(e[1]))
    def _ex_more_equal(self, e):
        return sp.Ge(self.ex(e[0]), self.ex(e[1]))
    def _ex_less_equal(self, e):
        return sp.Le(self.ex(e[0]), self.ex(e[1]))
    def _ex_and(self, e):
        return sp.And(self.ex(e[0]), self.ex(e[1]))
    def _ex_or(self, e):
        return sp.Or(self.ex(e[0]), self.ex(e[1]))
    def _ex_if(self, e):
        # Sympy piecewise takes only (expression, condition) pairs
        pairs = []
        pairs.append((self.ex(e._t), self.ex(e._i)))
        pairs.append((self.ex(e._e), True))
        return sp.Piecewise(*pairs)
    def _ex_piecewise(self, e):
        pairs = []
        n = len(e) // 2
        for i in xrange(0, n):
            pairs.append((self.ex(e._e[i]), self.ex(e._i[i])))
        pairs.append((self.ex(e._e[n]), True))
        return sp.Piecewise(*pairs)
    def _ex_opiecewise(self, e):
        return self.ex_if(e.if_tree())
    def _ex_polynomial(self, e):
        return self.ex(e.tree())
    def _ex_spline(self, e):
        return self.ex_opiecewise(e)
class SympyExpressionWriter(SymPyExpressionWriter):
    """
    Deprecated name for :class:`SymPyExpressionWriter`.
    """
