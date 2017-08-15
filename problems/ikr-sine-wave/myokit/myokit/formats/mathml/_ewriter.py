#
# MathML expression writer
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
class MathMLExpressionWriter(myokit.formats.ExpressionWriter):
    """
    This :class:`ExpressionWriter <myokit.formats.ExpressionWriter>` translates
    Myokit :class:`expressions <myokit.Expression>` to Content MathML or
    Presentation MathML.
    """
    def __init__(self):
        super(MathMLExpressionWriter, self).__init__()
        # Default element tree class
        import xml.etree.cElementTree as et
        self._et = et
        # Default mode
        self._pres = False
        # Default lhs conversion function
        def flhs(lhs):
            var = lhs.var()
            if type(var) == str:
                # This can happen with time variable of derivative if the
                # proper variable isn't set!
                return var
            return var.qname()
        self._flhs = flhs
        # Default number conversion function
        self._fnum = lambda x : myokit.strfloat(x.eval())
        # Default time variable
        self._tvar = myokit.Name('time')
    def set_element_tree_class(self, et):
        """
        By default this :class:`ExpressionWriter` uses the
        ``xml.etree.cElementTree`` module. This method can be used to change
        this behaviour by passing in a reference to a different implementation.
        """
        self._et = et
    def set_lhs_function(self, f):
        """
        Sets a naming function, will be called to get the variable name from a
         ``myokit.LhsExpression`` object.

        The argument ``f`` should be a function that takes an ``LhsExpression``
        as input and returns a string.
        """
        self._lhs = f
    def set_mode(self, presentation=False):
        """
        Enables or disables Presentation MathML mode.
        """
        self._pres = presentation == True
    def set_time_variable(self, time):
        """
        Sets the time variable to use for this expression writer's derivatives.
        """
        self._tvar = myokit.Name(time)
    def eq(self, eq, element=None):
        """
        Converts an equation to a string.
        
        The optional argument ``element`` can be used to pass in an ElementTree
        element. If given, this element will be updated with the generated xml
        and nothing will be returned.
        """
        if element is None:
            tag = self._et.Element('math')
            tag.attrib['xmlns'] = 'http://www.w3.org/1998/Math/MathML'
        else:
            tag = element
        if self._pres:
            t = self._et.SubElement(tag, 'mrow')
            self.ex(eq.lhs, t)
            x = self._et.SubElement(t, 'mo')
            x.text = '='
            self.ex(eq.rhs, t)
        else:
            t = self._et.SubElement(tag, 'apply')
            self._et.SubElement(t, 'eq')
            self.ex(eq.lhs, t)
            self.ex(eq.rhs, t)
        if element is None:
            enc = 'utf-8'
            return ''.join([self._et.tostring(kid, enc) for kid in tag])
    def ex(self, e, element=None):
        """
        Converts an expression to a string.

        The optional argument ``element`` can be used to pass in an ElementTree
        element. If given, this element will be updated with the generated xml
        and nothing will be returned.
        """
        if element is None:
            tag = self._et.Element('math')
            tag.attrib['xmlns'] = 'http://www.w3.org/1998/Math/MathML'
        else:
            tag = element
        self._ex(e, tag)
        if element is None:
            enc = 'utf-8'
            return ''.join([self._et.tostring(kid, enc) for kid in tag])
    def _ex(self, e, t):
        """
        Writes expression ``e`` to element ``t`` 
        """
        try:
            action = self._op_map[type(e)]
        except KeyError:
            raise Exception('Unsupported type: ' + str(type(e)))
        return action(e, t)
    def _ex_prefix(self, e, t, cml):
        """
        Exports e as a prefix expression with ContentML representation cml.
        """
        bra = e.bracket()
        if self._pres:
            row = self._et.SubElement(t, 'mrow')
            if bra:
                x = self._et.SubElement(row, 'mo')
                x.text = '('
            x = self._et.SubElement(row, 'mo')
            x.text = e.operator_rep()
            self._ex(e[0], row)
            if bra:
                x = self._et.SubElement(row, 'mo')
                x.text = ')'
        else:
            tag = self._et.SubElement(t, 'apply')
            self._et.SubElement(tag, cml)
            self._ex(e[0], tag)  
    def _ex_infix(self, e, t, cml):
        """
        Exports e as an infix expression with ContentML representation cml.
        """
        if self._pres:
            r = self._et.SubElement(t, 'mrow')
            k = self._et.SubElement(r, 'mfenced') if e.bracket(e[0]) else r
            self._ex(e[0], k)
            x = self._et.SubElement(r, 'mo')
            x.text = e.operator_rep()
            k = self._et.SubElement(r, 'mfenced') if e.bracket(e[1]) else r
            self._ex(e[1], k)
        else:
            a = self._et.SubElement(t, 'apply')
            self._et.SubElement(a, cml)
            self._ex(e[0], a)
            self._ex(e[1], a)
    def _ex_function(self, e, t, name):
        """
        Exports e as a function called name.
        """
        if self._pres:
            r = self._et.SubElement(t, 'mrow')
            x = self._et.SubElement(r, 'mi')
            x.text = name
            r = self._et.SubElement(r, 'mfenced')
            for op in e:
                self._ex(op, r)
        else:
            a = self._et.SubElement(t, 'apply')
            self._et.SubElement(a, name)
            for op in e:
                self._ex(op, a)
    def _ex_name(self, e, t):
        x = self._et.SubElement(t, 'mi' if self._pres else 'ci')
        x.text = self._flhs(e)
    def _ex_derivative(self, e, t):
        if self._pres:
            f = self._et.SubElement(t, 'mfrac')
            x = self._et.SubElement(f, 'mi')
            x.text = 'd' + self._flhs(e[0])
            x = self._et.SubElement(f, 'mi')
            x.text = 'dt'
        else:
            a = self._et.SubElement(t, 'apply')
            self._et.SubElement(a, 'diff')
            self._ex(self._tvar, self._et.SubElement(a, 'bvar'))
            self._ex(e[0], a)
    def _ex_number(self, e, t):
        x = self._et.SubElement(t, 'mn' if self._pres else 'cn')
        x.text = self._fnum(e)
    def _ex_prefix_plus(self, e, t):
        return self._ex_prefix(e, t, 'plus')
    def _ex_prefix_minus(self, e, t):
        return self._ex_prefix(e, t, 'minus')
    def _ex_plus(self, e, t):
        return self._ex_infix(e, t, 'plus')
    def _ex_minus(self, e, t):
        return self._ex_infix(e, t, 'minus')
    def _ex_multiply(self, e, t):
        return self._ex_infix(e, t, 'times')
    def _ex_divide(self, e, t):
        if self._pres:
            r = self._et.SubElement(t, 'mfrac')
            k = self._et.SubElement(r, 'mfenced') if e.bracket(e[0]) else r
            self._ex(e[0], k)
            k = self._et.SubElement(r, 'mfenced') if e.bracket(e[1]) else r
            self._ex(e[1], k)
        else:
            a = self._et.SubElement(t, 'apply')
            self._et.SubElement(a, 'divide')
            self._ex(e[0], a)
            self._ex(e[1], a)
    def _ex_quotient(self, e, t):
        return self._ex_infix(e, t, 'quotient')
    def _ex_remainder(self, e, t):
        return self._ex_infix(e, t, 'rem')
    def _ex_power(self, e, t):
        if self._pres:
            x = self._et.SubElement(t, 'msup')
            self._ex(e[0], x)
            self._ex(e[1], x)
        else:
            return self._ex_function(e, t, 'power')
    def _ex_sqrt(self, e, t):
        return self._ex_function(e, t, 'root')
    def _ex_sin(self, e, t):
        return self._ex_function(e, t, 'sin')
    def _ex_cos(self, e, t):
        return self._ex_function(e, t, 'cos')
    def _ex_tan(self, e, t):
        return self._ex_function(e, t, 'tan')
    def _ex_asin(self, e, t):
        return self._ex_function(e, t, 'arcsin')
    def _ex_acos(self, e, t):
        return self._ex_function(e, t, 'arccos')
    def _ex_atan(self, e, t):
        return self._ex_function(e, t, 'arctan')
    def _ex_exp(self, e, t):
        if self._pres:
            r = self._et.SubElement(t, 'msup')
            x = self._et.SubElement(r, 'mi')
            x.text = 'e'
            self._ex(e[0], r)
        else:
            a = self._et.SubElement(t, 'apply')
            x = self._et.SubElement(a, 'exp')
            self._ex(e[0], a)
    def _ex_log(self, e, t):
        # myokit.log(a)   > ln(a)
        # myokit.log(a,b) > log(b, a)
        # myokit.log10(a) > log(a)
        if self._pres:
            if len(e) == 1:
                r = self._et.SubElement(t, 'mrow')
                x = self._et.SubElement(r, 'mi')
                x.text = 'ln'
                x = self._et.SubElement(r, 'mfenced')
                self._ex(e[0], x)
            else:
                r = self._et.SubElement(t, 'mrow')
                s = self._et.SubElement(r, 'msub')
                x = self._et.SubElement(s, 'mi')
                x.text = 'log'
                self._ex(e[1], s)
                s = self._et.SubElement(r, 'mfenced')
                self._ex(e[0], s)
        else:
            if len(e) == 1:
                a = self._et.SubElement(t, 'apply')
                x = self._et.SubElement(a, 'ln')
                self._ex(e[0], a)
            else:
                a = self._et.SubElement(t, 'apply')
                x = self._et.SubElement(a, 'log')
                x = self._et.SubElement(a, 'logbase')
                self._ex(e[1], x)
                self._ex(e[0], a)
    def _ex_log10(self, e, t):
        return self._ex_function(e, t, 'log')
    def _ex_floor(self, e, t):
        return self._ex_function(e, t, 'floor')
    def _ex_ceil(self, e, t):
        return self._ex_function(e, t, 'ceiling')
    def _ex_abs(self, e, t):
        return self._ex_function(e, t, 'abs')
    def _ex_not(self, e, t):
        return self._ex_prefix(e, t, 'plus')
    def _ex_equal(self, e, t):
        return self._ex_infix(e, t, 'eq')
    def _ex_not_equal(self, e, t):
        return self._ex_infix(e, t, 'neq')
    def _ex_more(self, e, t):
        return self._ex_infix(e, t, 'gt')
    def _ex_less(self, e, t):
        return self._ex_infix(e, t, 'lt')
    def _ex_more_equal(self, e, t):
        return self._ex_infix(e, t, 'geq')
    def _ex_less_equal(self, e, t):
        return self._ex_infix(e, t, 'leq')
    def _ex_and(self, e, t):
        return self._ex_infix(e, t, 'and')
    def _ex_or(self, e, t):
        return self._ex_infix(e, t, 'or')
    def _ex_if(self, e, t):
        return self._ex_piecewise(e.piecewise(), t)
    def _ex_piecewise(self, e, t):
        if self._pres:
            w = self._et.SubElement(t, 'piecewise')
            for k, cond in enumerate(e._i):
                p = self._et.SubElement(w, 'piece')
                self._ex(e._e[k], p)
                self._ex(cond, p)
            p = self._et.SubElement(w, 'otherwise')
            self._ex(e._e[-1], p)
        else:
            w = self._et.SubElement(t, 'piecewise')
            for k, cond in enumerate(e._i):
                p = self._et.SubElement(w, 'piece')
                self._ex(e._e[k], p)
                self._ex(cond, p)
            p = self._et.SubElement(w, 'otherwise')
            self._ex(e._e[-1], p)
    def _ex_opiecewise(self, e, t):
        return self._ex_if(e.if_tree())
    def _ex_polynomial(self, e, t):
        return self.ex(e.tree())
    def _ex_spline(self, e, t):
        return self._ex_opiecewise(e)
