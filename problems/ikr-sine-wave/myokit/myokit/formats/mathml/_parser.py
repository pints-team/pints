#
# Converts MathML to Myokit Expressions
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
from myokit.mxml import dom_child, dom_next
class MathMLError(myokit.ExportError):
    """
    Raised if an error occurs during MathML import.
    """
def parse_mathml(s):
    """
    Parses a mathml string that should contain a single expression.
    """
    import xml.dom.minidom
    x = xml.dom.minidom.parseString(s)
    return parse_mathml_rhs(dom_child(x))
def parse_mathml_rhs(node, var_table=None, logger=None,
        number_post_processor=None, derivative_post_processor=None):
    """
    Takes a MathML node ``node`` (using the ``xml.dom.Node`` interface) and
    parses its contents into a :class:`myokit.Expression`.
    
    Not all of MathML is supported (so no integrals, set theory etc.) but only
    a subset common to electrophysiology. In addition, some not-so-common
    elements are supported because they're allowed appear in
    :class:`CellML <myokit.formats.cellml.CellMLImporter>` documents.

    Variable names will be returned as strings, unless the optional dict
    argument ``var_table`` is given. Note that the :class:`myokit.VarOwner`
    classes support the dict interface.

    If the argument ``logger`` is given this will be used to log messages to,
    assuming the :class:`myokit.TextLogger` interface.
    
    Optional post-processing of numbers (``<cn>`` tags) can be added by passing
    in a callable ``number_post_processor(tag, number)``. This will be called
    after parsing each ``<cn>`` tag with the original node as the first
    argument (as an ``xml.dom.minidom`` node), and the created number object as
    the second (as a :class:`myokit.Number`). The function must return a new
    :class:`myokit.Number` object.
    
    Optional checking of derivatives (``<diff>`` tags) can be added by passing
    in a callable ``derivative_post_processor(time)``. This will be called with
    the :class:`myokit.Name` representing the variable with respect to which
    the derivative is being taken. This allows importers to ensure only
    time-derivatives are being loaded.
    
    The following MathML elements are recognised:

    Literals and references
    
    ``<ci>``
        Becomes a :class:`myokit.Name`.
    ``<diff>`` (with ``<bvar>`` and ``<degree>``)
        Becomes a :class:`myokit.Derivative`. Only first-order derivatives are
        supported. To check if the derivatives are all time-derivatives, the
        derivative post-processing function can be used.
    ``<cn>``
        Becomes a :class:`myokit.Number`. To process units which may be present
        in the tag's attributes (esp. in CellML) the number post-processing
        function can be used. 

    Algebra
    
    ``<plus>``
        Becomes a :class:`myokit.PrefixPlus`, a :class`myokit.Plus` or a tree
        of :class:`myokit.Plus` elements.
    ``<minus>``
        Becomes a :class:`myokit.PrefixMinus`, a :class`myokit.Minus` or a tree
        of :class:`myokit.Minus` elements.
    ``<times>``
        Becomes a :class:`myokit.Multiply` or a tree of
        :class:`myokit.Multiply` elements.
    ``<divide>``
        Becomes a :class:`myokit.Divide` or a tree of :class:`myokit.Divide`
        elements.
    ``<apply>``
        Used to indicate the tree structure of the equation. These get
        translated but don't have a Myokit counterpart.
        
    Functions
        
    ``<power>``
        Becomes a :class:`myokit.Power`.
    ``<root>`` (with ``<degree>``)
        Becomes a :class:`myokit.Sqrt`.
    ``<exp>``
        Becomes a :class:`myokit.Exp`.
    ``<ln>``
        Becomes a :class:`myokit.Log`.
    ``<log>`` (with ``<logbase>``)
        Becomes a :class:`myokit.Log10` or a :class:`myokit.Log`.
    ``<abs>``
        Becomes a :class:`myokit.Abs`.    
    ``<floor>``
        Becomes a :class:`myokit.Floor`.
    ``<ceiling>``
        Becomes a :class:`myokit.Ceil`.
    ``<quotient>``
        Becomes a :class:`myokit.Quotient`.
    ``<rem>``
        Becomes a :class:`myokit.Remainder`.
        
    Trigonometry
        
    ``<sin>``, ``<cos>`` and ``<tan>``
        Become :class:`myokit.Sin`, :class:`myokit.Cos` and
        :class:`myokit.Tan`.
    ``<arcsin>``, ``<arccos>`` and ``<arctan>``
        Become :class:`myokit.ASin`, :class:`myokit.ACos` and
        :class:`myokit.ATan`.
    ``<csc>``, ``<sec>`` and ``<cot>``
        Become ``1/sin``, ``1/cos`` and ``1/tan``.
    ``<arccsc>``, ``<arcsec>`` and ``<arccot>``
        Become ``asin(1/x)``, ``acos(1/x)`` and ``atan(1/x)``.

    Hyperbolic trigonometry
    
    ``<sinh>``
        Becomes ``0.5 * (exp(x) - exp(-x))``.
    ``<cosh>``
        Becomes ``0.5 * (exp(x) + exp(-x))``.
    ``<tanh>``
        Becomes ``(exp(2 * x) - 1) / (exp(2 * x) + 1)``.
    ``<arcsinh>``
        Becomes ``log(x + sqrt(1 + x*x))``.
    ``<arccosh>``
        Becomes ``log(x + sqrt(x + 1) * sqrt(x - 1))``.
    ``<arctanh>``
        Becomes ``0.5 * (log(1 + x) - log(1 - x))``.
    ``<csch>``
        Becomes ``2 / (exp(x) - exp(-x))``.
    ``<sech>``
        Becomes ``2 / (exp(x) + exp(-x))``.
    ``<coth>``
        Becomes ``(exp(2 * x) + 1) / (exp(2 * x) - 1)``.
    ``<arccsch>``
        Becomes ``log(sqrt(1 + 1 / x^2) + 1 / x)``.
    ``<arcsech>``
        Becomes ``log(sqrt(1 / x - 1) * sqrt(1 / x + 1) + 1 / x)``
    ``<arccoth>``
        Becomes ``0.5 * (log(1 + 1/x) - log(1 - 1/x))``.
        
    Logic and relations
    
    ``<piecewise>``, ``<piece>`` and ``<otherwise>``
        Becomes a :class:`myokit.Piecewise`.
    ``<and>``, ``<or>`` and ``<not>``
        Become :class:`myokit.And`, :class:`myokit.Or` and :class:`myokit.Not`.
    ``<xor>``
        Becomes ``(x or y) and not(x and y)``
    ``<eq>`` and ``<neq>``
        Becomes :class:`myokit.Equal` and :class:`NotEqual`.
    ``<lt>`` and ``<gt>``
        Become :class:`myokit.Less` and :class:`myokit.More`.
    ``<leq>`` and ``<geq>``
        Become :class:`myokit.LessEqual` and :class:`myokit.MoreEqual`.
        
    Constants

    ``<pi>``
        Becomes ``3.14159265358979323846``
    ``<exponentiale>``
        Becomes ``exp(1)``
    ``<true>``
        Becomes ``1``
    ``<false>``
        Becomes ``0``

    There are a few elements supported by CellML, but not by Myokit.
    
    ``<semantics>``, ``<annotation>`` and ``<annotation-xml>``
        These are not present in any electrophysiology model in the database.
    ``<notanumber>`` and ``<infinity>``
        These have no place in an ODE.
    ``<factorial>``
        There is no cardiac electrophysiology model in the database that uses
        these. Plus, factorials require the idea of integers (Myokit only has
        Reals) and only factorial(x) for x in [0,1,2,...,12] can be
        calculated without integer overflows.
    
    Finally, Myokit, but not CellML, supports quotients and remainders.

    """
    def parsex(node):
        """
        Parses a mathml expression.
        """
        def chain(kind, node, unary=None):
            """
            Parses operands for chained operators (for example plus, minus,
            times and division).

            The argument ``kind`` must be the myokit expression type being
            parsed, ``node`` is a DOM node and ``unary``, if given, should be
            the unary expression type (unary Plus or unary Minus).
            """
            ops = []
            node = dom_next(node)
            while node:
                ops.append(parsex(node))
                node = dom_next(node)
            n = len(ops)
            if n < 1:
                raise MathMLError('Operator needs at least one operand.')
            if n < 2:
                if unary:
                    return unary(ops[0])
                else:
                    raise MathMLError('Operator needs at least two operands')
            ex = kind(ops[0], ops[1])
            for i in xrange(2, n):
                ex = kind(ex, ops[i])
            return ex
        # Start parsing
        name = node.tagName
        if name == 'apply':
            # Brackets, can be ignored in an expression tree.
            return parsex(dom_child(node))
        elif name == 'ci':
            # Reference
            var = str(node.firstChild.data).strip()
            if var_table:
                try:
                    var = var_table[var]
                except KeyError:
                    logger.warn('Unable to resolve reference to <' + str(var)
                        + '>.')
            return myokit.Name(var)
        elif name == 'diff':
            # Derivative
            # Check time variable
            bvar = dom_next(node, 'bvar')
            if derivative_post_processor:
                derivative_post_processor(parsex(dom_child(bvar, 'ci')))
            # Check degree, if given
            d = dom_child(bvar, 'degree')
            if d is not None:
                d = parsex(dom_child(d, 'cn')).eval()
                if not d == 1:
                    raise MathMLError('Only derivatives of degree one are'
                        ' supported.')
            # Create derivative and return
            x = dom_next(node, 'ci')
            if x is None:
                raise MathMLError('Derivative of an expression found: only'
                    ' derivatives of variables are supported.')
            return myokit.Derivative(parsex(x))
        elif name == 'cn':
            # Number
            number = parse_mathml_number(node, logger)
            if number_post_processor:
                return number_post_processor(node, number)
            return number
        #
        # Algebra
        #
        elif name == 'plus':
            return chain(myokit.Plus, node, myokit.PrefixPlus)
        elif name == 'minus':
            return chain(myokit.Minus, node, myokit.PrefixMinus)
        elif name == 'times':
            return chain(myokit.Multiply, node)
        elif name == 'divide':
            return chain(myokit.Divide, node)
        #
        # Functions
        #
        elif name == 'exp':
            return myokit.Exp(parsex(dom_next(node)))
        elif name == 'ln':
            return myokit.Log(parsex(dom_next(node)))
        elif name == 'log':
            if dom_next(node).tagName != 'logbase':
                return myokit.Log10(parsex(dom_next(node)))
            else:
                return myokit.Log(
                    parsex(dom_next(dom_next(node))),
                    parsex(dom_child(dom_next(node))))
        elif name == 'root':
            # Check degree, if given
            next = dom_next(node)
            if next.tagName == 'degree':
                # Degree given, return x^(1/d) unless d is 2
                d = parsex(dom_child(next))
                x = parsex(dom_next(next))
                if d.is_literal() and d.eval() == 2:
                    return myokit.Sqrt(x)
                return myokit.Power(x, myokit.Divide(myokit.Number(1), d))
            else:
                return myokit.Sqrt(parsex(next))
        elif name == 'power':
            n2 = dom_next(node)
            return myokit.Power(parsex(n2), parsex(dom_next(n2)))
        elif name == 'floor':
            return myokit.Floor(parsex(dom_next(node)))
        elif name == 'ceiling':
            return myokit.Ceil(parsex(dom_next(node)))
        elif name == 'abs':
            return myokit.Abs(parsex(dom_next(node)))
        elif name == 'quotient':
            n2 = dom_next(node)
            return myokit.Quotient(parsex(n2), parsex(dom_next(n2)))
        elif name == 'rem':
            n2 = dom_next(node)
            return myokit.Remainder(parsex(n2), parsex(dom_next(n2)))
        #
        # Trigonometry
        #
        elif name == 'sin':
            return myokit.Sin(parsex(dom_next(node)))
        elif name == 'cos':
            return myokit.Cos(parsex(dom_next(node)))
        elif name == 'tan':
            return myokit.Tan(parsex(dom_next(node)))
        elif name == 'arcsin':
            return myokit.ASin(parsex(dom_next(node)))
        elif name == 'arccos':
            return myokit.ACos(parsex(dom_next(node)))
        elif name == 'arctan':
            return myokit.ATan(parsex(dom_next(node)))
        #
        # Redundant trigonometry (CellML includes this)
        #
        elif name == 'csc':
            # Cosecant: csc(x) = 1 / sin(x)
            return myokit.Divide(myokit.Number(1),
                myokit.Sin(parsex(dom_next(node))))
        elif name == 'sec':
            # Secant: sec(x) = 1 / cos(x)
            return myokit.Divide(myokit.Number(1),
                myokit.Cos(parsex(dom_next(node))))
        elif name == 'cot':
            # Contangent: cot(x) = 1 / tan(x)
            return myokit.Divide(myokit.Number(1),
                myokit.Tan(parsex(dom_next(node))))
        elif name == 'arccsc':
            # ArcCosecant: acsc(x) = asin(1/x)
            return myokit.ASin(myokit.Divide(myokit.Number(1),
                parsex(dom_next(node))))
        elif name == 'arcsec':
            # ArcSecant: asec(x) = acos(1/x)
            return myokit.ACos(myokit.Divide(myokit.Number(1),
                parsex(dom_next(node))))
        elif name == 'arccot':
            # ArcCotangent: acot(x) = atan(1/x)
            return myokit.ATan(myokit.Divide(myokit.Number(1),
                parsex(dom_next(node))))
        #
        # Hyperbolic trigonometry (CellML again)
        #
        elif name == 'sinh':
            # Hyperbolic sine: sinh(x) = 0.5 * (e^x - e^-x)
            x = parsex(dom_next(node))
            return myokit.Multiply(myokit.Number(0.5), myokit.Minus(
                myokit.Exp(x), myokit.Exp(myokit.PrefixMinus(x))))
        elif name == 'cosh':
            # Hyperbolic cosine: cosh(x) = 0.5 * (e^x + e^-x)
            x = parsex(dom_next(node))
            return myokit.Multiply(myokit.Number(0.5), myokit.Plus(
                myokit.Exp(x), myokit.Exp(myokit.PrefixMinus(x))))
        elif name == 'tanh':
            # Hyperbolic tangent: tanh(x) = (e^2x - 1) / (e^2x + 1)
            x = parsex(dom_next(node))
            e2x = myokit.Exp(myokit.Multiply(myokit.Number(2), x))
            return myokit.Divide(myokit.Minus(e2x, myokit.Number(1)),
                myokit.Plus(e2x, myokit.Number(1)))
        #
        # Inverse hyperbolic trigonometry (CellML...)
        #
        elif name == 'arcsinh':
            # Inverse hyperbolic sine: asinh(x) = log(x + sqrt(1 + x*x))
            x = parsex(dom_next(node))
            return myokit.Log(myokit.Plus(x, myokit.Sqrt(myokit.Plus(
                myokit.Number(1), myokit.Multiply(x, x)))))
        elif name == 'arccosh':
            # Inverse hyperbolic cosine:
            #   acosh(x) = log(x + sqrt(x + 1) * sqrt(x - 1))
            x = parsex(dom_next(node))
            return myokit.Log(myokit.Plus(x, myokit.Multiply(myokit.Sqrt(
                myokit.Plus(x, myokit.Number(1))), myokit.Sqrt(
                myokit.Minus(x, myokit.Number(1))))))
        elif name == 'arctanh':
            # Inverse hyperbolic tangent:
            #   atanh(x) = 0.5 * (log(1 + x) - log(1 - x))
            x = parsex(dom_next(node))
            return myokit.Multiply(myokit.Number(0.5), myokit.Minus(
                myokit.Log(myokit.Plus(myokit.Number(1), x)), myokit.Log(
                myokit.Minus(myokit.Number(1), x))))
        #
        # Hyperbolic redundant trigonometry (CellML...)
        #
        elif name == 'csch':
            # Hyperbolic cosecant: csch(x) = 2 / (exp(x) - exp(-x))
            x = parsex(dom_next(node))
            return myokit.Divide(myokit.Number(2), myokit.Minus(
                myokit.Exp(x), myokit.Exp(myokit.PrefixMinus(x))))
        elif name == 'sech':
            # Hyperbolic secant: sech(x) = 2 / (exp(x) + exp(-x))
            x = parsex(dom_next(node))
            return myokit.Divide(myokit.Number(2), myokit.Plus(
                myokit.Exp(x), myokit.Exp(myokit.PrefixMinus(x))))
        elif name == 'coth':
            # Hyperbolic cotangent:
            #   coth(x) = (exp(2*x) + 1) / (exp(2*x) - 1)
            x = parsex(dom_next(node))
            e2x = myokit.Exp(myokit.Multiply(myokit.Number(2), x))
            return myokit.Divide(myokit.Plus(e2x, myokit.Number(1)),
                myokit.Minus(e2x, myokit.Number(1)))
        #
        # Inverse hyperbolic redundant trigonometry (CellML has a lot to answer
        # for...)
        #
        elif name == 'arccsch':
            # Inverse hyperbolic cosecant:
            #   arccsch(x) = log(sqrt(1 + 1/x^2) + 1/x)
            xi = myokit.Divide(myokit.Number(1), parsex(dom_next(node)))
            return myokit.Log(myokit.Plus(myokit.Sqrt(myokit.Number(1),
                myokit.Power(xi, myokit.Number(2))), xi))
        elif name == 'arcsech':
            # Inverse hyperbolic secant:
            #   arcsech(x) = log(sqrt(1/x - 1) * sqrt(1/x + 1) + 1/x)
            xi = myokit.Divide(myokit.Number(1), parsex(dom_next(node)))
            return myokit.Log(myokit.Plus(myokit.Multiply(
                myokit.Sqrt(myokit.Minus(xi, myokit.Number(1))),
                myokit.Sqrt(myokit.Plus(xi, myokit.Number(1)))), xi))
        elif name == 'arccoth':
            # Inverse hyperbolic cotangent:
            #   arccoth(x) = 0.5 * (log(1 + 1/x) - log(1 - 1/x))
            xi = myokit.Divide(myokit.Number(1), parsex(dom_next(node)))
            return myokit.Multiply(myokit.Number(0.5), myokit.Minus(
                myokit.Log(myokit.Plus(myokit.Number(1), xi)),
                myokit.Log(myokit.Minus(myokit.Number(1), xi))))
        #
        # Logic
        #
        elif name == 'and':
            return chain(myokit.And, node)
        elif name == 'or':
            return chain(myokit.Or, node)
        elif name == 'not':
            return chain(None, node, myokit.Not)
        elif name == 'eq' or name == 'equivalent':
            n2 = dom_next(node)
            return myokit.Equal(parsex(n2), parsex(dom_next(n2)))
        elif name == 'neq':
            n2 = dom_next(node)
            return myokit.NotEqual(parsex(n2), parsex(dom_next(n2)))
        elif name == 'gt':
            n2 = dom_next(node)
            return myokit.More(parsex(n2), parsex(dom_next(n2)))
        elif name == 'lt':
            n2 = dom_next(node)
            return myokit.Less(parsex(n2), parsex(dom_next(n2)))
        elif name == 'geq':
            n2 = dom_next(node)
            return myokit.MoreEqual(parsex(n2), parsex(dom_next(n2)))
        elif name == 'leq':
            n2 = dom_next(node)
            return myokit.LessEqual(parsex(n2), parsex(dom_next(n2)))
        elif name == 'piecewise':
            # Piecewise contains at least one piece, optionally contains an
            #  "otherwise". Syntax doesn't ensure this statement makes sense.
            conds = []
            funcs = []
            other = None
            piece = dom_child(node)
            while piece:
                if piece.tagName == 'otherwise':
                    if other is None:
                        other = parsex(dom_child(piece))
                    elif logger:
                        logger.warn('Multiple <otherwise> tags found in'
                            ' <piecewise> statement.')
                elif piece.tagName == 'piece':
                    n2 = dom_child(piece)
                    funcs.append(parsex(n2))
                    conds.append(parsex(dom_next(n2)))
                elif logger:
                    logger.warn('Unexpected tag type in <piecewise>: '
                        + '<' + piece.tagName + '>.')
                piece = dom_next(piece)
            if other is None:
                other = myokit.Number(0)
            # Create string of if statements
            args = []
            f = iter(funcs)
            for c in conds:
                args.append(c)
                args.append(f.next())
            args.append(other)
            return myokit.Piecewise(*args)
        #
        # Constants
        #
        elif name == 'pi':
            return myokit.Number('3.14159265358979323846')
        elif name == 'exponentiale':
            return myokit.Exp(myokit.Number(1))
        elif name == 'true':
            # This is corrent, even in Python True == 1 but not True == 2
            return myokit.Number(1)
        elif name == 'false':
            return myokit.Number(0)
        #
        # Unknown/unhandled elements
        #
        else:
            if logger:
                logger.warn('Unknown element: ' + name)
            ops = []
            node = dom_child(node) if dom_child(node) else dom_next(node)
            while node:
                ops.append(parsex(node))
                node = dom_next(node)
            return myokit.UnsupportedFunction(name, ops)
    # Remove math node, if given
    if node.tagName == 'math':
        node = dom_child(node)
    #TODO: Check xmlns?
    return parsex(node)
def parse_mathml_number(node, logger=None):
    """
    Parses a mathml <cn> tag to a :class:`myokit.Number`.

    The attribute ``node`` must be a DOM node representing a <cn> tag using the
    ``xml.dom.Node`` interface.

    If the argument ``logger`` is given this will be used to log messages to,
    assuming the :class:`myokit.TextLogger` interface.
    """
    kind = node.getAttribute('type')
    if kind == '':
        # Default type
        kind = 'real'
    if kind == 'real':
        # Float, specified as 123.123 (no exponent!)
        # May be in a different base than 10
        base = node.getAttribute('base')
        if base:
            raise MathMLError('BASE conversion for reals is not supported')
        return myokit.Number(str(node.firstChild.data).strip())
    elif kind == 'integer':
        # Integer in any given base
        base = node.getAttribute('base')
        numb = str(node.firstChild.data).strip()
        if base:
            v = int(numb, base)
            if logger:
                logger.log('Converted from '+str(numb) + ' to ' + str(v))
            numb = v
        return myokit.Number(numb)
    elif kind == 'double':
        # Floating point (positive, negative, exponents, etc)
        return myokit.Number(numb)
    elif kind == 'e-notation':
        # 1<sep />3 = 1e3
        sig = str(node.firstChild.data.strip())
        exp = str(node.firstChild.nextSibling.nextSibling.data).strip()
        numb = sig + 'e' + exp
        if logger:
            logger.log('Converted '+ sig + 'e' + str(exp) + '.')
        return myokit.Number(numb)
    elif kind == 'rational':
        # 1<sep />3 = 1 / 3
        num = str(node.firstChild.data.strip())
        den = str(node.firstChild.nextSibling.nextSibling.data).strip()
        numb = str(float(num)/float(den))
        if logger:
            logger.log('Converted ' + num + ' / ' + den + ' to ' + numb)
        return myokit.Number(numb)
    else:
        raise MathMLError('Unsupported <cn> type: ' + kind)
