#
# Contains a number of polynomial approximation algorithms.
#
# Some methods require a recent version of scipy to be installed.
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
import os
import myokit
import numpy as np
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Part I.
# Function handle creators and their myokit equivalents
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class FittedFunction(object):
    """
    Represents a function that was fitted to some data or another function.
    
    The used fitting procedure may have resulted in an absolute error
    ``abserr`` and a relative error ``relerr``. If so, this can be indicated by
    setting these properties to values other than ``None``.
    """
    def __init__(self, abserr=None, relerr=None):
        self._abserr = float(abserr) if abserr is not None else None
        self._relerr = float(relerr) if relerr is not None else None
    def abserr(self):
        """
        Returns the maximum absolute error found when fitting this function.
        """
        return self._abserr
    def relerr(self):
        """
        Returns the maximum relative error found when fitting this function.
        """
        return self._relerr
    def set_abserr(self, abserr):
        """
        Sets the maximum absolute error found when fitting this function.
        """
        self._abserr = float(abserr) if abserr is not None else None
    def set_relerr(self, relerr):
        """
        Sets the maximum relative error found when fitting this function.
        """
        self._relerr = float(relerr) if relerr is not None else None
class Polynomial(FittedFunction):
    """
    Represents a polynomial function.
    
    A polynomial is created from a list of coefficients ``c`` and is defined
    as::
    
        p(x) = c[0] + c[1]*x + c[2]*x**2 + ... + c[n]*x**n
        
    or, equivalently::
    
        P(x) = c[0] + x * (c[1] + x * (c[2] + ... + x * (c[-2] + x * c[-1])))
    
    A polynomial can be evaluated:
    
        >>> from myokit.lib.approx import Polynomial
        >>> f = Polynomial([1,2,3])
        >>> print(f(0), f(1), f(-2))
        (1, 6, 9)
        
    Polynomials work with numpy input:        
        
        >>> import numpy as np
        >>> x = np.array([0, 1, -2])
        >>> print(f(x))
        [1 6 9]
        
    A Polynomial is iterable, which provides (read-only) access to its
    coefficients:
    
        >>> print(', '.join(str(x) for x in f))
        1.0, 2.0, 3.0
        
    """
    def __init__(self, c):
        super(Polynomial, self).__init__()
        self._c = [float(x) for x in c]
        n = len(c)
        def f(x):
            v = c[-1]
            for i in xrange(n-2, -1, -1):
                v = v * x + c[i]
            return v
        self._func = f
    def __call__(self, x):
        return self._func(x)
    def __iter__(self):
        return iter(self._c)
    def __len__(self):
        return len(self._c)
    def __getitem__(self, key):
        return self._c.__getitem__(key)
    def myokit_form(self, lhs):
        """
        Returns this piecewise polynomial as a myokit expression with the
        indepent variable expressed by `lhs`.
        """
        return myokit.Polynomial(lhs, *[myokit.Number(c) for c in self._c])
    def __str__(self):
        return 'Polynomial([' \
            + ', '.join([str(x) for x in self._c]) + '])'
class PiecewisePolynomial(FittedFunction):
    """
    Represents a piecewise polynomial function.
    
    A piecewise polynomial is created from a list of ``n`` :class:`polynomials
    <Polynomial>` and a sorted list of ``n - 1`` splitting points.
    
    The resulting spline is calculated as::

                  x < k[0]   --> s(x) = p[0]
        k[0]   <= x < k[1]   --> s(x) = p[1]
                  :                   :
        k[n-3] <= x < k[n-2] --> s(x) = p[n-2] 
        k[n-2] <= x          --> s(x) = p[n-1]
        
    where ``p`` is the list of polynomial functions and ``k`` is the list of
    splitting points.
    
    A piecewise polynomial can be evaluated:
    
        >>> from myokit.lib.approx import Polynomial, PiecewisePolynomial
        >>> f = PiecewisePolynomial(
        ...     [Polynomial((1,2,3)),
        ...      Polynomial((2,3,4)),
        ...      Polynomial((3,4,5))],
        ...      [1.5, 2.5])
        >>> print(f(0), f(1), f(2), f(3), f(4))
        (1, 6, 24, 60, 99)
        
    Piecewise polynomials work with numpy input:        
        
        >>> import numpy as np
        >>> x = np.array([0, 1, 2, 3, 4])
        >>> print(f(x))
        [ 1  6 24 60 99]
        
    Instead of passing in Polynomial objects, a list of coefficient-lists can
    also be given.    
    """
    def __init__(self, p, k=None):
        super(PiecewisePolynomial, self).__init__()
        if k is None:
            # Clone
            self.set_relerr(p.relerr())
            self.set_abserr(p.abserr())
            k = p._k
            p = p._p
        # New object, check input
        n = len(p)
        if len(k) + 1 != n:
            raise ValueError('The number of polynomial pieces given must be'
                ' one more than the number of splitting points.')
        # Copy input and store
        self._k = k = [float(x) for x in k]
        self._c = c = [[float(x) for x in y] for y in p]        
        self._p = p = [Polynomial(x) for x in p]
        # Evaluating function
        if n == 1:
            self._f = p[0]
        else:
            def f(x):
                cond = [0] * n
                cond[0] = x < k[0]
                for i in xrange(0, n-2):
                    cond[i+1] = (x >= k[i]) * (x < k[i+1])
                cond[-1] = x >= k[-1]
                return np.piecewise(x, cond, self._p)
            self._f = f
    def __call__(self, x):
        try:
            x[0]
        except TypeError:
            # Not a vector type
            return self._f(np.array([x]))[0]
        return self._f(x)
    def _myokit_form(self, args):
        """
        Creates the appropriate myokit form for this class + subclasses.
        """
        return myokit.OrderedPiecewise(*args)
    def myokit_form(self, lhs):
        """
        Returns this piecewise polynomial as a myokit expression with the
        indepent variable expressed by `lhs`.
        """
        c = self._c
        p = self._p
        n = len(c) - 1
        if n == 0:
            return Polynomial(c[0]).myokit_form(lhs)
        args = [lhs.clone()]
        for i in xrange(0, n):
            args.append(p[i].myokit_form(lhs))
            args.append(myokit.Number(self._k[i]))
        args.append(p[n].myokit_form(lhs))
        return self._myokit_form(args)
class Spline(PiecewisePolynomial):
    """
    *Extends:* :class:`PiecewisePolynomial`
    
    A spline is created from a list of ``n`` :class:`polynomials<Polynomial>`
    and a sorted list of ``n + 1`` points, including the start
    and end point of the domain the spline is defined on.
    
    Outside the domain boundaries, the values of the 
    
    The resulting spline is calculated as::

                  x < k[0]   --> s(x) = p[1]
        k[0]   <= x < k[1]   --> s(x) = p[2]
                  :                   :
        k[n-3] <= x < k[n-2] --> s(x) = p[n-2] 
        k[n-2] <= x          --> s(x) = p[n-1]
    
    The spline's knots can be retrieved using :meth:`knots()`.
    
    Using the :class:`Spline` class instead of :class:`PiecewisePolynomial`
    indicates that the function is first-order differentiable at the knots, 
    although this is not checked internally.
    
    Splines can be cloned using the syntax ``s_clone = Spline(s)``.
    """
    def __init__(self, p, k=None):
        if k is None:
            clone = p
            p = clone.pieces()
            k = clone.knots()
        else:
            clone = None
            if len(p) + 1 != len(k):
                raise ValueError('The number of polynomial pieces given must'
                    ' be one less than the number of knots.')
        super(Spline, self).__init__(p, k[1:-1])
        self._knots = [float(x) for x in k]
        if clone:
            self.set_abserr(clone.abserr())
            self.set_relerr(clone.relerr())
    def _myokit_form(self, args):
        return myokit.Spline(*args)
    def knots(self):
        """
        Returns this piecewise function's knots.
        """
        return np.array(self._knots)
    def number_of_pieces(self):
        """
        Returns the number of pieces in this spline.
        """
        return len(self._p)
    def pieces(self):
        """
        Returns the individual polynomial pieces this spline is composed of.
        """
        return list(self._p)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Part II
# Functions that fit a polynomial to a function over an interval
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class FittingError(Exception):
    """
    Error raised when a fitting procedure fails.
    """
    pass
def fit_lagrange_polynomial(f, i, n=None, x=None):
    """
    Calculates the Lagrange polynomial of degree ``n`` to the function ``f``
    on the interval ``i=[a,b]``. The result is returned as a
    :class:`Polynomial`.

    Instead of (or in addition to) giving the degree ``n``, a list of ``n+1``
    points on the interval ``i`` can be passed in. In these points, the
    approximation will match the function exactly.
    """
    # Check interval
    try:
        if len(i) != 2:
            raise ValueError('The given interval must have length 2')
    except TypeError:
        raise ValueError('The given interval must contain two values.')
    a,b = i
    if a >= b:
        raise ValueError('The first point in the interval must be smaller than'
            ' the second.')
    # Check function
    try:
        f(a)
    except TypeError:
        raise ValueError('The given function must have a single positional'
            ' argument.')
    # Check order and nodes
    if x is None:
        if n is None:
            raise ValueError('Either n or x must always be set.')
        # Create initial nodes
        x = np.arange(n, -1, -1)
        x = np.cos(np.pi * (2*x + 1) / (2*(n+1)))
        x = a + 0.5 * (x + 1) * (b - a)
    else:
        try:
            m = len(x)
        except TypeError:
            raise ValueError('The given value for x must be a sequence type'
                ' containing n+1 points on the interval i.')
        if n is None:
            n = m - 1
        elif n != m - 1:
            raise ValueError('The given value for x must contain n+1 points on'
                ' the interval i.')
    # Check if n is sensible :)
    if n < 1:
        raise ValueError('The polynomial must have at least order 1.')
    # Calculate lagrange polynomial coefficients and return
    c = np.zeros(n+1)
    for k in xrange(0, n + 1):
        p = np.array(list(x[0:k]) + list(x[k+1:n+1]))
        q = np.zeros(n+1)
        q[-1] = 1
        q[-2] -= p[0]
        r = f(x[k]) / (x[k] - p[0])
        for i in xrange(1, n):
            for j in xrange(i, 0, -1):
                q[-j-2] -= p[i] * q[-j-1]
            q[-2] -= p[i]
            r /= x[k] - p[i]
        c += q * r
    return Polynomial(c)
def fit_remez_polynomial(f, i, n, stop_tol=1e-6, max_iter=100):
    """
    Calculates an approximating polynomial of degree ``n`` to the function
    ``f`` on the interval ``i=[a,b]``. The result is returned as a 
    :class:`Polynomial`.

    This function uses the Remez exchange algorithm to iteratively calculate a
    polynomial that minimises the supremum of ``|f - g|`` over the interval
    ``[a, b]``. The function will stop when the difference between succesive
    errors is less than ``stop_tol`` or ``max_iter`` is reached.
    """
    from numpy.linalg import solve
    from scipy.optimize import brentq, fmin_tnc
    # Test order
    if n < 1:
        raise ValueError('The polynomial degree must be 1 or higher.')
    if int(n) != n:
        raise ValueError('The polynomial degree must be an integer.')
    # Test interval input
    try:
        if len(i) != 2:
            raise ValueError('Interval must have length 2.')
    except TypeError:
        raise ValueError('The second argument to remez() must be an interval'
            ' specified as a sequence type (a,b).')
    a, b = i
    if a >= b:
        raise ValueError('The first point in the interval must be smaller than'
            ' the second.')
    # Test function input
    try:
        test = f(a)
    except TypeError:
        raise ValueError('The first argument to remez() must be a callable'
            ' function with a single positional argument.')
    # Constrained minimization
    def fmin(e, x, a, b):
        return fmin_tnc(e,(x,),bounds=((a,b),),messages=0,approx_grad=True)[0]
    # Get initial nodes (Chebyshev)
    x = np.arange(n+1, -1, -1)
    x = np.cos(np.pi * (2*x + 1) / (2*(n+2)))
    x = a + 0.5 * (x + 1) * (b - a)
    # Pre-allocate some variables
    A = np.ones((n+2, n+2))
    X = np.arange(a, b, 101)
    F = f(X)
    z = np.zeros(n+1)
    # Iterate
    e_last = None
    for k in xrange(0, max_iter):
        # Build linear system and solve to find coefficients
        for i in xrange(0, n+2):
            A[i, 0] = -1 if i % 2 else 1
            A[i, 2] = x[i]
            for j in xrange(1, n):
                A[i, 2+j] = A[i, 1+j] * x[i]
        B = f(x)
        C = solve(A, B)
        c = C[1:,]
        # Calculate maximum absolute deviation
        g = Polynomial(c)
        E = lambda t : -abs(g(t) - f(t))
        e = -E(fmin(E, 0.5*(b-a), a, b))[0]
        e2 = np.max(abs(g(X) - F))
        if e2 / e > 5.0:
            e = e2  # Just in case fmin messes up :(
        g.set_relerr(e)
        # Return if e - e_last is below the threshold
        if e_last is not None:
            if abs(e - e_last) < stop_tol:
                break
        e_last = e
        # Return if max iterations reached (no need to calculate unused roots)
        if k + 1 == max_iter:
            break
        # Find roots of error function in intervals between points
        E = lambda t : g(t) - f(t)
        for i in xrange(0, n+1):
            z[i] = brentq(E, x[i], x[i+1])
        # Find points of maximum deviation between these roots, use as new x
        E2 = lambda t : -(g(t) - f(t))*(g(t) - f(t))
        x[0] = fmin(E2, x[0], a, z[0])
        x[n+1] = fmin(E2, x[n+1], z[n], b)
        for i in xrange(1, n+1):
            x[i] = fmin(E2, x[i], z[i-1], z[i])
    # Return result
    return g
def solve_cubic_spline(f, p):
    """
    Finds the coefficients for a cubic spline matching function ``f`` on the
    knots given by ``p`` and returns the resulting :class:`Spline`.

    The list of knots must include the left and right-most points of the
    spline, so that the (number of knots) = 1 + (number of pieces).

    The argument ``f`` must be the handle to a function ``f(x)``. The array
    ``p`` must contain ``n+1`` increasing x-values (the "knots"). The
    approximating spline ``g`` matches ``f`` exactly at every knot. At the
    outermost knots the first derivative of ``g`` matches ``f``'s first
    derivative. At all inner knots, the first derivatives of the two touching
    pieces will be equal.
    """
    n = len(p)
    # Get derivative at a and b, assuming the function exists in between
    h = (p[-1] - p[0]) / 1e9
    f1a = (f(p[0] + h) - f(p[0])) / h
    f1b = (f(p[-1]) - f(p[-1] - h)) / h
    # Create matrix to find coefficients of splines
    h = p[1:] - p[:-1]
    A = np.zeros((n, n))
    A[0,0] = 2 * h[0]
    A[0,1] = h[0]
    A[-1,-1] = 2 * h[-1]
    A[-1,-2] = h[-1]
    for i in xrange(1, n-1):
        A[i, i-1:i+2] = h[i-1], 2*(h[i-1]+h[i]), h[i]
    F = f(p)
    B = np.zeros(n)
    B[0]  =  3/h[0]  * (F[1]  - F[0] ) - 3 * f1a
    B[-1] = -3/h[-1] * (F[-1] - F[-2]) + 3 * f1b
    for i in xrange(1, n-1):
        B[i] = 3./h[i] * (F[i+1] - F[i]) - 3./h[i-1] * (F[i] - F[i-1])
    Z = np.linalg.solve(A, B)
    # Now, create matrix with the polynomial coeffs of the splines
    del(A, B)
    C = np.zeros((n-1,4))
    C[0:, 0] = F[0:-1]
    C[0:, 2] = Z[0:-1]
    for i in xrange(n-2, -1, -1):
        C[i, 1] = (F[i+1] - F[i]) / h[i] - (h[i]/3)*(2*Z[i]+Z[i+1])
        C[i, 3] = (Z[i+1]-Z[i]) / (3 * h[i])
    # Rework the coefficients so that "x" can be used instead of "(x-p)"
    for i in xrange(0, n-1):
        p1 = p[i]
        p2 = p1 * p1
        C[i,0] -= C[i,1]*p1 - C[i,2]*p2 + C[i,3]*p2*p1
        C[i,1] -= 2.*C[i,2]*p1 - 3.*C[i,3]*p2
        C[i,2] -= 3.*C[i,3]*p1
    # Create and return a spline
    return Spline(C, p)
def fit_cubic_spline(f, i, reltol=1e-3, abstol=None, min_pieces=2,
    max_pieces=50, regular_spacing=False):
    """
    Creates a cubic spline approximation to ``f`` on the given interval
    ``i=[a,b]``.

    The number of pieces is started at ``min_pieces`` and increased until the
    specified absolute tolerance ``abstol`` and relative tolerance ``reltol``
    are met. If the tolerance can't be met with at most ``max_pieces``, then
    ``None`` is returned.

    The result is returned as a :class:`Spline`.
    
    The routine starts with regularly spaced knots, and then adds a new knot at
    the position with the highest error. To disable this behaviour, and use
    regularly spaced intervals throughout the routine, set ``regular_spacing``
    to ``True``.
    """
    # Check interval
    try:
        if len(i) != 2:
            raise ValueError('The given interval must have length 2')
    except TypeError:
        raise ValueError('The given interval must contain two values.')
    lo, up = i
    lo, up = float(lo), float(up)
    if lo >= up:
        raise ValueError('The first point in the interval must be smaller than'
            ' the second.')
    # Check function
    try:
        f(lo)
    except TypeError:
        raise ValueError('The given function must have a single positional'
            ' argument.')
    # Estimate range of function
    X = np.linspace(lo, up, 1000)
    F = f(X)
    H = np.max(F) - np.min(F)
    # Attempt fit
    k = np.linspace(lo, up, 1 + min_pieces)    # Because #knots = #pieces + 1
    for i in xrange(min_pieces, 1 + max_pieces):
        g = solve_cubic_spline(f, k)
        # Get max deviation, position of max deviation
        a = np.abs(F - g(X))
        r = np.argmax(a)
        q = X[r]
        a = a[r]
        r = a / H
        # Return?
        if abstol is None or a < abstol:
            if reltol is None or r < reltol:
                g.set_abserr(a)
                g.set_relerr(r)
                return g
        # Continue, add knot
        if regular_spacing:
            k = np.linspace(lo, up, 2 + i)
        else:
            k = np.sort(np.append(k, q))
    raise FittingError('Reached maximum number of pieces.')
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Part III.
# Methods to approximate univariate functions in models
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def suitable_expressions(model, var, blacklist=None, require_exp=False,
    exclude_conditionals=False):
    """
    Extracts the RHS expressions from `model` suitable for approximation.
    "Suitable" functions are single-variabled functions dependent on variable
    `var`.
    
    Arguments:
    
    ``model``
        The model to search in
    ``var``
        The variable all expressions must be a univariate function of
    ``blacklist=None``
        A list of variables (or variable qnames) whose RHS to exclude.
    ``require_exp=False``
        With this set to ``True``, only expressions containing an ``exp()``
        function are considered.
    ``exclude_conditionals=False``
        With this set to ``True``, any functions containing a conditional
        statement will be excluded.
    
    The returned value is a list of tuples ``(lhs, rhs)``, where ``rhs`` is the
    expressions defining ``lhs``, and ``rhs`` is deemed "suitable".
    """
    # Require valid model
    model.validate()
    # Get left-hand side expression for variable
    if isinstance(var, myokit.LhsExpression):
        if isinstance(var, myokit.Derivative):
            raise ValueError('Cannot fit with respect to a state.')
        lhs = var
        var = lhs.var()
        # Test if variable is from correct model
        if var.model() != model:
            raise ValueError('The given variable must be a variable in the'
                ' given model.')
    else:
        if type(var) in [str, unicode]:
            lhs = myokit.Name(model.get(var))
        elif isinstance(var, myokit.Variable):
            lhs = myokit.Name(var)
        else:
            raise ValueError('The argument "var" must be either an'
                ' LhsExpression, a Variable or a string referencing a'
                ' model variable')
    # Test suitability of variable
    if lhs.is_constant():
        raise ValueError('The given variable can not be constant.')
    # Parse blacklist, convert to list of LhsExpression objects
    if blacklist:
        bl = set()
        for x in blacklist:
            if isinstance(x, myokit.LhsExpression):
                bl.add(x)
            elif isinstance(x, myokit.Variable):  
                bl.add(x.lhs())
            else:
                x = model.get(x)
                bl.add(x.lhs())
        blacklist = bl
    else:
        blacklist = set()
    # Iterate over variables, attempt to find suitable functions
    functions = []
    for v, deps in model.map_deep_dependencies(omit_states=False,
            filter_encompassed=True).iteritems():
        # Exclude blacklisted lhs expressions
        if v in blacklist:
            continue
        # Exclude the function defining the selected variable
        if v.var() == var:
            continue
        # Exclude constants
        if v.is_constant():
            continue
        # Exclude variables not dependent on the selected variable
        if lhs not in deps:
            continue
        # Exclude functions depending on other variables (except constants)
        if len(deps) > 1:
            can_fit = True
            for d in deps:
                if not d.is_constant() and d != lhs:
                    can_fit = False
                    break
            if not can_fit:
                continue
        # Exclude conditional functions
        rhs = v.rhs()
        if rhs.is_conditional():
            continue
        # Exclude functions without exp()
        if require_exp and not rhs.contains_type(myokit.Exp):
            continue
        # Variable is ok!
        functions.append((v, rhs))
    return functions
class Fitter(myokit.TextLogger):
    """
    Attempts to update a myokit model by approximating functions of a single
    variable with a known range (typically the membrane potential).
    """
    def __init__(self):
        super(Fitter, self).__init__()
        self._changed = None
    def fit(self, model, var, rng, report=None):
        """
        Attempts to approximate suitable functions in the given model.

        A variable ``var`` must be specified, as well as the range ``rng`` that
        this variable is likely to take. A typical example would be the
        variable ``membrane.V`` and the range ``(-100, 50)``.

        The variable can be specified as a name (string), an object
        representing a variable or an LhsExpression.

        The range can be specified as any sequence type of length 2.

        If the variable ``report`` is set, a small HTML based report will be
        generated and stored in the directory indicated by ``report``.
        """
        # Copy model
        self.original_model = model
        model = model.clone()
        # Get left-hand side expression for variable
        if isinstance(var, myokit.LhsExpression):
            if isinstance(var, myokit.Derivative):
                raise ValueError('Cannot fit with respect to a state.')
            # Get equivalent variable from cloned model
            var = model.get(var.var().qname())
            lhs = myokit.Name(var)
        else:
            if type(var) in [str, unicode]:
                lhs = myokit.Name(model.get(var))
            elif isinstance(var, myokit.Variable):
                # Get equivalent variable from cloned model
                var = model.get(var.var().qname())
                lhs = myokit.Name(var)
            else:
                raise ValueError('The argument "var" must be either an'
                    ' LhsExpression, a Variable or a string referencing a'
                    ' model variable')
        # Test suitability of variable
        if lhs.is_constant():
            raise ValueError('The given variable can not be constant.')
        # Test suitability of range
        if len(rng) < 2:
            raise ValueError('The argument "rng" must contain the lower and'
                ' upper boundaries of the variable\'s name.')
        if rng[0] == rng[1]:
            raise ValueError('The given range must contain two distinct'
                ' values.')
        if rng[0] > rng[1]:
            rng = rng.reverse()
        # Reset list of changed variables
        self._changed = []
        # Go!
        return self._fit(model, lhs, rng, report)
    def changed_variables(self):
        """
        Returns a list containing the names of the variables that were updated
        during the last run.
        """
        return list(self._changed)
class PolynomialFitter(Fitter):
    """
    Attempts to update a myokit model by approximating functions of a single
    variable with a known range (typically the membrane potential) with a
    single polynomial
    """
    name = 'polynomial'
    def __init__(self):
        super(PolynomialFitter, self).__init__(reltol=5e-3, max_order=5)
        self._max_order = 5
        self._reltol = float(reltol)
    def _fit(self, model, lhs, rng, report):
        var = lhs.var()
        # Create set of points on range to be used to calculate max deviation
        xrng = np.linspace(rng[0], rng[1], 1000)
        # Build report
        if report:
            self.log('Loading modules to build report')
            from myokit.mxml import TinyHtmlPage
            import matplotlib.pyplot as pl
            # Create report directories
            report = os.path.abspath(report)
            self.log('Saving report to ' + str(report))
            if not os.path.isdir(report):
                self.log('Creating directory...')
                os.makedirs(report)
            # Create html page
            page = TinyHtmlPage()
            title = 'Scanning ' + model.name()
            page.set_title(title)
            page.append('h1').text(title)
            page.append('h2').text('Method: ' + self.name.upper())
            div = page.append('div')
            div.append('p').text('Attempting to approximate expressions with a'
                ' single polynomial of at most degree '
                + str(self._max_order) + '.')
            div.append('p').text('Accepting approximations if the (estimated)'
                ' maximum error divided by the (estimated) range is less than '
                + str(self._reltol))
        # Replacement functions will be stored here
        fits = {}
        # Extract suitable lhs,rhs pairs
        suitable = suitable_expressions(model, lhs)
        # Attempt to approximate them
        count = 0
        for lhs, rhs in suitable:
            count += 1
            # Create python function for evaluations
            f = lhs.var().pyfunc()
            # Indicate interesting variable has been found
            name = lhs.var().qname()
            self.log('Selected: ' + name + ' = ' + str(rhs))
            # Get function f on interval, estimate range
            F = f(xrng)
            H = np.max(F) - np.min(F)
            # Attempt fits of increasing order
            fit = g = deviation = None
            try:
                for n in xrange(2, 1 + self._max_order):
                    c, g, e = fit_remez_polynomial(f, rng, n)
                    deviation = e / H
                    if deviation < self._reltol:
                        self.log('Found good approximation of order ' + str(n))
                        fit = g.myokit_form(lhs)
                        self.log('App: ' + fit.code())
                        break
            except FittingError as e:
                self.log('FittingError: ' + str(e))
            except Exception as e:
                import traceback
                self.log('Exception ' + '* '*29)
                self.log('')
                self.log(traceback.format_exc())
                self.log('')
                self.log('* '*34)
            if report:
                div = page.append('div')
                div.set_attr('style', 'page-break-before: always;')
                div.append('h2').text(str(count) + '. ' + name)
                div.append('p').math(rhs)
                if g:
                    # Create figure
                    pl.figure()
                    ax = pl.subplot(111)
                    xplot = np.linspace(rng[0], rng[1], 1000)
                    pl.plot(xplot, f(xplot), label='f(x)', lw=3)
                    pl.plot(xplot, g(xplot), label='L_'+str(n))
                if fit:
                    div.append('h3').text('Found approximation of order '
                        + str(n))
                    div.append('p').math(fit)
                    div.append('p').text('(Maximum deviation / range) is'
                        ' approximately ' + str(deviation))
                else:
                    div.append('h3').text('No approximation found')
                    div.append('p').text('Approximation of order ' + str(n)
                        + ' failed with an estimated maximum relative'
                        ' deviation of ' + str(deviation) + '.')
                if g:
                    # Add plot to report
                    box = ax.get_position()
                    ax.set_position([box.x0,box.y0, box.width*0.8, box.height])
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    iname = os.path.join(report, name + '.png')
                    self.log('Saving plot to ' + iname)
                    pl.savefig(iname)
                    pl.close()
                    div.append('h3').text('Graphical representation')
                    div.append('img').set_attr('src', iname)
            # Save change to dict
            if fit:
                self.log('Saving approximation...')
                fits[lhs.var()] = fit
            else:
                self.log('No suitable approximation found.')
        # Create updated model
        for var, fit in fits.iteritems():
            var.set_rhs(fit)
            var.meta['approximate'] = 'polynomial'
            self._changed.append(var.qname())
        # Validate model, enable crude fixes!
        model.validate(fix_crudely = True)
        # Write report(s) to file
        if report:
            fname = os.path.join(report, 'index.html')
            self.log('Writing report to ' + fname)
            with open(fname, 'w') as f:
                f.write(page.html(pretty=False))
        # Return updated model
        self.log('Returning model...')
        return model
class CubicSplineFitter(Fitter):
    """
    Attempts to approximating functions of a single variable with a known range
    (typically the membrane potential) using a cubic spline.
    
    Arguments:
    
    ``abstol``
        The absolute tolerance to fit to.
    ``reltol``
        The relative tolerance to fit to.
    ``min_pieces``
        The minimum number of pieces in the spline.
    ``max_pieces``
        The maximum number of pieces in the spline.
    ``blacklist``
        A list of variables not to approximate.
    
    """
    name = 'cubic_spline'
    def __init__(self, abstol=None, reltol=0.001, min_pieces=1, max_pieces=200,
            blacklist=None):
        super(CubicSplineFitter, self).__init__()
        self._min_pieces = int(min_pieces)
        self._max_pieces = int(max_pieces)
        self._abstol = abstol
        self._reltol = reltol
        self._blacklist = tuple(blacklist) if blacklist else ()
    def _fit(self, model, lhs, rng, report):
        var = lhs.var()
        # Build report
        if report:
            self.log('Loading modules to build report')
            from myokit.mxml import TinyHtmlPage
            import matplotlib.pyplot as pl
            # Create report directories
            report = os.path.abspath(report)
            self.log('Saving report to ' + str(report))
            if not os.path.isdir(report):
                self.log('Creating directory...')
                os.makedirs(report)
            # Create html page
            page = TinyHtmlPage()
            title = 'Scanning ' + model.name()
            page.set_title(title)
            page.append('h1').text(title)
            page.append('h2').text('Method: ' + self.name.upper())
            div = page.append('div')
            div.append('p').text('Attempting to approximate equations using a'
                ' piecewise cubic spline containing at most '
                + str(self._max_pieces) + ' pieces.')
            div.append('p').text('Accepting approximations if the (estimated)'
                ' maximum error divided by the (estimated) range is less than '
                + str(self._reltol) + '.')
            # X-values for plotting
            xplot = np.linspace(rng[0], rng[1], 1000)
        # Replacement functions will be stored here
        fits = {}
        # Find suitable lhs,rhs pairs
        suitable = suitable_expressions(model, lhs,
            blacklist=self._blacklist)
        count = 0
        for v, rhs in suitable:
            count += 1
            name = v.var().qname()
            self.log('Selected: ' + name + ' = ' + str(rhs))
            # Attempt fits of increasing order
            fit = c = p = g = a = r = None
            f = v.var().pyfunc()
            try:
                g = fit_cubic_spline(f, rng,
                    min_pieces=self._min_pieces, max_pieces=self._max_pieces,
                    abstol=self._abstol, reltol=self._reltol)
                n = len(g.pieces())
                if g.relerr() < self._reltol:
                    self.log('Found approximation with ' + str(n) + ' pieces.')
                    fit = g.myokit_form(lhs)
            except FittingError as e:
                self.log('FittingError: ' + str(e))
            except Exception as e:
                import traceback
                self.log('Exception ' + '* '*29)
                self.log('')
                self.log(traceback.format_exc())
                self.log('')
                self.log('* '*34)
            if report:
                div = page.append('div')
                div.set_attr('style', 'page-break-before: always;')
                div.append('h2').text(str(count) + '. ' + name)
                div.append('p').math(rhs)
                if g:
                    # Create figure                
                    pl.figure()
                    ax = pl.subplot(111)
                    pl.plot(xplot, g(xplot), label='L_'+str(n))
                    # Show nodes
                    knots = g.knots()
                    pl.plot(knots, f(knots), 'x', label='nodes',markersize=8)
                # Write report
                if fit:
                    div.append('h3').text('Accepted approximation using '
                         + str(n) + ' pieces.')
                else:
                    div.append('h3').text('No acceptable approximation found.')
                if g:
                    div.append('p').text('(Maximum deviation / range) is'
                        ' approximately ' + str(g.relerr()))
                    # Add plot to report
                    box = ax.get_position()
                    ax.set_position([box.x0,box.y0, box.width*0.8, box.height])
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    iname = os.path.join(report, name + '.png')
                    self.log('Saving plot to ' + iname)
                    pl.savefig(iname)
                    pl.close()
                    div.append('h3').text('Graphical representation')
                    div.append('img').set_attr('src', iname)
            # Save change to dict
            if fit:
                self.log('Saving approximation...')
                fits[v.var()] = fit
            else:
                self.log('No suitable approximation found.')
        # Create updated model
        for var, fit in fits.iteritems():
            var.set_rhs(fit)
            var.meta['approximate'] = 'cubic_spline'
            self._changed.append(var.qname())
        # Validate model, enable crude fixes!
        model.validate(fix_crudely=True)
        # Write report(s) to file
        if report:
            fname = os.path.join(report, 'index.html')
            self.log('Writing report to ' + fname)
            with open(fname, 'w') as f:
                f.write(page.html(pretty=False))
        # Return updated model
        self.log('Returning model...')
        return model
# Create dict (name:class)
FITTERS = [
    PolynomialFitter,
    CubicSplineFitter,
    ]
FITTERS = dict(zip([x.name for x in FITTERS], FITTERS))
del(x) # Prevent x being exposed as part of this module.
def list_fitters():
    """
    Returns a dict mapping fitter names to classes.
    """
    return dict(FITTERS)
