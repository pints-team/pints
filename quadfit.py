def quadfit(x, y):
    """
    Calculates the unique quadratic polynomial through a set of points.

    The argument ``y`` must be a sequence of ``m`` scalars, while ``x`` should
    contain ``m`` points of dimension ``n > 0``.

    The function calculates ``A``, ``B`` and ``C`` such that

        y[k] = A + B' * x[k] + (1/2) * x[k]' * C * x[k]

    for ``k = 1,2,...,m``. Here ``A`` is a scalar, ``B`` is a column vector of
    size ``n`` and ``C`` is a symmetric ``n`` by ``n`` matrix.

    To get a solvable system, the number of points ``m`` must equal the number
    of unknowns in ``A``, ``B`` and ``C``, such that::

        m = 1 + n + sum(1,2,...,n)
          = 1 + n + n * (n + 1) / 2
          = (n / 2 + 1) * (n + 1)
          = (n + 1) * (n + 2) / 2

    For example, for the simplest case where ``n = 1`` we get ``m = 3``. For
    quadratic polynomials on a two-dimensional space we get ``n = 2`` so we
    need ``m = 6`` data points.

    Arguments:

    ``x``
        A sequence of ``m`` points, each of the same dimension ``n``. Using
        numpy, this can also be given as an ``m`` by ``n`` matrix.
    ``y``
        A sequence of ``m`` scalars.

    Output is a tuple ``(A, B, C)``:

    ``A``
        A scalar.
    ``B``
        A vector of shape ``(n, )``
    ``C``
        A symmetrical matrix of shape ``(n, n)``.

    Example 1::

        def f(x):
            a = [7, -3, 2]
            return a[0] + a[1] * x + a[2] * x**2
        x = [-2, 1, 6]
        y = [f(i) for i in x]
        A, B, C = quadfit(x, y)    # Returns 7, [-3] and [[2]]

    Example 2::

        def f(x, y):
            a = 5, 4, 3, 1, -2, -4
            return a[0] + a[1]*x + a[2]*y + a[3]*x**2 + a[4]*x*y + a[5]*y**2
        x = [[-2, -1], [-1,3], [0,-1], [1,2], [2,2], [3,-4]]
        y = [f(*i) for i in x]
        A, B, C = quadfit(x, y)    # Returns 5, [4, 3] and [[2, -2], [-2, -8]]

    Example 3::

        def f(x, y, z):
            a = 3, 2, 1, -1, -6, 5, 4, 3, 2, 1
            return (a[0] + a[1]*x + a[2]*y + a[3]*z
                + a[4]*x**2 + a[5]*x*y + a[6]*x*z
                + a[7]*y**2 + a[8]*y*z
                + a[9]*z**2)
        x = [[-2, -1, 0], [-1,2,3], [0,2,-1], [1,1,2], [2,2,2], [-1,3,-4],
             [4,2,-1], [4,1,2], [4,2,2], [1,2,3]]
        y = [f(*i) for i in x]
        A, B, C = quadfit(x, y)


    """
    X, Y = x, y
    del(x, y)
    # Test if x and y are the same size
    m = len(Y)
    if len(X) != m:
        raise ValueError(
            'The input sequences x and y must have the same number of'
            ' elements.')
    # Test if all entries in x have the same dimension
    try:
        n = len(X[0])
    except TypeError:
        n = 1
        X = [np.array([x]) for x in X]
    # Make sure all entries of X have shape (m,)
    X = [np.array(x).reshape((n, )) for x in X]
    # Create array of floats
    X = np.array(X, dtype=float).reshape((m, n))
    # Test if all entries in x are unique
    if len(set([tuple(x) for x in X])) != m:
        raise ValueError('All points in x must be unique.')
    # Test if all entries in y are scalar
    try:
        Y = [float(y) for y in Y]
    except Exception:
        raise ValueError('All entries in y must be floats.')
    # Create array of floats
    Y = np.array(Y, dtype=float).reshape((m, 1))
    # Test if the correct number of points was given:
    expected = (n + 1) * (n + 2) / 2
    if m != expected:
        raise ValueError(
            'Invalid number of points given. Exactly ' + str(expected)
            + ' points are required to fit to x data of dimension ' + str(n)
            + ', currently got ' + str(m) + ' points.')
    # Create matrix of type "(1, x, y, z)". In other words, create a matrix
    # whose first column is m ones, and whose remaining columns are equal to X.
    # For example, if x = [[1, 2], [3, 4], [5, 6]] the matrix should be:
    #  T = [[ 1, 1, 2 ],
    #       [ 1, 3, 4 ],
    #       [ 1, 5, 6 ]]
    # Permutations of this matrix give the powers of x to use in the
    # polynomial: [1, x, y] --> 1*1, 1*x, 1*y, x*x, x*y, y*y
    # Where (x,y) is a point in x
    T = np.concatenate((np.ones((m, 1)), X), axis=1)
    # Create matrix of type "(1, x, y, z, x**2, xy, xz, y**2, yz, z**2)"
    # Do this by multiplying the first element with everything, the second with
    # everything starting from the second, etc:
    #  1*1, 1*x, 1*y, 1*z, x*x, x*y, x*z, y*y, y*z, z*z
    # These powers can be filled in to create a matrix that can be solved to
    # obtain the constants for the polynomial.
    R = np.zeros((m, m))
    k = 0
    for i in range(1 + n):
        for j in range(i, 1 + n):
            R[:, k] = T[:, i] * T[:, j]
            k += 1
    p = np.linalg.solve(R, Y)
    # Format output
    # A is a scalar
    A = p[0]
    # B is a column vector of shape (n,)
    B = p[1:1 + n].reshape((n,))
    # To create C, we first construct an upper triangular matrix D of shape n
    # by n, such that y = A + B' * X + x' * D * x
    D = np.zeros((n, n))
    k = 1 + n
    for i in range(n):
        D[i, i:n] = p[k:k + n - i].transpose()
        k += n - i
    # Now, C = D' + D
    C = D.transpose() + D
    # Return output
    return A, B, C


def quadfit_count(n):
    """
    Given a dimensionality of the input space ``m``, this method returns
    the number of points needed to construct a quadratic polynomial::

        qaudfit_count(n) = (n + 1) * (n + 2) / 2

    """
    n = int(n)
    return (n + 1) * (n + 2) // 2


def quadfit_crit(A, B, C):
    """
    Given an ``A``, ``B`` and ``C`` such as returned by :meth:`quadfit`, this
    method calculates the critical point for the polynomail described by::

        A + B' * x + (1/2) * x' * C * x

    Arguments:

    ``A``
        A scalar.
    ``B``
        A vector of shape ``(n, )``
    ``C``
        A symmetric matrix of size ``(n x n)``.

    The output is a point in n-dimensional space.
    """
    return np.linalg.solve(C, -B)


def quadfit_minimum(A, B, C):
    """
    Given an ``A``, ``B`` and ``C`` such as returned by :meth:`quadfit`, this
    method tests if the polynomial is strictly convex by testing if the
    symmetrical matrix ``C`` is positive definite. This is done by attempting
    to calculate a Cholesky decomposition, which will only succeed if ``C``
    is positive definite.

    The method returns ``True`` if the quadratic polynomial is strictly convex.
    """
    try:
        np.linalg.cholesky(C)
        return True
    except np.linalg.LinAlgError as e:
        if 'not positive definite' in str(e):
            return False
        raise
