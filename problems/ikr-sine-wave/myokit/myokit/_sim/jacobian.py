#
# Two Jacobian calculating tools sharing the same C++ source code.
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
# Location of C source file
SOURCE_FILE = 'jacobian.cpp'
class JacobianTracer(myokit.CppModule):
    """
    Given a model and a simulation log, this class can revisit every logged
    point and generate the corresponding Jacobian.
    
    When created, a ``JacobianTracer`` will generate and compile a back-end
    that uses automatic differentiation to evaluate the partial derivatives of
    the model's right-hand side function at any given point in time.
    
    For a model::
    
        dx/dt = f(x, t, i)
        
    with state x, time t and inputs i, the class will generate the partial
    derivatives of each component of f with respect to each component of x.
    
    Methods to calculate eigenvalues and isolate dominant eigenvalues are
    provided.
    
    N.B. The partial derivatives can not be calculated for the following 
    functions: ``floor``, ``ceil``, ``abs``, quotients and remainders. If these
    are encountered the resulting derivatives will be yielded as ``NaN``.
    However, in many cases, these functions will only occur as part of a
    condition in an if statement, so the ``NaN``'s won't propagate to the final
    result.
    """
    _index = 0 # Unique id
    def __init__(self, model):
        super(JacobianTracer, self).__init__()
        # Require a valid model
        model.validate()
        # Clone model
        self._model = model.clone()
        # Create ordered list of input labels used in the model
        self._inputs = [label for label, var in self._model.bindings()]
        # Extension module id
        JacobianTracer._index += 1
        module_name = 'myokit_JacobianTracer_' + str(JacobianTracer._index)
        # Template arguments
        args = {
            'module_name' : module_name,
            'model'       : self._model,
            'inputs'      : self._inputs,
            }
        fname = os.path.join(myokit.DIR_CFUNC, SOURCE_FILE)
        # Debug
        if myokit.DEBUG:
            print(self._code(fname, args,
                line_numbers=myokit.DEBUG_LINE_NUMBERS))
            import sys
            sys.exit(1)
        # Compile extension
        libs = ['m']
        libd = []
        incd = [myokit.DIR_CFUNC]
        self._ext = self._compile(module_name, fname, args, libs, libd, incd)
    def dominant_eigenvalues(self, log=None, block=None):
        """
        Calculates the dominant eigenvalues of the jacobian matrix for each
        point in time. The returned value is 1d numpy array.
        
        The "dominant eigenvalue" is defined as the eigenvalue with the largest
        magnitude (``sqrt(a + bi)``). Note that the returned values may be
        complex.
        
        If a :class:`DataLog` suitable for :meth:`jacobians` is given as
        ``log``, the jacobians are calculated on the fly. To re-use a set of
        jacobians generated earlier, pass in the :class:`DataBlock2d` generated
        by :meth:`jacobians` as ``block``.
        """
        if log:
            block = self.jacobians(log)
        if block:
            return block.dominant_eigenvalues('jacobians')
        raise ValueError('This method requires either a DataLog suitable'
            ' for the method jacobians() or a DataBlock2d it generated.')
    def jacobians(self, log):
        """
        Calculates the Jacobian matrix for each point in ``log`` and returns
        a :class:`DataBlock2d` containing all the information from the log as
        well as a 2d data series containing the jacobians (stored under the
        key "jacobians").
        
        The given :class:`DataLog` must contain logged results for all
        states in the model and any bound variables used by the model. Bound
        variables whose value does not appear in the log must be unbound before
        creating the :class:`JacobianTracer`. Only results from one-dimensional
        simulations are supported.
        """
        # Test if all states are in log
        n = None
        states = []
        for v in self._model.states():
            try:
                states.append(log[v.qname()])
            except KeyError:
                raise ValueError('Given log must contain all state variables.')
            if n is None:
                n = len(states[-1])
            elif n != len(states[-1]):
                raise ValueError('Each entry in the log must have the same'
                    ' length.')
        # Extract a value for every required input
        inputs = []
        for label in self._inputs:
            v = self._model.binding(label)
            try:
                inputs.append(log[v.qname()])
            except KeyError:
                raise ValueError('The given log must contain logged data for'
                    ' input used by the model. Missing: <' + v.qname() + '> '
                    ' which is bound to ' + label + '.')
        # Create data block
        tvar = self._model.time().qname()
        try:
            time = log[tvar]
        except KeyError:
            raise ValueError('The given log must contain an entry for <'
                + time + '>.')
        nstates = self._model.count_states()
        block = myokit.DataBlock2d(nstates, nstates, time)
        for k, v in log.iteritems():
            if k != tvar:
                block.set0d(k, v)
        # Create iterators over lists of state and input values
        istates = [iter(x) for x in states]
        iinputs = [iter(x) for x in inputs]
        # Length of derivs and partials lists
        ns = self._model.count_states()
        ns2 = ns * ns
        # Pass every state into the generator, store the output
        partials = []
        for i in xrange(n):
            state = [x.next() for x in istates]
            bound = [x.next() for x in iinputs]
            deriv = [0] * ns
            partial = [0] * ns2
            self._ext.calculate(state, bound, deriv, partial)
            # Discard derivatives
            # Convert partial derivatives to numpy array and store
            partial = np.array(partial, copy=False)
            partial = partial.reshape((ns, ns))
            partials.append(partial)
        partials = np.array(partials)
        # Create a simulation
        block.set2d('jacobians', partials, copy=False)
        return block
    def largest_eigenvalues(self, log=None, block=None):
        """
        Calculates the largest eigenvalues of the jacobian matrix at each point
        in time. The returned value is 1d numpy array.
        
        The "largest eigenvalue" is defined as the eigenvalue with the most
        positive real part. Note that the returned values may be complex.
        
        If a :class:`DataLog` suitable for :meth:`jacobians` is given as
        ``log``, the jacobians are calculated on the fly. To re-use a set of
        jacobians generated earlier, pass in the :class:`DataBlock2d` generated
        by :meth:`jacobians` as ``block``.
        """
        if log:
            block = self.jacobians(log)
        if block:
            return block.largest_eigenvalues('jacobians')
        raise ValueError('This method requires either a DataLog suitable'
            ' for the method jacobians() or a DataBlock2d it generated.')
class JacobianCalculator(myokit.CppModule):
    """
    Given a cell model, this class can calculate Jacobian matrices for any
    point in the state space.
    
    The given model will be cloned before use. No inputs are provided by the
    jacobian calculator, so all default values will be used.
    
    N.B. The partial derivatives can not be calculated for the following
    functions: ``floor``, ``ceil``, ``abs``, quotients and remainders. If these
    are encountered the resulting derivatives will be yielded as ``NaN``.
    However, in many cases, these functions will only occur as part of a
    condition in an if statement, so the ``NaN``'s won't propagate to the final
    result.
    """
    _index = 0 # Unique id
    def __init__(self, model):
        super(JacobianCalculator, self).__init__()
        # Require a valid model
        model.validate()
        # Clone model
        self._model = model.clone()
        # Unbind all inputs
        for label, var in self._model.bindings():
            var.set_binding(None)
        # Extension module id
        JacobianCalculator._index += 1
        module_name = 'myokit_JacobianCalculator_' \
            + str(JacobianCalculator._index)
        # Template arguments
        args = {
            'module_name' : module_name,
            'model'       : self._model,
            'inputs'      : [],
            }
        fname = os.path.join(myokit.DIR_CFUNC, SOURCE_FILE)
        # Debug
        if myokit.DEBUG:
            print(self._code(fname, args,
                line_numbers=myokit.DEBUG_LINE_NUMBERS))
            import sys
            sys.exit(1)
        # Compile extension
        libs = ['m']
        libd = []
        incd = [myokit.DIR_CFUNC]
        self._ext = self._compile(module_name, fname, args, libs, libd, incd)
    def calculate(self, state):
        """
        Calculates both the derivatives ``f`` and the Jacobian ``J`` at the
        given state ``x`` and returns a tuple ``(f(x), J(x))``.
        
        The order of state variables must be that specified by the model (i.e.
        the one obtained from calling :meth:`Model.states()`).
        """
        # Check state vector
        n = self._model.count_states()
        if len(state) != n:
            raise ValueError('State vector must have length ' + str(n) + '.')
        try:
            state = [float(x) for x in state]
        except ValueError:
            raise ValueError('State vector must contain floats.')
        # Create input vector
        inputs = []
        # Create output vectors
        deriv = [0] * n
        partial = [0] * n * n
        # Run!
        self._ext.calculate(state, inputs, deriv, partial)
        # Create numpy versions and return
        deriv = np.array(deriv, copy=False)
        partial = np.array(partial, copy=False).reshape((n, n))
        return deriv, partial
    def newton_root(self, x=None, accuracy=0, max_iter=50, damping=1):
        """
        Uses Newton's method to search for a stable point.
        
        An initial guess can be given as ``x``, if no guess is provided the
        model's initial state is used.
        
        Search is halted if the next iteration doesn't change the result, when
        ``max|f(x)| < accuracy`` or after ``max_iter`` iterations. The
        accuracy and maximum iterations criteria can be disabled by setting
        the parameters to 0.
        
        A damping factor can be applied to every step by setting a damping
        factor ``damping`` to some value between ``0`` and ``1``. With
        ``damping=1`` the full step suggested by Newton's method is taken. With
        any smaller value only a fraction of the suggested step is made.
        
        Returns a tuple ``(x*, f*, j*, e*)``, where ``x*`` is a root, ``f*`` is
        the derivative vector at ``x*``, ``j*`` is the Jacobian matrix at this
        point and ``e* = max|f(x*)|``.
        """
        # Check damping variable
        if damping <= 0 or damping > 1:
            raise ValueError('Damping must be between 0 and 1.')
        # Get initial state
        if x is None:
            x = self._model.state()
        x = np.array(x)
        # Calculate derivatives & jacobian
        f, j = self.calculate(x)
        e = np.max(np.abs(f))
        # Iterations
        iterations = 0
        # Best solution
        best = x, f, j, e
        # Start
        while e > accuracy:
            # Solve J*s = -f
            #s = np.linalg.solve(j, -f)
            s = np.dot(np.linalg.pinv(j), -f)
            # Estimate relative step size
            if not np.any(x == 0):
                # Calculate relative step size
                d = np.max(np.abs(s / x))
            elif not np.all(x == 0):
                # Remove zeros and calculate relative step size
                d = np.nonzero(x)
                d = np.max(np.abs(s[d] / x[d])) # Bugfix due to Enno de Lange
            else:
                # Unable to calculate
                d = 1
            # Provide maximum relative step size based damping
            d = min(100 / d, damping)
            # Take step to next point
            x2 = x + d * s
            if np.all(x2 == x):
                break
            x = x2
            # Check iterations
            iterations += 1
            if max_iter > 0 and iterations >= max_iter:
                break
            # Calculate derivatives & jacobian
            f, j = self.calculate(x)
            e = np.max(np.abs(f))
            if e < best[3]:
                best = x, f, j, e
        return best
