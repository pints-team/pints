<?
# jacobian.cpp
#
# Used to generate Jacobian matrices for a model given a set of logged points.
#
# Required variables
# -----------------------------------------------------------------------------
# module_name      A module name
# model            A myokit model
# inputs           An ordered list of input labels used in the model
# -----------------------------------------------------------------------------
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
import myokit.formats.cpp as cpp

# Get model
model.reserve_unique_names(*cpp.keywords)
model.create_unique_names()

# Get equations
equations = model.solvable_order()

# Get expression writer
w = cpp.CppExpressionWriter()

# Set if-then-else function
w.set_condition_function('ifte')

# Define var/lhs function
def v(var):
    # Explicitly asked for derivative?
    if isinstance(var, myokit.Derivative):
        return 'D_' + var.var().uname()
    # Convert LhsExpressions to Variables
    if isinstance(var, myokit.Name):
        var = var.var()
    # Handle inputs, states, constants and others
    return 'V_' + var.uname()
w.set_lhs_function(v)

# Tab
tab = '    '

?>
#include <Python.h>

// Number of states
#define N_STATE <?= model.count_states() ?>
#define N_STATE2 <?= model.count_states() ** 2 ?>

// Number of inputs
#define N_INPUT <?= len(inputs) ?>

// Define numerical type
typedef long double Real;

// Number of derivatives in each derivative vector
#define N_DIFFS <?= model.count_states() ?>

// Load differential object
#include "differential.hpp"

// Define differential type.
typedef FirstDifferential Diff;

<?
print('// Aliases of state variable derivatives')
for var in model.states():
    print('#define ' + v(var.lhs()) + ' deriv[' + str(var.indice()) + ']')
print('')

print('// Aliases of state variable values')
for var in model.states():
    print('#define ' + v(var) + ' state[' + str(var.indice()) + ']')
print('')

print('// Aliases of input variables')
for k, label in enumerate(inputs):
    print('#define ' + v(model.binding(label)) + ' inputs[' + str(k) + ']')
print('')

print('// Aliases of constants and calculated constants')
for group in equations.itervalues():
    for eq in group.equations(const=True):
        if isinstance(eq.rhs, myokit.Number):
            print('#define ' + v(eq.lhs) + ' ' + w.ex(eq.rhs))
        else:
            print('#define ' + v(eq.lhs) + ' (' + w.ex(eq.rhs) + ')')
print('')

print('// Declare remaining variables')
for var in model.variables(state=False, const=False, bound=False, deep=True):
    print('static Diff ' + v(var) + ';')
print('')

?>

// Simple exceptions
PyObject* e(const char* msg)
{
    PyErr_SetString(PyExc_Exception, msg);
    return 0;
}

// Right-hand-side function of the model ODE
static int
rhs(Diff* state, Diff* deriv, Real* inputs)
{
<?
for label, eqs in equations.iteritems():
    if eqs.has_equations(const=False, bound=False):
        print(tab + '// ' + label)
        for eq in eqs.equations(const=False, bound=False):
            print(tab + w.eq(eq) + ';')
        print(tab)
?>
    return 0;
}

// Input arguments
PyObject* arg_state;     // The state variable values (input)
PyObject* arg_input;     // The input variable values (input)
PyObject* arg_deriv;     // The calculated derivatives (output)
PyObject* arg_partial;   // The calculated partial derivatives (output)

// State vector
Diff* state;
Diff* deriv;
Real* input;

// Clean-up method
PyObject* clean()
{
    free(state); state = NULL;
    free(deriv); deriv = NULL;
    free(input); input = NULL;
    
    return 0;
}


// Python callable methods
extern "C" {

    // Calculates the derivatives and partial derivatives for a particular point in
    // time.
    static PyObject*
    calculate(PyObject* self, PyObject* args)
    {
        int i, j;
    
        // Check input arguments
        if (!PyArg_ParseTuple(args, "OOOO",
                &arg_state,
                &arg_input,
                &arg_deriv,
                &arg_partial
                )) {
            PyErr_SetString(PyExc_Exception, "Incorrect input arguments.");
            // Nothing allocated yet, no pyobjects _created_, return directly
            return 0;
        }
        
        // Check given state vector
        if (!PyList_Check(arg_state)) { return e("Not a list: arg_state."); }
        if (PyList_Size(arg_state) != N_STATE) { return e("Incorrect length: arg_state."); }
        for(i=0; i<N_STATE; i++) {
            if (!PyFloat_Check(PyList_GetItem(arg_state, i))) {
                return e("Must contain floats: arg_state.");
            }
        }
        
        // Check given input vector
        if (!PyList_Check(arg_input)) { return e("Not a list: arg_input."); }
        if (PyList_Size(arg_input) != N_INPUT) { return e("Incorrect length: arg_input."); }
        for(i=0; i<N_INPUT; i++) {
            if (!PyFloat_Check(PyList_GetItem(arg_input, i))) {
                return e("Must contain floats: arg_input.");
            }
        }

        // Check given derivatives vector
        if (!PyList_Check(arg_deriv)) { return e("Not a list: arg_deriv."); }
        if (PyList_Size(arg_deriv) != N_STATE) { return e("Incorrect length: arg_deriv."); }
        
        // Check given partial derivatives vector
        if (!PyList_Check(arg_partial)) { return e("Not a list: arg_partial."); }
        if (PyList_Size(arg_partial) != N_STATE2) { return e("Incorrect length: arg_partial."); }
        
        // From this point on, memory will be allocated
        // Returning after this point should happen with clean()
        
        // Create state vector, derivatives vector & input vector
        state = (Diff*)malloc(sizeof(Diff) * N_STATE);
        deriv = (Diff*)malloc(sizeof(Diff) * N_STATE);
        input = (Real*)malloc(sizeof(Real) * N_INPUT);
        for(i=0; i<N_STATE; i++) {
            state[i] = Diff(PyFloat_AsDouble(PyList_GetItem(arg_state, i)), i);
        }
        for(i=0; i<N_INPUT; i++) {
            input[i] = PyFloat_AsDouble(PyList_GetItem(arg_input, i));
        }
        
        // Run!
        rhs(state, deriv, input);
        
        // Copy results into output objects
        for(i=0; i<N_STATE; i++) {
            PyList_SetItem(arg_deriv, i, PyFloat_FromDouble(deriv[i].value()));
            for(j=0; j<N_STATE; j++) {
                PyList_SetItem(arg_partial, i*N_STATE+j, PyFloat_FromDouble(deriv[i][j]));
            }
        }
    
        // Finished succesfully, clean up and return
        clean();
        Py_RETURN_NONE;
    }

    // Methods in this module
    static PyMethodDef SimMethods[] = {
        {"calculate", calculate, METH_VARARGS, "Calculates the derivatives and partial derivatives."},
        {NULL},
    };

    // Module definition
    PyMODINIT_FUNC
    init<?=module_name?>(void) {
        (void) Py_InitModule("<?= module_name ?>", SimMethods);
    }
}

// Remove aliases of states, input variables, constants
<?
for var in model.states():
    print('#undef ' + v(var.lhs()))
for var in model.states():
    print('#undef ' + v(var))
for label in inputs:
    print('#undef ' + v(model.binding(label)))
for group in equations.itervalues():
    for eq in group.equations(const=True):
        print('#undef ' + v(eq.lhs))
?>
