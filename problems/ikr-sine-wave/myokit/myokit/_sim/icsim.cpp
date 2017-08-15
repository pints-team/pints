<?
# icsim.cpp
#
# Runs a simulation with differential objects, to obtain the state and the
# partial derivatives of the state with respect to the initial conditions.
#
# Required variables
# -----------------------------------------------------------------------------
# module_name      A module name
# model            A myokit model
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

# Get mapping of bound variables
bound = model.prepare_bindings({
    'time' : 'engine_time',
    'pace' : 'engine_pace',
    })

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
    # Handle bound variables, states, constants and others
    return 'V_' + var.uname()
w.set_lhs_function(v)

# Tab
tab = '    '

?>
#include <Python.h>
#include "pacing.h"

// Number of states
#define N_STATE <?= model.count_states() ?>
#define N_MATRIX <?= model.count_states() ** 2 ?>

// Define numerical type
typedef double Real;

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

print('// Aliases of constants and calculated constants')
for group in equations.itervalues():
    for eq in group.equations(const=True):
        if isinstance(eq.rhs, myokit.Number):
            print('#define ' + v(eq.lhs) + ' ' + w.ex(eq.rhs))
        else:
            print('#define ' + v(eq.lhs) + ' (' + w.ex(eq.rhs) + ')')
print('')

print('// Declare remaining variables')
for var in model.variables(state=False, const=False, deep=True):
    print('static Diff ' + v(var) + ';')
print('')

?>

// Inputs
double engine_time;
double engine_pace;

// Simple exceptions
PyObject* e(const char* msg)
{
    PyErr_SetString(PyExc_Exception, msg);
    return 0;
}

// Right-hand-side function of the model ODE
static int
rhs(Diff* state, Diff* deriv)
{
<?
for label, eqs in equations.iteritems():
    if eqs.has_equations(const=False):
        print(tab + '// ' + label)
        for eq in eqs.equations(const=False):
            var = eq.lhs.var()
            if var in bound:
                print(tab + v(var) + ' = ' + bound[var] + ';')
            else:
                print(tab + w.eq(eq) + ';')
        print(tab)
?>
    return 0;
}

// Adds a variable to the logging lists. Returns 1 if successful.
static int
log_add(PyObject* log_dict, PyObject** logs, Diff** vars, int i, const char* name, const Diff* var)
{
    int added = 0;
    PyObject* key = PyString_FromString(name);
    if (PyDict_Contains(log_dict, key)) {
        logs[i] = PyDict_GetItem(log_dict, key);
        vars[i] = (Diff*)var;
        added = 1;
    }
    Py_DECREF(key);
    return added;
}

// Input arguments
double tmin;                // The initial simulation time
double tmax;                // The final simulation time
double default_dt;          // The default step size
PyObject* state_in;         // The initial state
PyObject* deriv_in;         // The initial partial derivatives (as a list)
PyObject* state_out;        // The final state
PyObject* deriv_out;        // The final partial derivatives (as a list)
PyObject* protocol;         // The pacing protocol (if any)
PyObject* log_dict;         // The simulation log to log to
PyObject* log_deriv;        // A list to store lists of partial derivatives in
double log_interval;        // The logging interval

// State vector & derivatives
Diff* state;
Diff* deriv;

// Step size
// Typically, dt = default_dt. However, if that dt would take the simulation
// beyond the next pacing event or the end of the simulation, it will be
// shortened to arrive there exactly.
double dt;
double dt_min;  // Minimum step size

// Simulation state
int running;

// Logging
PyObject** logs = NULL;     // An array of lists to log into
Diff** vars = NULL;         // An array of pointers to variables to log
int n_vars;                 // Number of logging variables
unsigned long ilog;         // Index of next logging point
double tlog;                // Time of next logging point

// Pacing
ESys pacing = NULL;         // Pacing system
double tpace;               // Time of next event

// Temporary python objects
PyObject* flt = NULL;               // PyFloat, various uses
PyObject* ret = NULL;               // PyFloat, used as return value
PyObject* list_update_str = NULL;   // PyString, used to call "append" method
PyObject* list = NULL;              // A new list, created to hold derivatives

// Python callable methods
extern "C" {
    
    /*
     * Cleans up after a simulation
     */
    static PyObject*
    sim_clean()
    {
        if (running != 0) {
            // Done with str="append", decref it
            Py_XDECREF(list_update_str); list_update_str = NULL;

            // Free allocated memory
            free(state); state = NULL;
            free(deriv); deriv = NULL;
            free(vars); vars = NULL;
            free(logs); logs = NULL;
            
            // Free pacing system space
            ESys_Destroy(pacing); pacing = NULL;

            // No longer running
            running = 0;
        }

        // Return 0, allowing the construct
        //  PyErr_SetString(PyExc_Exception, "Oh noes!");
        //  return sim_clean()
        // to terminate a python function.
        return 0;
    }
    static PyObject*
    py_sim_clean(PyObject *self, PyObject *args)
    {
        sim_clean();
        
        Py_RETURN_NONE;
    }

    /*
     * Initializes a simulation
     */
    static PyObject*
    sim_init(PyObject* self, PyObject* args)
    {
        int i, j;
        
        // Check if already running
        if (running != 0) {
            PyErr_SetString(PyExc_Exception, "Simulation already initialized.");
            return 0;
        }
    
        // Check input arguments
        if (!PyArg_ParseTuple(args, "dddOOOOOOOd",
                &tmin,
                &tmax,
                &default_dt,
                &state_in,
                &deriv_in,
                &state_out,
                &deriv_out,
                &protocol,
                &log_dict,
                &log_deriv,
                &log_interval
                )) {
            PyErr_SetString(PyExc_Exception, "Incorrect input arguments.");
            // Nothing allocated yet, no pyobjects _created_, return directly
            return 0;
        }
        
        // Check tmin, tmax
        if (tmax < tmin) return e("Error: tmax < tmin!");
        
        // Check default step size
        if (default_dt <= 0) return e("Error: step size must be > 0");
        dt_min = default_dt * 1e-2;
        
        // Check initial state vector state_in
        if (!PyList_Check(state_in)) { return e("Not a list: state_in."); }
        if (PyList_Size(state_in) != N_STATE) { return e("Incorrect length: state_in."); }
        for(i=0; i<N_STATE; i++) {
            if (!PyFloat_Check(PyList_GetItem(state_in, i))) {
                return e("Must contain floats: state_in.");
            }
        }

        // Check initial derivatives vector deriv_in
        if (!PyList_Check(deriv_in)) { return e("Not a list: deriv_in."); }
        if (PyList_Size(deriv_in) != N_MATRIX) { return e("Incorrect length: deriv_in."); }
        for(i=0; i<N_MATRIX; i++) {
            if (!PyFloat_Check(PyList_GetItem(deriv_in, i))) {
                return e("Must contain floats: deriv_in.");
            }
        }
        
        // Check final state vector state_out
        if (!PyList_Check(state_out)) { return e("Not a list: state_out."); }
        if (PyList_Size(state_out) != N_STATE) { return e("Incorrect length: state_out."); }

        // Check final derivatives vector deriv_out
        if (!PyList_Check(deriv_out)) { return e("Not a list: deriv_out."); }
        if (PyList_Size(deriv_out) != N_MATRIX) { return e("Incorrect length: deriv_out."); }
        
        // Check if the log is a dict
        if (!PyDict_Check(log_dict)) { return e("Not a dict: log_dict."); }
        
        // Check list for logging derivatives in
        if (!PyList_Check(log_deriv)) { return e("Not a list: log_deriv."); }
        if (PyList_Size(log_deriv) != 0) { return e("Not empty: log_deriv."); }

        ///////////////////////////////////////////////////////////////////////
        //
        // From this point on, memory will be allocated. Any further errors
        // should call sim_clean() before returning
        //
        
        // From this point on, we're running!
        running = 1;
        
        // Initialize state vector
        state = (Diff*)malloc(sizeof(Diff) * N_STATE);
        for(i=0; i<N_STATE; i++) {
            state[i] = Diff(PyFloat_AsDouble(PyList_GetItem(state_in, i)));
            for(j=0; j<N_STATE; j++) {
                state[i][j] = PyFloat_AsDouble(PyList_GetItem(deriv_in, i*N_STATE+j));
            }
        }
        
        // Initialize derivatives vector
        deriv = (Diff*)malloc(sizeof(Diff) * N_STATE);
        
        // Set up pacing
        ESys_Flag flag_pacing;
        pacing = ESys_Create(&flag_pacing);
        if (flag_pacing != ESys_OK) { ESys_SetPyErr(flag_pacing); return sim_clean(); }
        flag_pacing = ESys_Populate(pacing, protocol);
        if (flag_pacing != ESys_OK) { ESys_SetPyErr(flag_pacing); return sim_clean(); }
        flag_pacing = ESys_AdvanceTime(pacing, tmin, tmax);
        if (flag_pacing != ESys_OK) { ESys_SetPyErr(flag_pacing); return sim_clean(); }
        
        // Initialize inputs
        engine_time = tmin;
        engine_pace = ESys_GetLevel(pacing, NULL);
        
        // Evaluate derivatives at this point. This will be used for logging
        // and to take the first step.
        rhs(state, deriv);
        
        //
        // Running & logging:
        //  - We start at time t, with a known state y and inputs i
        //  - At this point, we make a call to rhs()
        //  - Now we have y(t), all intermediary variables at t, and ydot(t)
        //  - If needed, log everything for time-point t
        //  - Now we can update to time t+dt
        // So, before the simulation starts y is known and rhs() is called to
        // evaluate the rest. If no previous logged data is present, this first
        // data point is logged.
        // At each simulation step:
        //  - y is updated to time t + t_step
        //  - rhs(y) is called to evaluate everything else
        //  - t is updated to t + tstep
        //  - If needed, everything is logged for this new t
        //
        
        // Next event & logging times
        tpace = ESys_GetNextTime(pacing, NULL);
        tlog = tmin;
        
        // Set up logging
        n_vars = PyDict_Size(log_dict);
        logs = (PyObject**)malloc(sizeof(PyObject*)*n_vars);
        vars = (Diff**)malloc(sizeof(Diff*)*n_vars);
        i = 0;
<?
for var in model.variables(deep=True, const=False):
    print(tab*2 + 'i += log_add(log_dict, logs, vars, i, "' + var.qname() + '", &' + v(var)  + ');')
?>
        if (i != n_vars) {
            PyErr_SetString(PyExc_Exception, "Unknown variables found in logging dictionary.");
            return sim_clean();
        }
    
        // Always store initial position in logs
        list_update_str = PyString_FromString("append");

        // Log variables
        for(i=0; i<n_vars; i++) {
            flt = PyFloat_FromDouble(vars[i]->value()); // Append doesn't steal
            ret = PyObject_CallMethodObjArgs(logs[i], list_update_str, flt, NULL);
            Py_DECREF(flt); flt = NULL;
            Py_XDECREF(ret);
            if (ret == NULL) {
                PyErr_SetString(PyExc_Exception, "Call to append() failed on logging list.");
                return sim_clean();
            }
        }
        ret = NULL;
        
        // Log partial derivatives
        list = PyList_New(N_MATRIX);
        if (list == NULL) return sim_clean();
        for(i=0; i<N_STATE; i++) {
            for(j=0; j<N_STATE; j++) {
                PyList_SetItem(list, i*N_STATE+j, PyFloat_FromDouble(state[i][j]));
            }
        }
        if (PyList_Append(log_deriv, list) != 0) {
            Py_DECREF(list);
            list = NULL;
            return sim_clean();
        }
        Py_DECREF(list);
        list = NULL;
        
        // Set periodic log point 1 log_interval ahead
        ilog = 1;
        tlog = tmin + log_interval;

        // Done!
        Py_RETURN_NONE;
    }
    
    /*
     * Takes the next steps in a simulation run
     */
    static PyObject*
    sim_step(PyObject *self, PyObject *args)
    {
        ESys_Flag flag_pacing;
        int i, j;
        int steps_taken = 0;    // Steps taken during this call
        double d;
        
        while(1) {
        
            // Calculate next step size
            dt = default_dt;
            d = tpace - engine_time; if (d > dt_min && d < dt) dt = d;
            d = tmax - engine_time; if (d > dt_min && d < dt) dt = d;
            d = tlog - engine_time; if (d > dt_min && d < dt) dt = d;
            
            // Advance to next time step
            for(i=0; i<N_STATE; i++) state[i] += deriv[i] * dt;
            engine_time += dt;
            flag_pacing = ESys_AdvanceTime(pacing, engine_time, tmax);
            if (flag_pacing!=ESys_OK) { ESys_SetPyErr(flag_pacing); return sim_clean(); }
            tpace = ESys_GetNextTime(pacing, NULL);
            engine_pace = ESys_GetLevel(pacing, NULL);
            rhs(state, deriv);

            // Check if we're finished
            // Do this *before* logging (half-open interval rule)
            if (engine_time >= tmax) break;
            
            // Logging
            if (engine_time >= tlog) {
                
                // Log variables
                for(i=0; i<n_vars; i++) {
                    flt = PyFloat_FromDouble(vars[i]->value());
                    ret = PyObject_CallMethodObjArgs(logs[i], list_update_str, flt, NULL);
                    Py_DECREF(flt); flt = NULL;
                    Py_XDECREF(ret);
                    if (ret == NULL) {
                        PyErr_SetString(PyExc_Exception, "Call to append() failed on logging list.");
                        return sim_clean();
                    }
                }
                ret = NULL;
                
                // Log partial derivatives
                list = PyList_New(N_MATRIX);
                if (list == NULL) return sim_clean();
                for(i=0; i<N_STATE; i++) {
                    for(j=0; j<N_STATE; j++) {
                        PyList_SetItem(list, i*N_STATE+j, PyFloat_FromDouble(state[i][j]));
                    }
                }
                if (PyList_Append(log_deriv, list) != 0) {
                    Py_DECREF(list); list = NULL;
                    return sim_clean();
                }
                Py_DECREF(list); list = NULL;

                // Calculate next logging point
                ilog++;
                tlog = tmin + (double)ilog * log_interval;
            }
            
            // Report back to python after every x steps
            steps_taken++;
            if (steps_taken >= 20) {
                return PyFloat_FromDouble(engine_time);
            }
        }
        
        // Set final state & partial derivatives
        for(i=0; i<N_STATE; i++) {
            PyList_SetItem(state_out, i, PyFloat_FromDouble(state[i].value()));
            // PyList_SetItem steals a reference: no need to decref the Float!
            for(j=0; j<N_STATE; j++) {
                PyList_SetItem(deriv_out, i*N_STATE+j, PyFloat_FromDouble(state[i][j]));
            }
        }

        // Clean up and return
        sim_clean();
        return PyFloat_FromDouble(engine_time);
    }
    
    // Methods in this module
    static PyMethodDef SimMethods[] = {
        {"sim_init", sim_init, METH_VARARGS, "Initialize the simulation."},
        {"sim_step", sim_step, METH_VARARGS, "Perform the next step in the simulation."},
        {"sim_clean", py_sim_clean, METH_VARARGS, "Clean up after an aborted simulation."},
        {NULL},
    };

    // Module definition
    PyMODINIT_FUNC
    init<?=module_name?>(void) {
        (void) Py_InitModule("<?= module_name ?>", SimMethods);
    }
}

// Remove aliases of states, bound variables, constants
<?
for var in model.states():
    print('#undef ' + v(var.lhs()))
for var in model.states():
    print('#undef ' + v(var))
for group in equations.itervalues():
    for eq in group.equations(const=True):
        print('#undef ' + v(eq.lhs))
?>
