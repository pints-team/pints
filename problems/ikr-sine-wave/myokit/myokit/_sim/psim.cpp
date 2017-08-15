<?
# psim.cpp
#
# Runs a simulation with differential objects, to obtain the state and the
# partial derivatives of the state with respect to a list of parameters.
#
# Required variables
# -----------------------------------------------------------------------------
# module_name      A module name
# model            A myokit model
# variables        A list of variables y whose derivatives dy/dp to track
# parameters       A list of parameters p (all literal constants)
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

// Number of states, variables, parameters
#define NS <?= model.count_states() ?>
#define NV <?= len(variables) ?>
#define NP <?= len(parameters) ?>
#define NSP <?= model.count_states() * len(parameters) ?>
#define NVP <?= len(variables) * len(parameters) ?>

// Define numerical type
typedef double Real;

// Number of derivatives in each differential vector (required for differential.hpp)
#define N_DIFFS <?= len(parameters) ?>

// Load differential object
#include "differential.hpp"

// Define differential type.
typedef FirstDifferential Diff;

<?
print('// Aliases of state variable values')
for var in model.states():
    print('#define ' + v(var) + ' state[' + str(var.indice()) + ']')
print('')

print('// Aliases of state variable derivatives')
for var in model.states():
    print('#define ' + v(var.lhs()) + ' state_ddt[' + str(var.indice()) + ']')
print('')

print('// Aliases of parameters')
for k, var in enumerate(parameters):
    print('#define ' + v(var.lhs()) + ' param[' + str(k) + ']')
print('')

print('// Constants & calculated constants (may depend on parameters!)')
for group in equations.itervalues():
    for eq in group.equations(const=True):
        if eq.lhs.var() not in parameters:
            if isinstance(eq.rhs, myokit.Number):
                print('static Real ' + v(eq.lhs) + ' = ' + w.ex(eq.rhs) + ';')
            else:
                print('static Diff ' + v(eq.lhs) + ';')
print('')

print('// Declare remaining variables')
for var in model.variables(state=False, const=False, deep=True):
    print('static Diff ' + v(var) + ';')
print('')

?>

// Inputs
double engine_time;
double engine_pace;

// Simple exception raising with return e("message")
PyObject* e(const char* msg)
{
    PyErr_SetString(PyExc_Exception, msg);
    return 0;
}

// Calculated constants
static int
calculate_constants(Diff* state, Diff* state_ddt, Diff* param)
{
<?
for group in equations.itervalues():
    for eq in group.equations(const=True):
        if eq.lhs.var() not in parameters:
            if not isinstance(eq.rhs, myokit.Number):
                print(tab + w.eq(eq) + ';')
?>
}

// Right-hand-side function of the model ODE
static int
rhs(Diff* state, Diff* state_ddt, Diff* param)
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
PyObject* param_in;         // The parameter values
PyObject* state_in;         // The initial state
PyObject* state_ddp_in;     // The initial state-parameter-derivatives (as a list)
PyObject* state_out;        // The final state
PyObject* state_ddp_out;    // The final state-parameter-derivatives (as a list)
PyObject* protocol;         // The pacing protocol (if any)
PyObject* log_dict;         // The simulation log to log to
PyObject* log_varab_ddp;    // A list to store lists of variable-parameter-derivatives in
double log_interval;        // The logging interval

// State vector & state vector time derivatives
Diff* state;
Diff* state_ddt;

// Parameters
Diff* param;

// Step size
// Typically, dt = default_dt. However, if that dt would take the simulation
// beyond the next pacing event or the end of the simulation, it will be
// shortened to arrive there exactly.
double dt;
double dt_min;      // Minimum step size

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
            free(state_ddt); state_ddt = NULL;
            free(param); param = NULL;
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
        if (!PyArg_ParseTuple(args, "dddOOOOOOOOd",
                &tmin,
                &tmax,
                &default_dt,
                &param_in,
                &state_in,
                &state_ddp_in,
                &state_out,
                &state_ddp_out,
                &protocol,
                &log_dict,
                &log_varab_ddp,
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
        if (PyList_Size(state_in) != NS) { return e("Incorrect length: state_in."); }
        for(i=0; i<NS; i++) {
            if (!PyFloat_Check(PyList_GetItem(state_in, i))) {
                return e("Must contain floats: arg_state.");
            }
        }

        // Check initial state derivatives vector state_ddp_in
        if (!PyList_Check(state_ddp_in)) { return e("Not a list: state_ddp_in."); }
        if (PyList_Size(state_ddp_in) != NSP) { return e("Incorrect length: state_ddp_in."); }
        for(i=0; i<NSP; i++) {
            if (!PyFloat_Check(PyList_GetItem(state_ddp_in, i))) {
                return e("Must contain floats: state_ddp_in.");
            }
        }
        
        // Check parameter values vector param_in
        if (!PyList_Check(param_in)) { return e("Not a list: param_in."); }
        if (PyList_Size(param_in) != NP) { return e("Incorrect length: param_in."); }
        for(i=0; i<NP; i++) {
            if (!PyFloat_Check(PyList_GetItem(param_in, i))) {
                return e("Must contain floats: param_in.");
            }
        }
        
        // Check final state vector state_out
        if (!PyList_Check(state_out)) { return e("Not a list: state_out."); }
        if (PyList_Size(state_out) != NS) { return e("Incorrect length: state_out."); }

        // Check final derivatives vector state_ddp_out
        if (!PyList_Check(state_ddp_out)) { return e("Not a list: state_ddp_out."); }
        if (PyList_Size(state_ddp_out) != NSP) { return e("Incorrect length: state_ddp_out."); }
        
        // Check if the log is a dict
        if (!PyDict_Check(log_dict)) { return e("Not a dict: log_dict."); }
        
        // Check list for logging derivatives in
        if (!PyList_Check(log_varab_ddp)) { return e("Not a list: log_varab_ddp."); }
        if (PyList_Size(log_varab_ddp) != 0) { return e("Not empty: log_varab_ddp."); }

        ///////////////////////////////////////////////////////////////////////
        //
        // From this point on, memory will be allocated. Any further errors
        // should call sim_clean() before returning
        //
        
        // From this point on, we're running!
        running = 1;
        
        // Initialize state vector
        state = (Diff*)malloc(sizeof(Diff) * NS);
        for(i=0; i<NS; i++) {
            state[i] = Diff(PyFloat_AsDouble(PyList_GetItem(state_in, i)));
            for(j=0; j<NP; j++) {
                state[i][j] = PyFloat_AsDouble(PyList_GetItem(state_ddp_in, i*NP+j));
            }
        }
        
        // Initialize state-time-derivatives vector
        state_ddt = (Diff*)malloc(sizeof(Diff) * NS);
        
        // Initialize parameter vector
        param = (Diff*)malloc(sizeof(Diff) * NP);
        for(i=0; i<NP; i++) {
            param[i] = Diff(PyFloat_AsDouble(PyList_GetItem(param_in, i)), i);
            param[i][i] = 1.0;
        }
        
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
        
        // Calculate constants
        calculate_constants(state, state_ddt, param);
        
        // Evaluate derivatives at this point. This will be used for logging
        // and to take the first step.
        rhs(state, state_ddt, param);
        
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
        
        // Always store initial position
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
        
        // Log variable-parameter-derivatives
        list = PyList_New(NVP);
        if (list == NULL) return sim_clean();
<?
NP = len(parameters)
for i, var in enumerate(variables):
    for j, par in enumerate(parameters):
        print(tab*2 + 'PyList_SetItem(list, ' + str(i*NP+j) + ', PyFloat_FromDouble(' + v(var) + '[' + str(j) + ']));')
?>
        if (PyList_Append(log_varab_ddp, list) != 0) {
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
            for(i=0; i<NS; i++) {
                state[i] += state_ddt[i] * dt;
            }
            engine_time += dt;
            flag_pacing = ESys_AdvanceTime(pacing, engine_time, tmax);
            if (flag_pacing!=ESys_OK) { ESys_SetPyErr(flag_pacing); return sim_clean(); }
            tpace = ESys_GetNextTime(pacing, NULL);
            engine_pace = ESys_GetLevel(pacing, NULL);
            rhs(state, state_ddt, param);
            
            // Check for NaN, these will eventually propagate to all variables,
            // so we only have to check a single one.
            if (isnan(state[0].value())) {
                PyErr_SetString(PyExc_Exception, "NaN occurred in state vector during simulation. Perhaps there is an error in the model code or the step size should be reduced.");
                return sim_clean();
            }

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
                
                // Log variable-parameter-derivatives
                list = PyList_New(NVP);
                if (list == NULL) return sim_clean();
<?
NP = len(parameters)
for i, var in enumerate(variables):
    for j, par in enumerate(parameters):
        print(tab*4 + 'PyList_SetItem(list, ' + str(i*NP+j) + ', PyFloat_FromDouble(' + v(var) + '[' + str(j) + ']));')
?>
                if (PyList_Append(log_varab_ddp, list) != 0) {
                    Py_DECREF(list);
                    list = NULL;
                    return sim_clean();
                }
                Py_DECREF(list);
                list = NULL;
                    
                // Calculate next logging point
                ilog++;
                tlog = tmin + (double)ilog * log_interval;
            }
            
            // Report back to python after every x steps
            steps_taken++;
            if (steps_taken >= 100) {
                return PyFloat_FromDouble(engine_time);
            }
        }
        
        // Set final state & state-parameter-derivatives
        for(i=0; i<NS; i++) {
            PyList_SetItem(state_out, i, PyFloat_FromDouble(state[i].value()));
            // PyList_SetItem steals a reference: no need to decref the Float!
            for(j=0; j<NP; j++) {
                PyList_SetItem(state_ddp_out, i*NP+j, PyFloat_FromDouble(state[i][j]));
            }
        }

        // Clean up and return
        sim_clean();
        return PyFloat_FromDouble(engine_time);
    }
    
    /*
     * Alters the value of a (literal) constant
     */
    static PyObject*
    sim_set_constant(PyObject *self, PyObject *args)
    {
        // Check input arguments
        char* name;
        double value;
        if (!PyArg_ParseTuple(args, "sd", &name, &value)) {
            PyErr_SetString(PyExc_Exception, "Expected input arguments: name (str), value (Float).");
            // Nothing allocated yet, no pyobjects _created_, return directly
            return 0;
        }

<?
for var in model.variables(const=True, deep=True):
    if var.is_literal() and var not in parameters:
        print(tab + 'if(strcmp("' + var.qname() + '", name) == 0) {')
        print(tab + tab + v(var) + ' = value;')
        print(tab + tab + 'Py_RETURN_NONE;')
        print(tab + '}')
?>
        char errstr[200];
        sprintf(errstr, "Constant not found: <%s>", name);
        PyErr_SetString(PyExc_Exception, errstr);
        return 0;
    }
    
    // Methods in this module
    static PyMethodDef SimMethods[] = {
        {"set_constant", sim_set_constant, METH_VARARGS, "Change a (literal) constant."},
        {"sim_clean", py_sim_clean, METH_VARARGS, "Clean up after an aborted simulation."},
        {"sim_init", sim_init, METH_VARARGS, "Initialize the simulation."},
        {"sim_step", sim_step, METH_VARARGS, "Perform the next step in the simulation."},
        {NULL},
    };

    // Module definition
    PyMODINIT_FUNC
    init<?=module_name?>(void) {
        (void) Py_InitModule("<?= module_name ?>", SimMethods);
    }
}

// Remove aliases of states, bound variables
<?
for var in model.states():
    print('#undef ' + v(var.lhs()))
for var in model.states():
    print('#undef ' + v(var))
?>
