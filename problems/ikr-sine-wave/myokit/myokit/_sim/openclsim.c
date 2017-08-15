<?
# opencl.c
#
# A pype template for opencl driven 1d or 2d simulations
#
# Required variables
# -----------------------------------------------------------------------------
# module_name       A module name
# model             A myokit model, cloned with independent components
# precision         A myokit precision constant
# dims              The number of dimensions, either 1 or 2
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
import myokit.formats.opencl as opencl

tab = '    '
?>
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "pacing.h"
#include "mcl.h"

// Show debug output
//#define MYOKIT_DEBUG

#define n_state <?= str(model.count_states()) ?>

typedef <?= ('float' if precision == myokit.SINGLE_PRECISION else 'double') ?> Real;

/*
 * Adds a variable to the logging lists. Returns 1 if successful.
 *
 * Arguments
 *  log_dict : The dictionary of logs passed in by the user
 *  logs     : Pointers to a log for each logged variables
 *  vars     : Pointers to each variable to log
 *  i        : The index of the next logged variable
 *  name     : The variable name to search for in the dict
 *  var      : The variable to add to the logs, if its name is present
 * Returns 0 if not added, 1 if added.
 */
static int log_add(PyObject* log_dict, PyObject** logs, Real** vars, int i, char* name, const Real* var)
{
    int added = 0;
    PyObject* key = PyString_FromString(name);
    if(PyDict_Contains(log_dict, key)) {
        logs[i] = PyDict_GetItem(log_dict, key);
        vars[i] = (Real*)var;
        added = 1;
    }
    Py_DECREF(key);
    return added;
}

/*
 * Simulation variables
 *
 */
// Simulation state
int running = 0;    // 1 if a simulation has been initialized, 0 if it's clean

// Input arguments
PyObject *platform_name;// A python string specifying the platform to use
PyObject *device_name;  // A python string specifying the device to use
char* kernel_source;    // The kernel code
int nx;                 // The number of cells in the x direction
int ny;                 // The number of cells in the y direction
double gx;              // The cell-to-cell conductance in the x direction
double gy;              // The cell-to-cell conductance in the y direction
double tmin;            // The initial simulation time
double tmax;            // The final simulation time
double default_dt;      // The default time between steps
PyObject* state_in;     // The initial state
PyObject* state_out;    // The final state
PyObject *protocol;     // A pacing protocol
PyObject *log_dict;     // A logging dict
double log_interval;    // The time between log writes
PyObject *inter_log;    // A list of intermediary variables to log
PyObject *field_data;   // A list containing all field data

// OpenCL objects
cl_context context = NULL;
cl_command_queue command_queue = NULL;
cl_program program = NULL;
cl_kernel kernel_cell;
cl_kernel kernel_diff;
cl_kernel kernel_dif2;
cl_kernel kernel_dif3;
cl_mem mbuf_state = NULL;
cl_mem mbuf_idiff = NULL;
cl_mem mbuf_inter_log = NULL;
cl_mem mbuf_field_data = NULL;
cl_mem mbuf_conn1 = NULL;       // Connections: Cell 1
cl_mem mbuf_conn2 = NULL;       // Connections: Cell 2
cl_mem mbuf_conn3 = NULL;       // Connections: Conductance between 1 and 2

// Input vectors to kernels
Real *rvec_state = NULL;
Real *rvec_idiff = NULL;
Real *rvec_inter_log = NULL;
Real *rvec_field_data = NULL;
int *rvec_conn1 = NULL;
int *rvec_conn2 = NULL;
Real *rvec_conn3 = NULL;
size_t dsize_state;
size_t dsize_idiff;
size_t dsize_inter_log;
size_t dsize_field_data;
size_t dsize_conn1 = NULL;
size_t dsize_conn2 = NULL;
size_t dsize_conn3 = NULL;

// Timing
double engine_time;     // The current simulation time
double dt;              // The next step size
double dt_min;          // The minimal time increase
double tnext_pace;      // The next pacing event start/stop

// Halt on NaN
int halt_sim;

// Pacing
ESys pacing = NULL;
double engine_pace = 0;

// Diffusion currents enabled/disabled
int diffusion;

// Arbitrary geometry diffusion
PyObject* connections;  // List of connection tuples
int n_connections;

// OpenCL work group size
size_t local_work_size[2]; 
// Total number of work items rounded up to a multiple of the local size
size_t global_work_size[2];
// Work items for arbitrary geometry diffusion step
size_t local_work_size_conn[1];
size_t global_work_size_conn[1];

// Kernel arguments copied into "Real" type
Real arg_time;
Real arg_pace;
Real arg_dt;
Real arg_gx;
Real arg_gy;

// Logging
PyObject** logs = NULL; // An array of pointers to a PyObject
Real** vars = NULL;     // An array of pointers to values to log
int n_vars;             // Number of logging variables
double tlog;            // Time of next logging point (for periodic logging)
double tnext_log;       // The next logging point
unsigned long inext_log;// The number of logged steps
int logging_diffusion;  // True if diffusion current is being logged.
int logging_states;     // True if any states are being logged
int logging_inters;     // True if any intermediary variables are being logged.
int n_inter;            // The number of intermediary variables to log
// The relationship between n_inter and logging_inters isn't straightforward:
// n_inter is the number of different model variables (so membrane.V, not
// 1.2.membrane.V) being logged, while logging_inters is 1 if at least one
// simulation variable (1.2.membrane.V) is listed in the given log.
int n_field_data;       // The number of floats in the field data

// Temporary objects: decref before re-using for another var
// (Unless you got it through PyList_GetItem or PyTuble_GetItem)
PyObject* flt = NULL;               // PyFloat, various uses
PyObject* ret = NULL;               // PyFloat, used as return value
PyObject* list_update_str = NULL;   // PyString, used to call "append" method

/*
 * Cleans up after a simulation
 *
 */
static PyObject*
sim_clean()
{
    #ifdef MYOKIT_DEBUG
    printf("Clean called.\n");
    #endif

    if(running) {
        #ifdef MYOKIT_DEBUG
        printf("Cleaning.\n");
        #endif
        
        // Wait for any remaining commands to finish        
        clFlush(command_queue);
        clFinish(command_queue);
    
        // Decref opencl objects
        clReleaseKernel(kernel_cell); kernel_cell = NULL;
        if (diffusion) {
            if (connections == Py_None) {    
                clReleaseKernel(kernel_diff); kernel_diff = NULL;
            } else {
                clReleaseKernel(kernel_dif2); kernel_dif2 = NULL;
                clReleaseKernel(kernel_dif3); kernel_dif3 = NULL;
            }
        }
        clReleaseProgram(program); program = NULL;
        clReleaseMemObject(mbuf_state); mbuf_state = NULL;
        clReleaseMemObject(mbuf_idiff); mbuf_idiff = NULL;
        clReleaseMemObject(mbuf_inter_log); mbuf_inter_log = NULL;
        clReleaseMemObject(mbuf_field_data); mbuf_field_data = NULL;
        if (diffusion && (connections != Py_None)) {
            clReleaseMemObject(mbuf_conn1); mbuf_conn1 = NULL;
            clReleaseMemObject(mbuf_conn2); mbuf_conn2 = NULL;
            clReleaseMemObject(mbuf_conn3); mbuf_conn3 = NULL;
        }
        clReleaseCommandQueue(command_queue); command_queue = NULL;
        clReleaseContext(context); context = NULL;
        
        // Free pacing system memory
        ESys_Destroy(pacing); pacing = NULL;
        
        // Free dynamically allocated arrays
        free(rvec_state); rvec_state = NULL;
        free(rvec_idiff); rvec_idiff = NULL;
        free(rvec_inter_log); rvec_inter_log = NULL;
        free(rvec_field_data); rvec_field_data = NULL;
        free(rvec_conn1); rvec_conn1 = NULL;
        free(rvec_conn2); rvec_conn2 = NULL;
        free(rvec_conn3); rvec_conn3 = NULL;
        free(logs); logs = NULL;
        free(vars); vars = NULL;
        
        // No longer need update string
        Py_XDECREF(list_update_str); list_update_str = NULL;

        // No longer running
        running = 0;
    }
    #ifdef MYOKIT_DEBUG
    else
    {
        printf("Skipping cleaning: not running!\n");
    }
    #endif
    
    // Return 0, allowing the construct
    //  PyErr_SetString(PyExc_Exception, "Oh noes!");
    //  return sim_clean()
    //to terminate a python function.
    return 0;
}
static PyObject*
py_sim_clean()
{
    #ifdef MYOKIT_DEBUG
    printf("Python py_sim_clean called.\n");
    #endif

    sim_clean();
    Py_RETURN_NONE;
}

/*
 * Sets up a simulation
 *
 *
 */
static PyObject*
sim_init(PyObject* self, PyObject* args)
{
    #ifdef MYOKIT_DEBUG
    // Don't buffer stdout
    setbuf(stdout, NULL); // Don't buffer stdout
    printf("Starting initialization.\n");
    #endif
    
    // Check if already running
    if(running != 0) {
        PyErr_SetString(PyExc_Exception, "Simulation already initialized.");
        return 0;
    }
    
    // Set all pointers used in sim_clean to null
    command_queue = NULL;
    kernel_cell = NULL;
    kernel_diff = NULL;
    kernel_dif2 = NULL;
    kernel_dif3 = NULL;
    program = NULL;
    mbuf_state = NULL;
    mbuf_idiff = NULL;
    mbuf_inter_log = NULL;
    mbuf_field_data = NULL;
    mbuf_conn1 = NULL;
    mbuf_conn2 = NULL;
    mbuf_conn3 = NULL;
    context = NULL;
    pacing = NULL;
    rvec_state = NULL;
    rvec_idiff = NULL;
    rvec_inter_log = NULL;
    rvec_field_data = NULL;
    rvec_conn1 = NULL;
    rvec_conn2 = NULL;
    rvec_conn3 = NULL;
    logs = NULL;
    vars = NULL;
    list_update_str = NULL;

    // Check input arguments
    if(!PyArg_ParseTuple(args, "OOsiibddOdddOOOOdOO",
            &platform_name,
            &device_name,
            &kernel_source,
            &nx,
            &ny,
            &diffusion,
            &gx,
            &gy,
            &connections,
            &tmin,
            &tmax,
            &default_dt,
            &state_in,
            &state_out,
            &protocol,
            &log_dict,
            &log_interval,
            &inter_log,
            &field_data
            )) {
        PyErr_SetString(PyExc_Exception, "Wrong number of arguments.");
        // Nothing allocated yet, no pyobjects _created_, return directly
        return 0;
    }
    dt = default_dt;
    dt_min = dt * 1e-2;
    arg_dt = (Real)dt;
    arg_gx = (Real)gx;
    arg_gy = (Real)gy;
    halt_sim = 0;
    
    #ifdef MYOKIT_DEBUG
    printf("Retrieved function arguments.\n");
    #endif
    
    // Now officialy running :)
    running = 1;
    
    ///////////////////////////////////////////////////////////////////////////
    //
    // From this point on, use "return sim_clean()" to abort.
    //
    //
    
    #ifdef MYOKIT_DEBUG
    printf("Running!\n");
    printf("Checking input arguments.\n");
    #endif
    
    int i, j, k;
    
    //
    // Check state in and out lists 
    //
    if(!PyList_Check(state_in)) {
        PyErr_SetString(PyExc_Exception, "'state_in' must be a list.");
        return sim_clean();
    }
    if(PyList_Size(state_in) != nx * ny * n_state) {
        PyErr_SetString(PyExc_Exception, "'state_in' must have size nx * ny * n_states.");
        return sim_clean();
    }
    if(!PyList_Check(state_out)) {
        PyErr_SetString(PyExc_Exception, "'state_out' must be a list.");
        return sim_clean();
    }
    if(PyList_Size(state_out) != nx * ny * n_state) {
        PyErr_SetString(PyExc_Exception, "'state_out' must have size nx * ny * n_states.");
        return sim_clean();
    }
    
    //
    // Check inter_log list of intermediary variables to log
    //
    if(!PyList_Check(inter_log)) {
        PyErr_SetString(PyExc_Exception, "'inter_log' must be a list.");
        return sim_clean();
    }
    n_inter = PyList_Size(inter_log);
    
    //
    // Check field data
    //
    if(!PyList_Check(field_data)) {
        PyErr_SetString(PyExc_Exception, "'field_data' must be a list.");
        return sim_clean();
    }
    n_field_data = PyList_Size(field_data);

    //
    // Set up pacing system
    //
    ESys_Flag flag_pacing;
    pacing = ESys_Create(&flag_pacing);
    if(flag_pacing!=ESys_OK) { ESys_SetPyErr(flag_pacing); return sim_clean(); }
    flag_pacing = ESys_Populate(pacing, protocol);
    if(flag_pacing!=ESys_OK) { ESys_SetPyErr(flag_pacing); return sim_clean(); }
    flag_pacing = ESys_AdvanceTime(pacing, tmin, tmax);
    if(flag_pacing!=ESys_OK) { ESys_SetPyErr(flag_pacing); return sim_clean(); }
    tnext_pace = ESys_GetNextTime(pacing, NULL);
    engine_pace = ESys_GetLevel(pacing, NULL);
    arg_pace = (Real)engine_pace;
    
    //
    // Set simulation starting time 
    //
    engine_time = tmin;
    arg_time = (Real)engine_time;

    //
    // Create opencl environment
    //
    
    #ifdef MYOKIT_DEBUG
    printf("Creating vectors.\n");
    #endif
    
    // Create state vector, set initial values
    dsize_state = nx*ny*n_state * sizeof(Real);
    rvec_state = (Real*)malloc(dsize_state);
    for(i=0; i<nx*ny*n_state; i++) {
        flt = PyList_GetItem(state_in, i);    // Don't decref! 
        if(!PyFloat_Check(flt)) {
            char errstr[200];
            sprintf(errstr, "Item %d in state vector is not a float.", i);
            PyErr_SetString(PyExc_Exception, errstr);
            return sim_clean();
        }
        rvec_state[i] = (Real)PyFloat_AsDouble(flt);
    }

    // Create diffusion current vector
    if (diffusion) {
        dsize_idiff = nx*ny * sizeof(Real);
        rvec_idiff = (Real*)malloc(dsize_idiff);
        for(i=0; i<nx*ny; i++) rvec_idiff[i] = 0.0;    
    } else {
        dsize_idiff = sizeof(Real);
        rvec_idiff = (Real*)malloc(dsize_idiff);
    }
    
    // Create vector of intermediary variables to log
    if(n_inter) {
        dsize_inter_log = nx*ny*n_inter * sizeof(Real);
        rvec_inter_log = (Real*)malloc(dsize_inter_log);
        for(i=0; i<nx*ny*n_inter; i++) rvec_inter_log[i] = 0.0;
    } else {
        dsize_inter_log = sizeof(Real);
        rvec_inter_log = (Real*)malloc(dsize_inter_log);
    }
    
    // Create vector of field data
    if(n_field_data) {
        dsize_field_data = n_field_data * sizeof(Real);
        rvec_field_data = (Real*)malloc(dsize_field_data);
        for(i=0; i<n_field_data; i++) {
            flt = PyList_GetItem(field_data, i);    // No need to decref
            if(!PyFloat_Check(flt)) {
                char errstr[200];
                sprintf(errstr, "Item %d in field data is not a float.", i);
                PyErr_SetString(PyExc_Exception, errstr);
                return sim_clean();
            }
            rvec_field_data[i] = (Real)PyFloat_AsDouble(flt);
        }
    } else {
        dsize_field_data = sizeof(Real);
        rvec_field_data = (Real*)malloc(dsize_field_data);
    }
    
    // Set up arbitrary-geometry diffusion
    if(connections != Py_None) { // Actually preferred way of checking!
        if(!PyList_Check(connections)) {
            PyErr_SetString(PyExc_Exception, "Connections should be None or a list");
            return sim_clean();
        }
        n_connections = PyList_Size(connections);
        dsize_conn1 = n_connections * sizeof(int);
        dsize_conn2 = dsize_conn1;
        dsize_conn3 = n_connections * sizeof(Real);
        rvec_conn1 = (int*)malloc(dsize_conn1);
        rvec_conn2 = (int*)malloc(dsize_conn2);
        rvec_conn3 = (Real*)malloc(dsize_conn3);        
        for(i=0; i<n_connections; i++) {
            flt = PyList_GetItem(connections, i);   // Borrowed reference
            if(!PyTuple_Check(flt)) {
                PyErr_SetString(PyExc_Exception, "Connections list must contain all tuples");
                return sim_clean();
            }
            if(PyTuple_Size(flt) != 3) {
                PyErr_SetString(PyExc_Exception, "Connections list must contain only 3-tuples");
                return sim_clean();
            }
            ret = PyTuple_GetItem(flt, 0);  // Borrowed reference
            if(!PyInt_Check(ret)) {
                PyErr_SetString(PyExc_Exception, "First item in each connection tuple must be int");
                return sim_clean();
            }
            rvec_conn1[i] = (int)PyInt_AsLong(ret);
            ret = PyTuple_GetItem(flt, 1);  // Borrowed reference
            if(!PyInt_Check(ret)) {
                PyErr_SetString(PyExc_Exception, "Second item in each connection tuple must be int");
                return sim_clean();
            }
            rvec_conn2[i] = (int)PyInt_AsLong(ret);
            ret = PyTuple_GetItem(flt, 2);  // Borrowed reference
            if(!PyFloat_Check(ret)) {
                PyErr_SetString(PyExc_Exception, "Third item in each connection tuple must be float");
                return sim_clean();
            }
            rvec_conn3[i] = (Real)PyFloat_AsDouble(ret);
        }
    }
        
    #ifdef MYOKIT_DEBUG
    printf("Created vectors.\n");
    #endif

    #ifdef MYOKIT_DEBUG
    printf("Setting work group sizes.\n");
    #endif
    // Work group size and total number of items
    local_work_size[0] = 8;    
    local_work_size[1] = (ny > 1) ? 8 : 1;
    global_work_size[0] = mcl_round_total_size(local_work_size[0], nx);
    global_work_size[1] = (ny == 1) ? 1 : mcl_round_total_size(local_work_size[1], ny);
    if (connections != Py_None) {
        local_work_size_conn[0] = local_work_size[0];
        global_work_size_conn[0] = mcl_round_total_size(local_work_size_conn[0], n_connections);
    }
    #ifdef MYOKIT_DEBUG
    printf("Work group sizes determined.\n");
    #endif
    
    // Get platform and device id
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    if (mcl_select_device(platform_name, device_name, &platform_id, &device_id)) {
        // Error message set by mcl_select_device
        return sim_clean();
    }
    #ifdef MYOKIT_DEBUG
    printf("Selected platform and device id.\n");
    if (platform_id == NULL) {
        printf("No preferred platform set.\n");
    } else {
        printf("Preferred platform set.\n");
    }
    if (device_id == NULL) {
        printf("No preferred device set.\n");
    } else {
        printf("Preferred device set.\n");
    }
    #endif

    // Create a context and command queue
    #ifdef MYOKIT_DEBUG
    printf("Attempting to create OpenCL context...\n");
    #endif
    cl_int flag;
    if (platform_id != NULL) {
        #ifdef MYOKIT_DEBUG
        printf("Creating context with context_properties\n");
        #endif
        cl_context_properties context_properties[] = 
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
        context = clCreateContext(context_properties, 1, &device_id, NULL, NULL, &flag);
    } else {
        context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &flag);
    }
    if(mcl_flag(flag)) return sim_clean();
    #ifdef MYOKIT_DEBUG
    printf("Created context.\n");
    #endif

    // Create command queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &flag);
    if(mcl_flag(flag)) return sim_clean();
    #ifdef MYOKIT_DEBUG
    printf("Created command queue.\n");
    #endif  
        
    // Create memory buffers on the device
    mbuf_state = clCreateBuffer(context, CL_MEM_READ_WRITE, dsize_state, NULL, &flag);
    if(mcl_flag(flag)) return sim_clean();
    mbuf_idiff = clCreateBuffer(context, CL_MEM_READ_WRITE, dsize_idiff, NULL, &flag);
    if(mcl_flag(flag)) return sim_clean();
    mbuf_inter_log = clCreateBuffer(context, CL_MEM_READ_WRITE, dsize_inter_log, NULL, &flag);
    if(mcl_flag(flag)) return sim_clean();
    mbuf_field_data = clCreateBuffer(context, CL_MEM_READ_ONLY, dsize_field_data, NULL, &flag);
    if(mcl_flag(flag)) return sim_clean();
    if(connections != Py_None) {
        mbuf_conn1 = clCreateBuffer(context, CL_MEM_READ_ONLY, dsize_conn1, NULL, &flag);
        if(mcl_flag(flag)) return sim_clean();
        mbuf_conn2 = clCreateBuffer(context, CL_MEM_READ_ONLY, dsize_conn2, NULL, &flag);
        if(mcl_flag(flag)) return sim_clean();
        mbuf_conn3 = clCreateBuffer(context, CL_MEM_READ_ONLY, dsize_conn3, NULL, &flag);
        if(mcl_flag(flag)) return sim_clean();
    }
    
    #ifdef MYOKIT_DEBUG
    printf("Created buffers.\n");
    printf("State buffer size:%d.\n", dsize_state);
    printf("Idiff buffer size:%d.\n", dsize_idiff);
    printf("Inter-log buffer size:%d.\n", dsize_inter_log);
    printf("Field-data buffer size:%d.\n", dsize_field_data);
    printf("Connections-1 buffer size:%d.\n", dsize_conn1);
    printf("Connections-2 buffer size:%d.\n", dsize_conn2);
    printf("Connections-3 buffer size:%d.\n", dsize_conn3);
    #endif  

    // Copy data into buffers
    flag = clEnqueueWriteBuffer(command_queue, mbuf_state, CL_TRUE, 0, dsize_state, rvec_state, 0, NULL, NULL);
    if(mcl_flag(flag)) return sim_clean();
    flag = clEnqueueWriteBuffer(command_queue, mbuf_idiff, CL_TRUE, 0, dsize_idiff, rvec_idiff, 0, NULL, NULL);
    if(mcl_flag(flag)) return sim_clean();
    flag = clEnqueueWriteBuffer(command_queue, mbuf_inter_log, CL_TRUE, 0, dsize_inter_log, rvec_inter_log, 0, NULL, NULL);
    if(mcl_flag(flag)) return sim_clean();
    flag = clEnqueueWriteBuffer(command_queue, mbuf_field_data, CL_TRUE, 0, dsize_field_data, rvec_field_data, 0, NULL, NULL);
    if(mcl_flag(flag)) return sim_clean();
    if(connections != Py_None) {
        flag = clEnqueueWriteBuffer(command_queue, mbuf_conn1, CL_TRUE, 0, dsize_conn1, rvec_conn1, 0, NULL, NULL);
        if(mcl_flag(flag)) return sim_clean();
        flag = clEnqueueWriteBuffer(command_queue, mbuf_conn2, CL_TRUE, 0, dsize_conn2, rvec_conn2, 0, NULL, NULL);
        if(mcl_flag(flag)) return sim_clean();
        flag = clEnqueueWriteBuffer(command_queue, mbuf_conn3, CL_TRUE, 0, dsize_conn3, rvec_conn3, 0, NULL, NULL);
        if(mcl_flag(flag)) return sim_clean();
    }
    #ifdef MYOKIT_DEBUG
    printf("Enqueued copying of data into buffers.\n");
    #endif
    
    // Wait for copy to be finished
    clFlush(command_queue);
    clFinish(command_queue);
    #ifdef MYOKIT_DEBUG
    printf("Command queue flushed.\n");
    #endif
    
    // Load and compile the program
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, &flag);
    if(mcl_flag(flag)) return sim_clean();
    #ifdef MYOKIT_DEBUG
    printf("Program created.\n");
    #endif
    const char options[] = "";
    //const char options[] = "-w"; // Suppress warnings
    flag = clBuildProgram(program, 1, &device_id, options, NULL, NULL);
    if(flag == CL_BUILD_PROGRAM_FAILURE) {
        // Build failed, extract log
        size_t blog_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &blog_size);
        char *blog = (char*)malloc(blog_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, blog_size, blog, NULL);
        fprintf(stderr, "OpenCL Error: Kernel failed to compile.\n");
        fprintf(stderr, "----------------------------------------");
        fprintf(stderr, "---------------------------------------\n");
        fprintf(stderr, "%s\n", blog);
        fprintf(stderr, "----------------------------------------");
        fprintf(stderr, "---------------------------------------\n");
    }
    if(mcl_flag(flag)) return sim_clean();
    #ifdef MYOKIT_DEBUG
    printf("Program built.\n");
    #endif
    
    // Create the kernels
    kernel_cell = clCreateKernel(program, "cell_step", &flag);
    if(mcl_flag(flag)) return sim_clean();
    if (diffusion) {
        if(connections == Py_None) {
            // Rectangular grid
            kernel_diff = clCreateKernel(program, "diff_step", &flag);
            if(mcl_flag(flag)) return sim_clean();
        } else {
            // Arbitrary geometry
            kernel_dif2 = clCreateKernel(program, "diff_arb_reset", &flag);
            if(mcl_flag(flag)) return sim_clean();
            kernel_dif3 = clCreateKernel(program, "diff_arb_step", &flag);
            if(mcl_flag(flag)) return sim_clean();
        }
    }
    #ifdef MYOKIT_DEBUG
    printf("Kernels created.\n");
    #endif

    // Pass arguments into kernels
    i = 0;
    if(mcl_flag(clSetKernelArg(kernel_cell, i++, sizeof(nx), &nx))) return sim_clean();
    if(mcl_flag(clSetKernelArg(kernel_cell, i++, sizeof(ny), &ny))) return sim_clean();
    if(mcl_flag(clSetKernelArg(kernel_cell, i++, sizeof(arg_time), &arg_time))) return sim_clean();
    if(mcl_flag(clSetKernelArg(kernel_cell, i++, sizeof(arg_dt), &arg_dt))) return sim_clean();
    if(mcl_flag(clSetKernelArg(kernel_cell, i++, sizeof(arg_pace), &arg_pace))) return sim_clean();
    if(mcl_flag(clSetKernelArg(kernel_cell, i++, sizeof(mbuf_state), &mbuf_state))) return sim_clean();
    if(mcl_flag(clSetKernelArg(kernel_cell, i++, sizeof(mbuf_idiff), &mbuf_idiff))) return sim_clean();
    if(mcl_flag(clSetKernelArg(kernel_cell, i++, sizeof(mbuf_inter_log), &mbuf_inter_log))) return sim_clean();
    if(mcl_flag(clSetKernelArg(kernel_cell, i++, sizeof(mbuf_field_data), &mbuf_field_data))) return sim_clean();
    
    if (diffusion) {
        // Calculate initial diffusion current 
        if(connections == Py_None) {
            // Rectangular diffusion
            i = 0;
            if(mcl_flag(clSetKernelArg(kernel_diff, i++, sizeof(nx), &nx))) return sim_clean();
            if(mcl_flag(clSetKernelArg(kernel_diff, i++, sizeof(ny), &ny))) return sim_clean();
            if(mcl_flag(clSetKernelArg(kernel_diff, i++, sizeof(arg_gx), &arg_gx))) return sim_clean();
            if(mcl_flag(clSetKernelArg(kernel_diff, i++, sizeof(arg_gy), &arg_gy))) return sim_clean();
            if(mcl_flag(clSetKernelArg(kernel_diff, i++, sizeof(mbuf_state), &mbuf_state))) return sim_clean();
            if(mcl_flag(clSetKernelArg(kernel_diff, i++, sizeof(mbuf_idiff), &mbuf_idiff))) return sim_clean();
            if(mcl_flag(clEnqueueNDRangeKernel(command_queue, kernel_diff, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL))) return sim_clean();
        } else {
            // Arbitrary geometry
            i = 0;
            if(mcl_flag(clSetKernelArg(kernel_dif2, i++, sizeof(nx), &nx))) return sim_clean();
            if(mcl_flag(clSetKernelArg(kernel_dif2, i++, sizeof(mbuf_idiff), &mbuf_idiff))) return sim_clean();
            if(mcl_flag(clEnqueueNDRangeKernel(command_queue, kernel_dif2, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL))) return sim_clean();
            i = 0;
            if(mcl_flag(clSetKernelArg(kernel_dif3, i++, sizeof(n_connections), &n_connections))) return sim_clean();
            if(mcl_flag(clSetKernelArg(kernel_dif3, i++, sizeof(mbuf_conn1), &mbuf_conn1))) return sim_clean();
            if(mcl_flag(clSetKernelArg(kernel_dif3, i++, sizeof(mbuf_conn2), &mbuf_conn2))) return sim_clean();
            if(mcl_flag(clSetKernelArg(kernel_dif3, i++, sizeof(mbuf_conn3), &mbuf_conn3))) return sim_clean();
            if(mcl_flag(clSetKernelArg(kernel_dif3, i++, sizeof(mbuf_state), &mbuf_state))) return sim_clean();
            if(mcl_flag(clSetKernelArg(kernel_dif3, i++, sizeof(mbuf_idiff), &mbuf_idiff))) return sim_clean();
            if(mcl_flag(clEnqueueNDRangeKernel(command_queue, kernel_dif3, 1, NULL, global_work_size_conn, local_work_size_conn, 0, NULL, NULL))) return sim_clean();
        }
    }
    
    #ifdef MYOKIT_DEBUG
    printf("Arguments passed into kernels.\n");
    #endif
    
    //
    // Set up logging system
    //

    if(!PyDict_Check(log_dict)) {
        PyErr_SetString(PyExc_Exception, "Log argument must be a dict.");
        return sim_clean();
    }
    n_vars = PyDict_Size(log_dict);
    #ifdef MYOKIT_DEBUG
    printf("Number of variables to log:%d.\n", n_vars);
    #endif
    logs = (PyObject**)malloc(sizeof(PyObject*)*n_vars); // Pointers to logging lists 
    #ifdef MYOKIT_DEBUG
    printf("Allocated log pointers:.\n");
    #endif
    vars = (Real**)malloc(sizeof(Real*)*n_vars); // Pointers to variables to log 
    #ifdef MYOKIT_DEBUG
    printf("Allocated var pointers.\n");
    #endif

    char log_var_name[1023];    // Variable names
    int k_vars = 0;             // Counting number of variables in log

    // Time and pace are set globally
<?
var = model.binding('time')
print(tab + 'k_vars += log_add(log_dict, logs, vars, k_vars, "' + var.qname() + '", &arg_time);')
var = model.binding('pace')
if var is not None:
    print(tab + 'k_vars += log_add(log_dict, logs, vars, k_vars, "' + var.qname() + '", &arg_pace);')
?>

    // Diffusion current
    logging_diffusion = 0;
    for(i=0; i<ny; i++) {
        for(j=0; j<nx; j++) {
<?
var = model.binding('diffusion_current')
if var is not None:
    if dims == 1:
        print(3*tab + 'sprintf(log_var_name, "%d.' + var.qname() + '", j);')
    else:
        print(3*tab + 'sprintf(log_var_name, "%d.%d.' + var.qname() + '", j, i);')
    print(3*tab + 'if(log_add(log_dict, logs, vars, k_vars, log_var_name, &rvec_idiff[i*nx+j])) {')
    print(4*tab + 'logging_diffusion = 1;')
    print(4*tab + 'k_vars++;')
    print(3*tab + '}')
?>
        }
    }
    
    // States
    logging_states = 0;
    for(i=0; i<ny; i++) {
        for(j=0; j<nx; j++) {
<?
for var in model.states():
    if dims == 1:
        print(3*tab + 'sprintf(log_var_name, "%d.' + var.qname() + '", j);')
    else:
        print(3*tab + 'sprintf(log_var_name, "%d.%d.' + var.qname() + '", j, i);' )
    print(3*tab + 'if(log_add(log_dict, logs, vars, k_vars, log_var_name, &rvec_state[(i*nx+j)*n_state+' + str(var.indice()) + '])) {')
    print(4*tab + 'logging_states = 1;')
    print(4*tab + 'k_vars++;')
    print(3*tab + '}')
?>
        }
    }
    
    // Intermediary variables
    logging_inters = 0;
    for(i=0; i<ny; i++) {
        for(j=0; j<nx; j++) {
            for(k=0; k<n_inter; k++) {
                ret = PyList_GetItem(inter_log, k); // Don't decref
<?
if dims == 1:
    print(4*tab + 'sprintf(log_var_name, "%d.%s", j, PyString_AsString(ret));')
else:
    print(4*tab + 'sprintf(log_var_name, "%d.%d.%s", j, i, PyString_AsString(ret));')

print(4*tab + 'if(log_add(log_dict, logs, vars, k_vars, log_var_name, &rvec_inter_log[(i*nx+j)*n_inter+k])) {')
print(5*tab + 'logging_inters = 1;')
print(5*tab + 'k_vars++;')
print(4*tab + '}')
?>
            }
        }
    }
    ret = NULL;

    // Check if log contained extra variables 
    if(k_vars != n_vars) {
        PyErr_SetString(PyExc_Exception, "Unknown variables found in logging dictionary.");
        return sim_clean();
    }
    
    #ifdef MYOKIT_DEBUG
    printf("Created log for %d variables.\n", n_vars);
    #endif  

    // Log update method:
    list_update_str = PyString_FromString("append");
    
    // Store initial position in logs
    // Skipping!
    
    // Next logging position: current time
    inext_log = 0;
    tnext_log = tmin;

    //
    // Done!
    //
    #ifdef MYOKIT_DEBUG
    printf("Finished initialization.\n");
    #endif    
    Py_RETURN_NONE;
}
    
/*
 * Takes the next steps in a simulation run
 */
static PyObject*
sim_step(PyObject *self, PyObject *args)
{
    ESys_Flag flag_pacing;
    long steps_left_in_run = 500 + 200000 / (nx * ny);
    if(steps_left_in_run < 1000) steps_left_in_run = 1000;
    cl_int flag;
    int i;
    double d = 0;
    
    while(1) {
    
        // Determine next timestep
        // Ensure next pacing event is simulated
        dt = default_dt;
        d = tmax - engine_time; if(d > dt_min && d < dt) dt = d;
        d = tnext_pace - engine_time; if(d > dt_min && d < dt) dt = d;
        d = tnext_log - engine_time; if(d > dt_min && d < dt) dt = d;
        arg_dt = (Real)dt;
    
        // Update states, advancing them to t+dt
        if(mcl_flag(clSetKernelArg(kernel_cell, 2, sizeof(Real), &arg_time))) return sim_clean();
        if(mcl_flag(clSetKernelArg(kernel_cell, 3, sizeof(Real), &arg_dt))) return sim_clean();
        if(mcl_flag(clSetKernelArg(kernel_cell, 4, sizeof(Real), &arg_pace))) return sim_clean();
        if(mcl_flag(clEnqueueNDRangeKernel(command_queue, kernel_cell, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL))) return sim_clean();
        
        // Update time, advancing it to t+dt
        engine_time += dt;
        arg_time = (Real)engine_time;
        
        // Advance pacing mechanism, advancing it to t+dt
        flag_pacing = ESys_AdvanceTime(pacing, engine_time, tmax);
        if (flag_pacing!=ESys_OK) { ESys_SetPyErr(flag_pacing); return sim_clean(); }
        tnext_pace = ESys_GetNextTime(pacing, NULL);
        engine_pace = ESys_GetLevel(pacing, NULL);
        arg_pace = (Real)engine_pace;
        
        // Update diffusion current, calculating it for time t+dt
        if (diffusion) {
            // Calculate initial diffusion current 
            if(connections == Py_None) {
                // Rectangular diffusion
                if(mcl_flag(clEnqueueNDRangeKernel(command_queue, kernel_diff, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL))) return sim_clean();
            } else {
                // Arbitrary geometry
                if(mcl_flag(clEnqueueNDRangeKernel(command_queue, kernel_dif2, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL))) return sim_clean();
                if(mcl_flag(clEnqueueNDRangeKernel(command_queue, kernel_dif3, 1, NULL, global_work_size_conn, local_work_size_conn, 0, NULL, NULL))) return sim_clean();
            }
        }
        
        // Check if we're finished
        // Do this before logging, to ensure we don't log the final time position!
        // Logging with fixed time steps should always be half-open: including the
        // first but not the last point in time.
        if(engine_time >= tmax || halt_sim) break;
        
        // Log new situation at t+dt
        // Note: states, time, pacing and diffusion are now at t+dt, however,
        // intermediary variables are still at time t.
        if(engine_time >= tnext_log) {
            if(logging_diffusion) {
                flag = clEnqueueReadBuffer(command_queue, mbuf_idiff, CL_TRUE, 0, dsize_idiff, rvec_idiff, 0, NULL, NULL);
                if(mcl_flag(flag)) return sim_clean();
            }
            if(logging_states) {
                flag = clEnqueueReadBuffer(command_queue, mbuf_state, CL_TRUE, 0, dsize_state, rvec_state, 0, NULL, NULL);
                if(mcl_flag(flag)) return sim_clean();
                if(isnan(rvec_state[0])) {
                    halt_sim = 1;
                }
            }
            if(logging_inters) {
                flag = clEnqueueReadBuffer(command_queue, mbuf_inter_log, CL_TRUE, 0, dsize_inter_log, rvec_inter_log, 0, NULL, NULL);
                if(mcl_flag(flag)) return sim_clean();
            }
            for(i=0; i<n_vars; i++) {
                flt = PyFloat_FromDouble(*vars[i]);
                ret = PyObject_CallMethodObjArgs(logs[i], list_update_str, flt, NULL);
                Py_DECREF(flt); flt = NULL;
                Py_XDECREF(ret);
                if(ret == NULL) {
                    PyErr_SetString(PyExc_Exception, "Call to append() failed on logging list.");
                    return sim_clean();
                }
            }
            ret = NULL;
            
            // Set next logging point
            inext_log++;
            tnext_log = tmin + (double)inext_log * log_interval;
            if (inext_log == 0) {
                // Unsigned int wraps around instead of overflowing, becomes zero again
                PyErr_SetString(PyExc_Exception, "Overflow in logged step count: Simulation too long!");
                return sim_clean();
            }
        }
                
        // Report back to python
        if(--steps_left_in_run == 0) {
            // For some reason, this clears memory
            clFlush(command_queue);
            clFinish(command_queue);
            return PyFloat_FromDouble(engine_time);
        }
    }

    #ifdef MYOKIT_DEBUG
    printf("Simulation finished.\n");
    #endif

    // Set final state
    flag = clEnqueueReadBuffer(command_queue, mbuf_state, CL_TRUE, 0, dsize_state, rvec_state, 0, NULL, NULL);
    if(mcl_flag(flag)) return sim_clean();
    for(i=0; i<n_state*nx*ny; i++) {
        PyList_SetItem(state_out, i, PyFloat_FromDouble(rvec_state[i]));
        // PyList_SetItem steals a reference: no need to decref the double!
    }
    
    #ifdef MYOKIT_DEBUG
    printf("Final state copied.\n");
    printf("Tyding up...\n");
    #endif  

    // Finish any remaining commands (shouldn't happen)
    clFlush(command_queue);
    clFinish(command_queue);

    sim_clean();    // Ignore return value
    
    if (halt_sim) {
        #ifdef MYOKIT_DEBUG
        printf("Finished tidiying up, ending simulation with nan.\n");
        #endif
        PyErr_SetString(PyExc_ArithmeticError, "Encountered nan in simulation.");
        return 0;
    } else {
        #ifdef MYOKIT_DEBUG
        printf("Finished tidiying up, ending simulation.\n");
        #endif  
        return PyFloat_FromDouble(engine_time);
    }
}

/*
 * Methods in this module
 */
static PyMethodDef SimMethods[] = {
    {"sim_init", sim_init, METH_VARARGS, "Initialize the simulation."},
    {"sim_step", sim_step, METH_VARARGS, "Perform the next step in the simulation."},
    {"sim_clean", py_sim_clean, METH_VARARGS, "Clean up after an aborted simulation."},
    {NULL},
};

/*
 * Module definition
 */
PyMODINIT_FUNC
init<?=module_name?>(void) {
    (void) Py_InitModule("<?= module_name ?>", SimMethods);
}
