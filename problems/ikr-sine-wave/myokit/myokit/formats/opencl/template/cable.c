<?
#
# cable.c
#
# A pype template for an OpenCL host file driving a cable simulation
#
# Required variables
# -------------------------------------------------------
# model       A model
# native_math True or False
# precision   A myokit precision constant
# -------------------------------------------------------
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
from myokit.formats import opencl

# Tab
tab = '    '

?>/*

OpenCL Host file for a cable simulation using <?= model.name() ?>

Generated on <?= myokit.date() ?> by myokit opencl export

*/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* Load the opencl libraries. */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/*
Note: Braces create local context for variables, but this is rendered as a
"compound statement, so that constructions like "if(ocl(expr)) {}" won't
compile. A do-while(0) block is expanded into a regular statement, which is
allowed in these contexts.
*/
#define ocl(_x) \
do { \
    cl_int _e = _x; \
    if (_e != CL_SUCCESS) { \
        ocl_print_error(_e, #_x); \
        abort(); \
    } \
} while(0)
# define ocl_check(_x, _e) \
do { \
    if (_e != CL_SUCCESS) { \
        ocl_print_error(_e, _x); \
        abort(); \
    } \
} while(0)

/* Show debug output */
//#define __MYOKIT_DEBUG
<?
print('#define n_state ' + str(model.count_states()))
print('')

if precision == myokit.SINGLE_PRECISION:
    print('/* Using single precision floats */')
    print('typedef float Real;')
else:
    print('/* Using double precision floats */')
    print('typedef double Real;')
?>
/*
 * Prints an opencl error
 */ 
void ocl_print_error(const int e, const char* s)
{
    switch(e) {
        case CL_SUCCESS:
            break;
        case CL_INVALID_KERNEL:
            fprintf(stderr, "OpenCL error: '%s' returned CL_INVALID_KERNEL\n", s);
            break;
        case CL_INVALID_ARG_INDEX:
            fprintf(stderr, "OpenCL error: '%s' returned CL_INVALID_ARG_INDEX\n", s);
            break;
        case CL_INVALID_ARG_VALUE:
            fprintf(stderr, "OpenCL error: '%s' returned CL_INVALID_ARG_VALUE\n", s);
            break;
        case CL_INVALID_MEM_OBJECT:
            fprintf(stderr, "OpenCL error: '%s' returned CL_INVALID_MEM_OBJECT\n", s);
            break;
        case CL_INVALID_SAMPLER:
            fprintf(stderr, "OpenCL error: '%s' returned CL_INVALID_SAMPLER\n", s);
            break;
        case CL_INVALID_ARG_SIZE:
            fprintf(stderr, "OpenCL error: '%s' returned CL_INVALID_ARG_SIZE\n", s);
            break;
        default:
            fprintf(stderr, "OpenCL error: '%s' returned %d!\n", s, e);
            break;
    };
}

/*
 * Finds the device id of a GPU or, failing that, a CPU.
 */
cl_device_id get_device_id()
{
    /* Get platforms */
    cl_platform_id platforms[100];  /* Array of ids (to be returned) */
	cl_uint platforms_n = 0;        /* Number of platforms */
	ocl(clGetPlatformIDs(100, platforms, &platforms_n));
    /* Scan for GPU devices */
    cl_device_id devices[100];
    cl_uint devices_n = 0;
	int i;
	cl_int err;
	/* Get preferred type */
	for (i=0; i<platforms_n; i++) {		
	    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 100, devices, &devices_n);
	    if(err == CL_SUCCESS) {
	        return devices[0];
        } else if (err != CL_DEVICE_NOT_FOUND) {
            ocl_check("clGetDeviceIDs", err);
        }
    }
    /* Get regardless of type */
    for (i=0; i<platforms_n; i++) {
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 100, devices, &devices_n);
	    if(err == CL_SUCCESS) {
	        return devices[0];
        } else if (err != CL_DEVICE_NOT_FOUND) {
            ocl_check("clGetDeviceIDs", err);
        }        
	}
	printf("OpenCL Error: Unable to detect an opencl device.\n");
	abort();
}

/*
 * Rounds up to the nearest multiple of ws_size.
 */
static int round_total_size(const int ws_size, const int total_size) 
{
    int size = (total_size / ws_size) * ws_size;
    if(size < total_size) size += ws_size;
    return size;
}

/*
 * Reads a kernel from a file
 */
char* read_kernel(const char* fname)
{
    /* Open file */
    FILE *fp = fopen(fname, "r");
    if (!fp) {
        fprintf(stderr, "Error: Unable to read kernel file.\n");
        abort();
    }
    
    /* Get length */
    fseek(fp, 0L, SEEK_END);
    long n_chars = ftell(fp);
    rewind(fp);

    /* Allocate enough space, initialized to zero */
    /* Extra byte stores the terminating zero. */
    char* text = (char*)calloc(n_chars + 1, sizeof(char));
    if (text == NULL) {
        fclose(fp);
        fprintf(stderr, "Error: can't allocate enough memory to read file.");
        abort();
    }
    
    /* Read bytes */
    fread(text, n_chars, 1, fp);
    
    /* Close file and return */
    fclose(fp);    
    return text;
}

/*
 * Sets the initial values in the state vector
 */
void set_initial_values(int n_cells, Real* s)
{
    int i;
    for(i=0; i<n_cells; i++) {
<?
for v in model.states():
    print(tab*2 + '*s = ' + myokit.strfloat(v.state_value()) + '; s++; // ' + str(v.qname()))
?>
    }
}   

/*
 * Runs a simulation
 *
 * n_cells      The number of cells (for example 256)
 * time_start   The initial simulation time (for example 0)
 * time_end     The final simulation time (for example 1000)
 * dt           The time step size (for example 0.005)
 * g            The cell-to-cell conductance (for example 1.5)
 */
int run(const int n_cells, const Real time_start, const Real time_end, const Real dt, const Real g)
{   
    /* Steps per log action */
    const int steps_per_log = (int)(1.0 / dt);

    int i;
    Real time = time_start;

    /* Work group size (in this case, one-dimensional) */
    size_t local_work_size  = 64; 
    /* Total number of work items rounded up to a multiple of the local size */
    size_t global_work_size = round_total_size(local_work_size, n_cells);
    
    /* Objects needing destruction */
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel_cell;
    cl_kernel kernel_diff;
    cl_mem mbuf_state = NULL;
    cl_mem mbuf_idiff = NULL;
    Real *state = NULL;
    Real *idiff = NULL;
    
    /* Get device id */
    cl_device_id device_id = get_device_id();
    #ifdef __MYOKIT_DEBUG
	{
	    char buffer[65536];
        ocl(clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
        printf("Using device: %s\n", buffer);
    }
    #endif

    /* Create a context and command queue */
    cl_int err;
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    ocl_check("clCreateContext", err);
    command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    #ifdef __MYOKIT_DEBUG
    printf("Created context and command queue.\n");
    #endif

    /* Create state vector */
    int dsize_state = n_cells * n_state * sizeof(Real);
    state = (Real*)malloc(dsize_state);
    set_initial_values(n_cells, state);
    
    /* Create diffusion current vector and copy to device */
    int dsize_idiff = n_cells * sizeof(Real);
    idiff = (Real*)malloc(dsize_idiff);
    for(i=0; i<n_cells; i++) idiff[i] = 0.0;
        
    /* Create memory buffers on the device */
    mbuf_state = clCreateBuffer(context, CL_MEM_READ_WRITE, dsize_state, NULL, &err);
    ocl_check("clCreateBuffer mbuf_state", err);
    mbuf_idiff = clCreateBuffer(context, CL_MEM_READ_ONLY, dsize_idiff, NULL, &err);
    ocl_check("clCreateBuffer mbuf_idiff", err);
    #ifdef __MYOKIT_DEBUG
    printf("Created buffers.\n");
    #endif  

    /* Copy data into buffers */
    ocl(clEnqueueWriteBuffer(command_queue, mbuf_state, CL_TRUE, 0, dsize_state, state, 0, NULL, NULL));
    ocl(clEnqueueWriteBuffer(command_queue, mbuf_idiff, CL_TRUE, 0, dsize_idiff, idiff, 0, NULL, NULL));
    #ifdef __MYOKIT_DEBUG
    printf("Set initial state.\n");
    #endif
    
    /* Load and compile the kernel program(s) */
    char* source = read_kernel("kernel.cl");
    program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
    ocl_check("clCreateProgramWithSource", err);
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        fprintf(stderr, "OpenCL Error: Kernel failed to compile.\n");
        fprintf(stderr, "----------------------------------------");
        fprintf(stderr, "---------------------------------------\n");        
        /* Extract build log */
        size_t blog_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &blog_size);
        char *blog = (char*)malloc(blog_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, blog_size, blog, NULL);
        fprintf(stderr, "%s\n", blog);
        fprintf(stderr, "----------------------------------------");
        fprintf(stderr, "---------------------------------------\n");
        abort();
    }
    ocl_check("clBuildProgram", err);
    #ifdef __MYOKIT_DEBUG
    printf("Program created and built.\n");
    #endif
    
    /* Create the kernels */
    kernel_cell = clCreateKernel(program, "cell_step", &err);
    ocl_check("clCreateKernel", err);
    kernel_diff = clCreateKernel(program, "diff_step", &err);
    ocl_check("clCreateKernel", err);
    #ifdef __MYOKIT_DEBUG
    printf("Kernels created.\n");
    #endif

    /* Pass arguments into kernels */
    ocl(clSetKernelArg(kernel_cell, 0, sizeof(n_cells), &n_cells));
    ocl(clSetKernelArg(kernel_diff, 1, sizeof(time), &time));
    ocl(clSetKernelArg(kernel_cell, 2, sizeof(dt), &dt));
    ocl(clSetKernelArg(kernel_cell, 3, sizeof(mbuf_state), &mbuf_state));
    ocl(clSetKernelArg(kernel_cell, 4, sizeof(mbuf_idiff), &mbuf_idiff));
    ocl(clSetKernelArg(kernel_diff, 0, sizeof(n_cells), &n_cells));    
    ocl(clSetKernelArg(kernel_diff, 1, sizeof(g), &g));
    ocl(clSetKernelArg(kernel_diff, 2, sizeof(mbuf_state), &mbuf_state));
    ocl(clSetKernelArg(kernel_diff, 3, sizeof(mbuf_idiff), &mbuf_idiff));
    #ifdef __MYOKIT_DEBUG
    printf("Arguments passed into kernels.\n");
    #endif
    
    /* Add log header */
    printf("engine.time");
    for(i=0; i<n_cells; i++) {
<?
for v in model.states():
    print(tab*2 + 'printf(",\\"%d.' + v.qname() + '\\"", i);')
?>
    }
    printf("\n");
    
    /* Start simulation */
    int steps = 0;
    int steps_till_log = 1;
    while(time < time_end) {
        /* Update cells */
        ocl(clSetKernelArg(kernel_cell, 1, sizeof(time), &time));
        ocl(clEnqueueNDRangeKernel(command_queue, kernel_diff, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
        ocl(clEnqueueNDRangeKernel(command_queue, kernel_cell, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
    
        /* Show results for first cell */
        if (--steps_till_log == 0) {
            ocl(clEnqueueReadBuffer(command_queue, mbuf_state, CL_TRUE, 0, dsize_state, state, 0, NULL, NULL));
            printf("%f", time);
            for(i=0; i<n_cells*n_state; i++) {
                printf(", %f", state[i]);
            }
            printf("\n");
            steps_till_log = steps_per_log;
        }
        
        steps++;
        time = time_start + steps * dt;
    }
        
done:
    /* Tidy up */
    #ifdef __MYOKIT_DEBUG
    printf("Tidying up.\n");
    #endif  

    ocl(clFlush(command_queue));
    ocl(clFinish(command_queue));
    ocl(clReleaseMemObject(mbuf_state));
    ocl(clReleaseMemObject(mbuf_idiff));
    ocl(clReleaseKernel(kernel_cell));
    ocl(clReleaseKernel(kernel_diff));
    ocl(clReleaseProgram(program)); 
    ocl(clReleaseCommandQueue(command_queue));    
    ocl(clReleaseContext(context));
    free(state);
    free(idiff);
    
    #ifdef __MYOKIT_DEBUG
    printf("Done.\n");
    #endif 
}

int main()
{
    run(
        256,        // n_cells
        0,          // time_start
        600,        // time_end
        0.005,      // dt
        1.5         // g
        );
        
    return 0;
}
