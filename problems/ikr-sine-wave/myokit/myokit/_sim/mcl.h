/*
 * mcl.h
 * 
 * Implements a number of OpenCL utility functions.
 *
 * This file is part of Myokit
 *  Copyright 2011-2017 Maastricht University
 *  Licensed under the GNU General Public License v3.0
 *  See: http://myokit.org
 *
 * Authors:
 *  Michael Clerx
 *
 * The string functions rtrim, ltrim and trim were taken from Wikipedia and
 * are not part of Myokit.
 *
 */
#ifndef MyokitOpenCL
#define MyokitOpenCL

#include <Python.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

// Load the opencl libraries.
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Show debug output
//#define MYOKIT_DEBUG

// Maximum number of platforms/devices to check for.
#define MCL_MAX_PLATFORMS 255
#define MCL_MAX_DEVICES 255

/*
 * String functions, straight from Wikipedia
 * https://en.wikipedia.org/wiki/Trimming_%28computer_programming%29#C.2FC.2B.2B
 */
void rtrim(char *str)
{
  size_t n;
  n = strlen(str);
  while (n > 0 && isspace((unsigned char)str[n - 1])) {
    n--;
  }
  str[n] = '\0';
}
void ltrim(char *str)
{
  size_t n;
  n = 0;
  while (str[n] != '\0' && isspace((unsigned char)str[n])) {
    n++;
  }
  memmove(str, str + n, strlen(str) - n + 1);
}
void trim(char *str)
{
  rtrim(str);
  ltrim(str);
}

/*
 * Checks for an opencl error, returns 1 if found and sets the error message,
 * else, returns 0.
 */ 
static int
mcl_flag(const cl_int flag)
{
    switch(flag) {
        case CL_SUCCESS:
            return 0;
        case CL_DEVICE_NOT_FOUND:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_DEVICE_NOT_FOUND");
            break;
        case CL_DEVICE_NOT_AVAILABLE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_DEVICE_NOT_AVAILABLE");
            break;
        case CL_COMPILER_NOT_AVAILABLE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_COMPILER_NOT_AVAILABLE");
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_MEM_OBJECT_ALLOCATION_FAILURE");
            break;
        case CL_OUT_OF_RESOURCES:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_OUT_OF_RESOURCES");
            break;
        case CL_OUT_OF_HOST_MEMORY:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_OUT_OF_HOST_MEMORY");
            break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_PROFILING_INFO_NOT_AVAILABLE");
            break;
        case CL_MEM_COPY_OVERLAP:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_MEM_COPY_OVERLAP");
            break;
        case CL_IMAGE_FORMAT_MISMATCH:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_IMAGE_FORMAT_MISMATCH");
            break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_IMAGE_FORMAT_NOT_SUPPORTED");
            break;
        case CL_BUILD_PROGRAM_FAILURE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_BUILD_PROGRAM_FAILURE");
            break;
        case CL_MAP_FAILURE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_MAP_FAILURE");
            break;
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_MISALIGNED_SUB_BUFFER_OFFSET");
            break;
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST");
            break;
        case CL_INVALID_VALUE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_VALUE");
            break;
        case CL_INVALID_DEVICE_TYPE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_DEVICE_TYPE");
            break;
        case CL_INVALID_PLATFORM:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_PLATFORM");
            break;
        case CL_INVALID_DEVICE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_DEVICE");
            break;
        case CL_INVALID_CONTEXT:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_CONTEXT");
            break;
        case CL_INVALID_QUEUE_PROPERTIES:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_QUEUE_PROPERTIES");
            break;
        case CL_INVALID_COMMAND_QUEUE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_COMMAND_QUEUE");
            break;
        case CL_INVALID_HOST_PTR:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_HOST_PTR");
            break;
        case CL_INVALID_MEM_OBJECT:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_MEM_OBJECT");
            break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_IMAGE_FORMAT_DESCRIPTOR");
            break;
        case CL_INVALID_IMAGE_SIZE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_IMAGE_SIZE");
            break;
        case CL_INVALID_SAMPLER:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_SAMPLER");
            break;
        case CL_INVALID_BINARY:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_BINARY");
            break;
        case CL_INVALID_BUILD_OPTIONS:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_BUILD_OPTIONS");
            break;
        case CL_INVALID_PROGRAM:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_PROGRAM");
            break;
        case CL_INVALID_PROGRAM_EXECUTABLE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_PROGRAM_EXECUTABLE");
            break;
        case CL_INVALID_KERNEL_NAME:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_KERNEL_NAME");
            break;
        case CL_INVALID_KERNEL_DEFINITION:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_KERNEL_DEFINITION");
            break;
        case CL_INVALID_KERNEL:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_KERNEL");
            break;
        case CL_INVALID_ARG_INDEX:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_ARG_INDEX");
            break;
        case CL_INVALID_ARG_VALUE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_ARG_VALUE");
            break;
        case CL_INVALID_ARG_SIZE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_ARG_SIZE");
            break;
        case CL_INVALID_KERNEL_ARGS:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_KERNEL_ARGS");
            break;
        case CL_INVALID_WORK_DIMENSION:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_WORK_DIMENSION");
            break;
        case CL_INVALID_WORK_GROUP_SIZE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_WORK_GROUP_SIZE");
            break;
        case CL_INVALID_WORK_ITEM_SIZE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_WORK_ITEM_SIZE");
            break;
        case CL_INVALID_GLOBAL_OFFSET:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_GLOBAL_OFFSET");
            break;
        case CL_INVALID_EVENT_WAIT_LIST:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_EVENT_WAIT_LIST");
            break;
        case CL_INVALID_EVENT:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_EVENT");
            break;
        case CL_INVALID_OPERATION:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_OPERATION");
            break;
        case CL_INVALID_GL_OBJECT:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_GL_OBJECT");
            break;
        case CL_INVALID_BUFFER_SIZE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_BUFFER_SIZE");
            break;
        case CL_INVALID_MIP_LEVEL:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_MIP_LEVEL");
            break;
        case CL_INVALID_GLOBAL_WORK_SIZE:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_GLOBAL_WORK_SIZE");
            break;
        case CL_INVALID_PROPERTY:
            PyErr_SetString(PyExc_Exception, "OpenCL error: CL_INVALID_PROPERTY");
            break;
        default:
        {
            char err[1024];
            sprintf("Unknown OpenCL error: %d", err, (int)flag);
            PyErr_SetString(PyExc_Exception, err);
            break;
        }
    };
    return 1;
}

/*
 * Searches for the preferred platform and device.
 *
 * Arguments:
 *  PyObject* platform  A string representing the platform, or None
 *  PyObject* device    A string representing the device, or None
 *  cl_platform_id* pid The returned cl_platform_id, or NULL
 *  cl_device_id* did   The returned cl_device_id
 * The returned value is 0 if no error occurred, 1 if an error did occur. In
 *  this case a python error message will also be set.
 */
int mcl_select_device(
    PyObject* platform,
    PyObject* device,
    cl_platform_id* pid,
    cl_device_id* did)
{
    // Check input
    const char* pname;
    const char* dname;
    if (platform != Py_None) {
        if (!PyString_Check(platform)) {
            PyErr_SetString(PyExc_Exception, "MCL_SELECT_DEVICE: 'platform' must be a string or None.");
            return 1;
        }
        pname = PyString_AsString(platform);
    }
    if (device != Py_None) {
        if (!PyString_Check(device)) {
            PyErr_SetString(PyExc_Exception, "MCL_SELECT_DEVICE: 'device' must be a string.");
            return 1;
        }
        dname = PyString_AsString(device);
    }
    
    #ifdef MYOKIT_DEBUG
    printf("Attempting to find platform and device.\n");
    if (platform == Py_None) {
        printf("No platform specified.\n");
    } else {
        printf("Selected platform: %s\n", pname);
    }
    if (device == Py_None) {
        printf("No device specified.\n");
    } else {
        printf("Selected device: %s\n", dname);
    }
    #endif
    
    // String containing name of platform/device
    char name[65536];

    // Get array of platform ids
    cl_platform_id platform_ids[MCL_MAX_PLATFORMS];
    cl_uint n_platforms = 0;
    cl_int flag = clGetPlatformIDs(MCL_MAX_PLATFORMS, platform_ids, &n_platforms);
    if(mcl_flag(flag)) return 1;
    if (n_platforms == 0) {
        PyErr_SetString(PyExc_Exception, "No OpenCL platforms found.");
        return 1;
    }
    
    // Platform unspecified
    if (platform == Py_None) {
    
        // Don't recommend a platform
        *pid = NULL;
    
        // No platform or device specified
        if (device == Py_None) {
            
            // Find any device on any platform, prefer GPU
            cl_device_id device_ids[1];
            cl_uint n_devices = 0;
            int i;
            for (i=0; i<n_platforms; i++) {        
                flag = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 1, device_ids, &n_devices);
                if(flag == CL_SUCCESS) {
                    // Set selected device and return.
                    *did = device_ids[0];
                    return 0;
                } else if(flag != CL_DEVICE_NOT_FOUND) {
                    mcl_flag(flag);
                    return 1;
                }
            }
            // No GPU found, now scan for any device
            for (i=0; i<n_platforms; i++) {
                flag = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 1, device_ids, &n_devices);
                if(flag == CL_SUCCESS) {
                    // Set selected device and return.
                    *did = device_ids[0];
                    return 0;
                } else if(flag != CL_DEVICE_NOT_FOUND) {
                    mcl_flag(flag);
                    return 1;
                }
            }
            // No device found
            PyErr_SetString(PyExc_Exception, "No OpenCL devices found.");
            return 1;

        // No platform specified, but there is a preferred device    
        } else {
        
            // Find specified device on any platform
            cl_device_id device_ids[MCL_MAX_DEVICES];
            cl_uint n_devices = 0;
            int i, j;
            for (i=0; i<n_platforms; i++) {
                flag = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, MCL_MAX_DEVICES, device_ids, &n_devices);
                if(flag == CL_SUCCESS) {
                    for (j=0; j<n_devices; j++) {
                        flag = clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, sizeof(name), name, NULL);
                        if(mcl_flag(flag)) return 1;
                        trim(name);
                        if (strcmp(name, dname) == 0) {
                            // Set selected device and return.
                            *did = device_ids[j];
                            return 0;
                        }
                    }
                } else if(flag != CL_DEVICE_NOT_FOUND) {
                    mcl_flag(flag);
                    return 1;
                }
            }
            // No device found
            PyErr_SetString(PyExc_Exception, "Specified OpenCL device not found.");
            return 1;
        }
    
    // Platform specified by user
    } else {
    
        // Find platform id
        int i;
        int found = 0;
        for (i=0; i<n_platforms; i++) {
            flag = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);
            if(mcl_flag(flag)) return 1;
            trim(name);
            if (strcmp(name, pname) == 0) {
                // Set selected platform
                *pid = platform_ids[i];
                found = 1;
                break;
            }
        }
        if (found == 0) {
            PyErr_SetString(PyExc_Exception, "Specified OpenCL platform not found.");
            return 1;
        }
            
        // Platform specified, but no preference for device
        if (device == Py_None) {
        
            // Find any device on specified platform, prefer GPU
            cl_device_id device_ids[1];
            cl_uint n_devices = 0;
            flag = clGetDeviceIDs(*pid, CL_DEVICE_TYPE_GPU, 1, device_ids, &n_devices);
            if(flag == CL_SUCCESS) {
                // Set selected device and return.
                *did = device_ids[0];
                return 0;
            } else if(flag != CL_DEVICE_NOT_FOUND) {
                mcl_flag(flag);
                return 1;
            }
            // No GPU found, return any device
            flag = clGetDeviceIDs(*pid, CL_DEVICE_TYPE_ALL, 1, device_ids, &n_devices);
            if(flag == CL_SUCCESS) {
                // Set selected device and return.
                *did = device_ids[0];
                return 0;
            } else if(flag != CL_DEVICE_NOT_FOUND) {
                mcl_flag(flag);
                return 1;
            }
            // No device found
            PyErr_SetString(PyExc_Exception, "No OpenCL devices found on specified platform.");
            return 1;    
        
        // Platform and device specified by user
        } else {
        
            // Find specified platform/device combo
            cl_device_id device_ids[MCL_MAX_DEVICES];
            cl_uint n_devices = 0;
            int j;
            flag = clGetDeviceIDs(*pid, CL_DEVICE_TYPE_ALL, MCL_MAX_DEVICES, device_ids, &n_devices);
            if(flag == CL_SUCCESS) {
                for (j=0; j<n_devices; j++) {
                    flag = clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, sizeof(name), name, NULL);
                    if(mcl_flag(flag)) return 1;
                    trim(name);
                    if (strcmp(name, dname) == 0) {
                        // Set selected device and return.
                        *did = device_ids[j];
                        return 0;
                    }
                }
            } else if(flag != CL_DEVICE_NOT_FOUND) {
                mcl_flag(flag);
                return 1;
            }
            // No device found
            PyErr_SetString(PyExc_Exception, "Specified OpenCL device not found on specified platform.");
            return 1;
        }
    }
}
    
/*
 * Rounds up to the nearest multiple of ws_size.
 */
static int
mcl_round_total_size(const int ws_size, const int total_size) 
{
    int size = (total_size / ws_size) * ws_size;
    if(size < total_size) size += ws_size;
    return size;
}

/* Memory used by mcl_device_info */
static PyObject* platforms = NULL;  // Tuple of platform dicts
static PyObject* platform = NULL;   // Temporary platform dict
static PyObject* devices = NULL;    // Temporary tuple of devices
static PyObject* device = NULL;     // Temporary device dict
static PyObject* items = NULL;      // Temporary tuple of item sizes
static PyObject* val;               // Temporary dictionary value
static size_t* work_item_sizes;     // Temporary array of work item sizes

/*
 * Tidies up if an error occurs in mcl_device_info
 */
static PyObject*
mcl_device_info_clean()
{
    Py_XDECREF(platforms); platforms = NULL;
    Py_XDECREF(platform); platform = NULL;
    Py_XDECREF(devices); devices = NULL;
    Py_XDECREF(device); device = NULL;
    Py_XDECREF(val); val = NULL;
    free(work_item_sizes); work_item_sizes = NULL;
    return 0;
}

/*
 * Returns information about the available OpenCL platforms and devices.
 *
 * Returns a reference to a tuple of platform dicts
 * 
 * platforms = (
 *      dict(platform) {
 *          'profile'    : str,
 *          'version'    : str,
 *          'name'       : str,
 *          'vendor'     : str,
 *          'extensions' : str,
 *          'devices'    : (
 *              dict(device) {
 *                  'name'       : str,
 *                  'vendor'     : str,
 *                  'version'    : str,
 *                  'driver'     : str,
 *                  'clock'      : int,     # Clock speed, in MHz
 *                  'global'     : int,     # Global memory, in bytes
 *                  'local'      : int,     # Local memory, in bytes
 *                  'const'      : int,     # Const memory, in bytes
 *                  'units'      : int,     # Computing units
 *                  'param'      : int,     # Max size of arguments passed to kernel
 *                  'groups'     : int,     # Max work group size
 *                  'dimensions' : int,     # Max work item dimensions
 *                  'items'      : (ints),  # Max work item sizes
 *                  }    
 *              ),
 *              ...
 *          },
 *          ...
 *     )
 */
static PyObject*
mcl_device_info()
{
    // Set all pointers used by clean() to null
    platforms = NULL;
    platform = NULL;
    devices = NULL;
    device = NULL;
    items = NULL;
    val = NULL;
    work_item_sizes = NULL;

    // Array of platform ids
    cl_platform_id platform_ids[MCL_MAX_PLATFORMS];

    // Number of platforms
    cl_uint n_platforms = 0;
    
    // Get platforms
    cl_int flag = clGetPlatformIDs(MCL_MAX_PLATFORMS, platform_ids, &n_platforms);
    if(mcl_flag(flag)) return mcl_device_info_clean();

    // Create platforms tuple
    platforms = PyTuple_New((size_t)n_platforms);
    
    if (n_platforms == 0) {
        // No platforms found
        return platforms;
    }
    
    // Devices & return values from queries
    cl_device_id device_ids[MCL_MAX_DEVICES];
    cl_uint n_devices = 0;
    cl_uint buf_uint;
    cl_ulong buf_ulong;
    size_t wgroup_size;
    size_t max_param;
    
    // String buffer
    char buffer[65536];
    
    // Check all platforms
    int i, j, k;
    for (i=0; i<n_platforms; i++) {
        // Create platform dict
        platform = PyDict_New();
        
        // Profile
        flag = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_PROFILE, sizeof(buffer), buffer, NULL);
        if(mcl_flag(flag)) return mcl_device_info_clean();
        val = PyString_FromString(buffer);
        PyDict_SetItemString(platform, "profile", val);
        Py_DECREF(val); val = NULL;

        // Version
        flag = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VERSION, sizeof(buffer), buffer, NULL);
        if(mcl_flag(flag)) return mcl_device_info_clean();
        val = PyString_FromString(buffer);
        PyDict_SetItemString(platform, "version", val);
        Py_DECREF(val); val = NULL;

        // Name
        flag = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL);
        if(mcl_flag(flag)) return mcl_device_info_clean();
        val = PyString_FromString(buffer);
        PyDict_SetItemString(platform, "name", val);
        Py_DECREF(val); val = NULL;

        // Vendor
        flag = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR, sizeof(buffer), buffer, NULL);
        if(mcl_flag(flag)) return mcl_device_info_clean();
        val = PyString_FromString(buffer);
        PyDict_SetItemString(platform, "vendor", val);
        Py_DECREF(val); val = NULL;

        // Extensions
        flag = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_EXTENSIONS, sizeof(buffer), buffer, NULL);
        if(mcl_flag(flag)) return mcl_device_info_clean();
        val = PyString_FromString(buffer);
        PyDict_SetItemString(platform, "extensions", val);
        Py_DECREF(val); val = NULL;

        // Devices
        flag = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, MCL_MAX_DEVICES, device_ids, &n_devices);
        if (flag == CL_DEVICE_NOT_FOUND) {
            n_devices = 0;
        } else if(mcl_flag(flag)) {
            return mcl_device_info_clean();
        }
        devices = PyTuple_New((size_t)n_devices);
        
        for (j=0; j<n_devices; j++) {
            // Create device dict
            device = PyDict_New();
        
            // Name
            flag = clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
            if(mcl_flag(flag)) return mcl_device_info_clean();
            val = PyString_FromString(buffer);
            PyDict_SetItemString(device, "name", val);
            Py_DECREF(val); val = NULL;
            
            // Vendor
            flag = clGetDeviceInfo(device_ids[j], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
            if(mcl_flag(flag)) return mcl_device_info_clean();
            val = PyString_FromString(buffer);
            PyDict_SetItemString(device, "vendor", val);
            Py_DECREF(val); val = NULL;
            
            // Device version
            flag = clGetDeviceInfo(device_ids[j], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
            if(mcl_flag(flag)) return mcl_device_info_clean();
            val = PyString_FromString(buffer);
            PyDict_SetItemString(device, "version", val);
            Py_DECREF(val); val = NULL;
            
            // Driver version
            flag = clGetDeviceInfo(device_ids[j], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
            if(mcl_flag(flag)) return mcl_device_info_clean();
            val = PyString_FromString(buffer);
            PyDict_SetItemString(device, "driver", val);
            Py_DECREF(val); val = NULL;
            
            // Clock speed (MHz)
            flag = clGetDeviceInfo(device_ids[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
            if(mcl_flag(flag)) return mcl_device_info_clean();
            val = PyInt_FromLong(buf_uint);
            PyDict_SetItemString(device, "clock", val);
            Py_DECREF(val); val = NULL;
            
            // Global memory (bytes)
            flag = clGetDeviceInfo(device_ids[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
            if(mcl_flag(flag)) return mcl_device_info_clean();
            val = PyInt_FromLong(buf_ulong);
            PyDict_SetItemString(device, "global", val);
            Py_DECREF(val); val = NULL;
            
            // Local memory (bytes)
            flag = clGetDeviceInfo(device_ids[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
            if(mcl_flag(flag)) return mcl_device_info_clean();
            val = PyInt_FromLong(buf_ulong);
            PyDict_SetItemString(device, "local", val);
            Py_DECREF(val); val = NULL;
            
            // Const memory (bytes)
            flag = clGetDeviceInfo(device_ids[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
            if(mcl_flag(flag)) return mcl_device_info_clean();
            val = PyInt_FromLong(buf_ulong);
            PyDict_SetItemString(device, "const", val);
            Py_DECREF(val); val = NULL;
            
            // Computing units
            flag = clGetDeviceInfo(device_ids[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
            if(mcl_flag(flag)) return mcl_device_info_clean();
            val = PyInt_FromLong(buf_uint);
            PyDict_SetItemString(device, "units", val);
            Py_DECREF(val); val = NULL;
            
            // Max workgroup size
            flag = clGetDeviceInfo(device_ids[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wgroup_size), &wgroup_size, NULL);
            if(mcl_flag(flag)) return mcl_device_info_clean();
            val = PyInt_FromLong(wgroup_size);
            PyDict_SetItemString(device, "groups", val);
            Py_DECREF(val); val = NULL;

            // Max workitem sizes
            flag = clGetDeviceInfo(device_ids[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(buf_uint), &buf_uint, NULL);
            if(mcl_flag(flag)) return mcl_device_info_clean();
            val = PyInt_FromLong(buf_uint);
            PyDict_SetItemString(device, "dimensions", val);
            Py_DECREF(val); val = NULL;
            
            work_item_sizes = (size_t*)malloc(buf_uint * sizeof(size_t));
            flag = clGetDeviceInfo(device_ids[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, buf_uint*sizeof(size_t), work_item_sizes, NULL);
            if(mcl_flag(flag)) return mcl_device_info_clean();
            items = PyTuple_New((size_t)buf_uint);
            for (k=0; k<buf_uint; k++) {
                PyTuple_SetItem(items, k, PyInt_FromLong(work_item_sizes[k]));
            }
            free(work_item_sizes); work_item_sizes = NULL;
            PyDict_SetItemString(device, "items", items);
            Py_DECREF(items); items = NULL;
            
            // Maximum size of a kernel parameter
            flag = clGetDeviceInfo(device_ids[j], CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(max_param), &max_param, NULL);
            if(mcl_flag(flag)) return mcl_device_info_clean();
            val = PyInt_FromLong(max_param);
            PyDict_SetItemString(device, "param", val);
            Py_DECREF(val); val = NULL;
            
            // Add device to devices tuple
            PyTuple_SetItem(devices, j, device);
            device = NULL;
        }
        
        // Add devices entry to platform dict
        PyDict_SetItemString(platform, "devices", devices);
        Py_DECREF(devices); devices = NULL;
        
        // Add platform to platforms tuple
        PyTuple_SetItem(platforms, i, platform);
        platform = NULL;
    }
    
    // Return platforms
    return platforms;
}

#undef MyokitOpenCL
#endif
