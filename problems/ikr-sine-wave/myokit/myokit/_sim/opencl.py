#
# OpenCL information class
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
import ConfigParser
# Settings file
SETTINGS_FILE = os.path.join(myokit.DIR_USER, 'preferred-opencl-device.ini')
# Location of C source for OpenCL info module
SOURCE_FILE = 'opencl.c'
class OpenCL(myokit.CModule):
    """
    Tests for OpenCL support and can return information about opencl
    simulations.
    """
    # Unique id for this object
    _index = 0 
    # Cached back-end object if compiled, False if compilation failed
    _instance = None
    # Cached compilation error message
    _message = None
    def __init__(self):
        super(OpenCL, self).__init__()
        # Create back-end and cache it
        OpenCL._index += 1
        mname = 'myokit_opencl_info_' + str(OpenCL._index)
        fname = os.path.join(myokit.DIR_CFUNC, SOURCE_FILE)
        args = {'module_name' : mname}
        libs = ['OpenCL']
        libd = list(myokit.OPENCL_LIB)
        incd = list(myokit.OPENCL_INC)
        incd.append(myokit.DIR_CFUNC)
        try:
            OpenCL._message = None
            OpenCL._instance = self._compile(
                mname, fname, args, libs, libd, incd)
        except myokit.CompilationError as e:
            OpenCL._instance = False
            OpenCL._message = e.message
    @staticmethod
    def info(formatted=False):
        """
        Queries the OpenCL installation for the available platforms and
        devices and returns a :class:`myokit.OpenCLInfo` object.
        
        If ``formatted=True`` is set, a formatted version of the information is
        returned instead.
        """
        info = OpenCLInfo(OpenCL._get_instance().info())
        return info.format() if formatted else info
    @staticmethod
    def _get_instance():
        """
        Returns a cached back-end, creates and returns a new back-end or raises
        a :class:`NoOpenCLError`.
        """
        # No instance? Create it
        if OpenCL._instance is None:
            OpenCL()
        # Instance creation failed, raise exception
        if OpenCL._instance is False:
            raise NoOpenCLError('OpenCL support not found.\n'
                + OpenCL._message)
        # Return instance
        return OpenCL._instance
    @staticmethod
    def load_selection():
        """
        Loads a platform/device selection from disk and returns a tuple
        ``(platform, device)``. Each entry in the tuple is either a string
        with the platform/device name, or ``None`` if no preference was set.
        """
        platform = device = None
        # Read ini file
        inifile = os.path.expanduser(SETTINGS_FILE)
        if os.path.isfile(inifile):
            config = ConfigParser.ConfigParser()
            config.read(inifile)
            def get(section, option):
                x = None
                if config.has_section(section):
                    if config.has_option(section, option):
                        x = config.get(section, option).strip()
                        if x:
                            return x
                return None
            platform = get('selection', 'platform')
            device = get('selection', 'device')
        return platform, device
    @staticmethod
    def save_selection(platform=None, device=None):
        """"
        Stores a platform/device selection to disk.
        
        Both platform and device are identified by their names.
        """
        # Create configuration
        config = ConfigParser.ConfigParser()
        config.add_section('selection')
        if platform:
            config.set('selection', 'platform', platform)
        if device:
            config.set('selection', 'device', device)
        # Write configuration to ini file
        inifile = os.path.expanduser(SETTINGS_FILE)
        with open(inifile, 'wb') as configfile:
            config.write(configfile)
    @staticmethod
    def selection_info():
        """
        Returns a list of platform/device combinations along with information
        allowing the user to select one.
        
        The returned output is a list of tuples, where each tuple has the form
        ``(platform_name, device_name, specs)``.
        
        A preferred device can be selected by passing one of the returned
        ``platform_name, device_name`` combinations to
        :meth:`OpenCL.set_preferred_device`.
        """
        devices = []
        for platform in OpenCL.info().platforms:
            for device in platform.devices:
                specs = clockspeed(device.clock)
                specs += ', ' + bytesize(device.globl) + ' global'
                specs += ', ' + bytesize(device.local) + ' local'
                specs += ', ' + bytesize(device.const) + ' const'
                devices.append((platform.name, device.name, specs))
        return devices
    @staticmethod
    def supported():
        """
        Returns ``True`` if OpenCL support has been detected on this system.
        """
        try:
            OpenCL._get_instance()
            return True
        except NoOpenCLError:
            return False
class OpenCLInfo(object):
    """
    Represents information about the available OpenCL platforms and devices.
    
    Each ``OpenCLInfo`` object has a property ``platforms``, containing a list
    (actually a tuple) of :class:`OpenCLPlatformInfo` objects.
    
    ``OpenCLInfo`` objects can be created by any OpenCL enabled part of Myokit.
    """
    def __init__(self, mcl_info):
        # mcl_info is a python object returned by mcl_device_info (mcl.h)
        self.platforms = tuple([OpenCLPlatformInfo(x) for x in mcl_info])
    def format(self):
        """
        Returns a formatted version of the information.
        """
        t = []
        for i, platform in enumerate(self.platforms):
            t.append('Platform ' + str(i))
            t.append(' Name       : ' + platform.name)
            t.append(' Vendor     : ' + platform.vendor)
            t.append(' Version    : ' + platform.version)
            t.append(' Profile    : ' + platform.profile)
            t.append(' Extensions : ' + platform.extensions)
            t.append(' Devices    :')
            for j, device in enumerate(platform.devices):
                t.append('  Device ' + str(j))  
                t.append('   Name            : ' + device.name)
                t.append('   Vendor          : ' + device.vendor)
                t.append('   Version         : ' + device.version)
                t.append('   Driver          : ' + device.driver)
                t.append('   Clock speed     : ' + str(device.clock) + ' MHz')
                t.append('   Global memory   : ' + bytesize(device.globl))
                t.append('   Local memory    : ' + bytesize(device.local))
                t.append('   Constant memory : ' + bytesize(device.const))
                t.append('   Max param size  : ' + str(device.param) +' bytes')
                t.append('   Max work groups : ' + str(device.groups))
                t.append('   Max work items  : ['
                    + ', '.join([str(x) for x in device.items]) + ']')
                
        return '\n'.join(t)
class OpenCLPlatformInfo(object):
    """
    Represents information about an OpenCL platform.
    
    An ``OpenCLPlatformInfo`` object has the following properties:
    
    ``name`` (string)
        This platform's name.
    ``vendor`` (string)
        The vendor of this platform.
    ``version`` (string)
        The OpenCL version supported by this platform.
    ``profile`` (string)
        The supported OpenCL profile of this platform.
    ``extensions`` (string)
        The available OpenCL extensions on this platform.
    ``devices`` (tuple)
        A tuple of device information dicts for the devices available on
        this platform.
    
    ``OpenCLPlatformInfo`` objects are created as part of a :class:`OpenCLInfo`
    objects, as returned by most OpenCL enabled parts of Myokit.
    """
    def __init__(self, platform):
        self.name = platform['name'].strip()
        self.vendor = platform['vendor'].strip()
        self.version = platform['version'].strip()
        self.profile = platform['profile'].strip()
        self.extensions = platform['extensions'].strip()
        self.devices = tuple(
            [OpenCLDeviceInfo(x) for x in platform['devices']])
class OpenCLDeviceInfo(object):
    """
    Represents information about an OpenCL device.
    
    An ``OpenCLDeviceInfo`` object has the following properties:
        
    ``name`` (string)
        This device's name.
    ``vendor`` (string)
        This device's vendor.
    ``version`` (string)
        The OpenCL version supported by this device.
    ``driver`` (string)
        The driver version for this device.
    ``clock`` (int)
        This device's clock speed (in MHz).
    ``globl`` (int)
        The available global memory on this device (in bytes).
    ``local`` (int)
        The available local memory on this device (in bytes).
    ``const`` (int)
        The available constant memory on this device (in bytes).
    ``units`` (int)
        The number of computing units on this device.
    ``param`` (int)
        The maximum total size (in bytes) of arguments passed to the
        kernel. This limits the number of arguments a kernel can get.
    ``groups`` (int)
        The maximum work group size.
    ``dimensions`` (int)
        The maximum work item dimension.
    ``items`` (tuple)
        A tuple of ints specifying the maximum work item size in each
        dimension.

    ``OpenCLDeviceInfo`` objects are created as part of a :class:`OpenCLInfo`
    objects, as returned by most OpenCL enabled parts of Myokit.
    """
    def __init__(self, device):
        self.name = device['name'].strip()
        self.vendor = device['vendor'].strip()
        self.version = device['version'].strip()
        self.driver = device['driver'].strip()
        self.clock = device['clock']
        self.globl = device['global']
        self.local = device['local']
        self.const = device['const']
        self.param = device['param']
        self.groups = device['groups']
        self.items = tuple(device['items'])
def bytesize(size):
    """
    Returns a formatted version of a ``size`` given in bytes.
    """
    # Format a size
    if size > 1073741824:
        return str(0.1 * int(10*(float(size) / 1073741824))) + ' GB'
    elif size > 1048576:
        return str(0.1 * int(10*(float(size) / 1048576))) + ' MB'
    elif size > 1024:
        return str(0.1 * int(10*(float(size) / 1024))) + ' KB'
    else:
        return str(size) + 'B'
def clockspeed(speed):
    """
    Returns a formatted version of a ``speed`` given in MHz.
    """
    # Format a size
    if speed > 1000:
        return str(0.1 * int(10*(float(speed) / 1000))) + ' GHz'
    else:
        return str(speed) + ' MHz'
class NoOpenCLError(myokit.MyokitError):
    """
    Raised when OpenCLInfo functions requiring OpenCL are called but no opencl
    support can be detected.
    """
class PreferredOpenCLPlatformNotFoundError(myokit.MyokitError):
    """
    Raised when the platform preferred by the user cannot be found.
    """
    def __init__(self, platform_name):
        super(PreferredOpenCLPlatformNotFoundError, self).__init__(
            'The preferred platform "' + platform_name + '" cannot be found.')
class PreferredOpenCLDeviceNotFoundError(myokit.MyokitError):
    """
    Raised when the device preferred by the user cannot be found.
    """
    def __init__(self, device_name, platform_name=None):
        msg = 'The preferred device "' + device_name + '" cannot be found'
        if platform_name is None:
            msg += '.'
        else:
            msg += ' on platform "' + platform_name + '".'
        super(PreferredOpenCLDeviceNotFoundError, self).__init__(msg)
