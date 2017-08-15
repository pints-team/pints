#
# Loads settings from configuration file in user home directory or attempts to
# guess best settings.
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
# Load Myokit, at least, the bit that's been setup so far. This just means
# this method will add a link to the myokit module already being loaded
# into this method's namespace. This allows us to use the constants defined
# before this method was called.
import myokit
# Load libraries
import os
import platform
import ConfigParser
def _create(path):
    """
    Attempts to guess the best settings and stores them in a new configuration
    file.
    """
    # Get operating system
    system = platform.system()
    # Create config parser
    config = ConfigParser.ConfigParser(allow_no_value=True)
    # Make the parser case sensitive (need for unix paths!)
    config.optionxform = str
    # General information
    config.add_section('myokit')
    config.set('myokit', '# This file can be used to set global configuration'
        ' options for Myokit.')
    # Date format
    config.add_section('time')
    config.set('time', '# Date format used throughout Myokit')
    config.set('time', '# Format should be acceptable for time.strftime')
    config.set('time', 'date_format', myokit.DATE_FORMAT)
    config.set('time', '# Time format used throughout Myokit')
    config.set('time', '# Format should be acceptable for time.strftime')
    config.set('time', '# Format should be acceptable for time.strftime')
    config.set('time', 'time_format', myokit.TIME_FORMAT)
    # Add line numbers to debug output of simulations
    config.add_section('debug')
    config.set('debug', '# Add line numbers to debug output of simulations')
    config.set('debug', 'line_numbers', myokit.DEBUG_LINE_NUMBERS)
    # GUI Backend
    config.add_section('gui')
    config.set('gui', '# Backend to use for graphical user interface.')
    config.set('gui', '# Valid options are "pyqt5", "pyqt4" or "pyside".')
    config.set('gui', '# Leave unset for automatic selection.')
    config.set('gui', '#backend = pyqt5')
    config.set('gui', '#backend = pyqt4')
    config.set('gui', '#backend = pyside')
    # Locations of sundials library
    config.add_section('sundials')
    config.set('sundials', '# Location of sundials shared libary files (.so'
        ' or .dll).')
    config.set('sundials', '# Multiple paths can be set using ; as separator.')
    if system == 'Windows':
        # All windowses
        # First, try finding local sundials install
        sundials_win = os.path.join(myokit.DIR_MYOKIT, '..','tools','sundials')
        sundials_win = os.path.abspath(sundials_win)
        # Now, set library path
        config.set('sundials', 'lib', ';'.join([
            os.path.join(sundials_win, 'lib'),
            'C:\\Program Files\\sundials\\lib',
            'C:\\Program Files (x86)\\sundials\\lib',
            ]))
    else:
        # Linux and OS/X
        # Standard linux and OS/X install: /usr/local/lib
        # Macports OS/X install: /opt/local/lib ??
        config.set('sundials', 'lib', ';'.join([
            '/usr/local/lib',
            '/opt/local/lib',
            ]))
    config.set('sundials', '# Location of sundials header files (.h).')
    config.set('sundials', '# Multiple paths can be set using ; as separator.')
    if system == 'Windows':
        # All windowses
        config.set('sundials', 'inc', ';'.join([
            os.path.join(sundials_win, 'include'),
            'C:\\Program Files\\sundials\\include',
            'C:\\Program Files (x86)\\sundials\\include',
            ]))
    else:
        # Linux and OS/X
        # Standard linux and OS/X install: /usr/local/include
        # Macports OS/X install: /opt/local/include
        config.set('sundials', 'inc', ';'.join([
            '/usr/local/include',
            '/opt/local/include',
            ]))
    # Locations of OpenCL libraries
    config.add_section('opencl')
    config.set('opencl', '# Location of opencl shared libary files (.so or'
        ' .dll).')
    config.set('opencl', '# Multiple paths can be set using ; as separator.')
    if system == 'Windows':
        # All windowses
        config.set('opencl', 'lib', ';'.join([
            'C:\\Program Files\\Intel\\OpenCL SDK\\6.3\\lib\\x86',
            'C:\\Program Files (x86)\\Intel\\OpenCL SDK\\6.3\\lib\\x86',
            'C:\\Program Files\\AMD APP SDK\\2.9\\bin\\x86',
            'C:\\Program Files (x86)\\AMD APP SDK\\2.9\\bin\\x86',
            'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v7.0\\lib\\Win32',
            'C:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA\\v7.0\\lib\\Win32',
            ]))
    else:
        # Linux and mac
        config.set('opencl', 'lib', ';'.join([
            '/usr/lib64',
            '/usr/lib64/nvidia',
            '/usr/local/cuda/lib64',
            ]))
    config.set('opencl', '# Location of opencl header files (.h).')
    config.set('opencl', '# Multiple paths can be set using ; as separator.')
    if system == 'Windows':
        # All windowses
        config.set('opencl', 'inc', ';'.join([
        # Enable for Intel OpenCL drivers
            'C:\\Program Files\\Intel\\OpenCL SDK\\6.3\\include',
            'C:\\Program Files (x86)\\Intel\\OpenCL SDK\\6.3\\include',
            'C:\\Program Files\\AMD APP SDK\\2.9\\include',
            'C:\\Program Files (x86)\\AMD APP SDK\\2.9\\include',
            'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v7.0\\include',
            'C:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA\\v7.0\\include',
            ]))
    else:
        # Linux and mac
        config.set('opencl', 'inc', ';'.join([
            '/usr/include/CL',
            '/usr/local/cuda/include',
            ]))
    # Write ini file
    try:
        with open(path, 'wb') as configfile:
            config.write(configfile)
    except Exception:
        print('Warning: Unable to write settings to ' + str(path))
def _load():
    """
    Reads the configuration file and attempts to set the library paths.
    """
    # Location of configuration file
    path = os.path.join(myokit.DIR_USER, 'myokit.ini')
    # No file present? Create one and return
    if not os.path.isfile(path):
        _create(path)
    # Create the config parser (no value allows comments)
    config = ConfigParser.ConfigParser(allow_no_value=True)
    # Make the parser case sensitive (need for unix paths!)
    config.optionxform = str
    # Parse the config file
    config.read(path)
    # Date format
    if config.has_option('time', 'date_format'):
        x = config.get('time', 'date_format')
        if x:
            myokit.DATE_FORMAT = x
    # Time format
    if config.has_option('time', 'time_format'):
        x = config.get('time', 'time_format')
        if x:
            myokit.TIME_FORMAT = x
    # Add line numbers to debug output of simulations
    if config.has_option('debug', 'line_numbers'):
        try:
            myokit.DEBUG_LINE_NUMBERS = config.getboolean('debug',
                'line_numbers')
        except ValueError:
            pass
    # GUI Backend
    if config.has_option('gui', 'backend'):
        x = config.get('gui', 'backend').strip().lower()
        if x == 'pyside':
            myokit.FORCE_PYSIDE = True
            myokit.FORCE_PYQT4  = False
            myokit.FORCE_PYQT5  = False
        elif x == 'pyqt' or x == 'pyqt4':
            myokit.FORCE_PYSIDE = False
            myokit.FORCE_PYQT4  = True
            myokit.FORCE_PYQT5  = False
        elif x == 'pyqt5':
            myokit.FORCE_PYSIDE = False
            myokit.FORCE_PYQT4  = False
            myokit.FORCE_PYQT5  = True        
        else:
            # If empty or invalid, don't adjust the settings!
            pass
    # Sundial libraries and header files
    if config.has_option('sundials', 'lib'):
        for x in config.get('sundials', 'lib').split(';'):
            myokit.SUNDIALS_LIB.append(x.strip())
    if config.has_option('sundials', 'inc'):
        for x in config.get('sundials', 'inc').split(';'):
            myokit.SUNDIALS_INC.append(x.strip())
    # OpenCL libraries and header files
    if config.has_option('opencl', 'lib'):
        for x in config.get('opencl', 'lib').split(';'):
            myokit.OPENCL_LIB.append(x.strip())
    if config.has_option('opencl', 'inc'):
        for x in config.get('opencl', 'inc').split(';'):
            myokit.OPENCL_INC.append(x.strip())
# Load settings
_load()
