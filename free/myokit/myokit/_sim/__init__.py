#
# This hidden module contains the core functions dealing with simulations and
# the data they generate.
#
# This file is part of Myokit
#  Copyright 2011-2016 Michael Clerx, Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Library imports
import os
import imp
import sys
import shutil
import tempfile
import traceback
# Windows fix: On win7 with MinGW, when running distutils from Qt the
# (deprecated) os.popen command fails. The docs suggest to replace calls to
# popen with subprocess.Popen. The following wrapper implements this
# dynamically.
import platform
if platform.system() == 'Windows':
    import subprocess
    def _ospop(command, mode='r', bufsize=0):
        if mode == 'r':
            return subprocess.Popen(command, shell=True, bufsize=bufsize,
                stdout=subprocess.PIPE).stdout
        else:
            return subprocess.Popen(command, shell=True, bufsize=bufsize,
                stdin=subprocess.PIPE).stdin
    os.popen = _ospop
# Distutils imports
# Setuptools has a load of patches/fixes for distutils, but causes errors
# with the current code.
from distutils.core import setup, Extension
# Myokit imports
import myokit
import myokit.pype as pype
class CModule(object):
    """
    Abstract base class for classes that dynamically create and compile a
    back-end C-module.
    """
    def _code(self, tpl, tpl_vars, line_numbers=False):
        """
        Returns the code that would be created by the equivalent call to
        :meth:`_compile()`.
        """
        if line_numbers:
            lines = []
            i = 1
            for line in self._export(tpl, tpl_vars).split('\n'):
                lines.append('{:4d}'.format(i) + ' ' + line)
                i += 1
            return '\n'.join(lines)
        else:
            return self._export(tpl, tpl_vars)
    def _compile(self, name, tpl, tpl_vars, libs, libd=None, incd=None):
        """
        Compiles a source template into a module and returns it.

        The module's name is specified by ``name``.

        The template to compile is given by ``tpl``, while any variables
        required to process the template should be given as the dict
        ``tpl_vars``.

        Any C libraries needed for compilation should be given in the sequence
        type ``libs``. Library dirs and include dirs can be passed in using
        ``libd`` and ``incd``.
        """
        src_file = self._source_file()
        d_cache = tempfile.mkdtemp('myokit')
        try:
            # Create output directories
            d_build = os.path.join(d_cache, 'build')
            d_modul = os.path.join(d_cache, 'module')
            os.makedirs(d_build)
            os.makedirs(d_modul)
            # Export c file
            src_file = os.path.join(d_cache, src_file)
            self._export(tpl, tpl_vars, src_file)
            # Create extension
            ext = Extension(
                name,
                sources=[src_file],
                libraries = libs,
                library_dirs = libd,
                include_dirs = incd,
                )
            # Compile, catch output
            with myokit.SubCapture() as s:
                try:
                    setup(name = name,
                        description = 'Temporary module',
                        ext_modules = [ext],
                        script_args = [
                        'build', '--build-base=' + d_build,
                        'install', '--install-lib=' + d_modul,
                        ])
                except (Exception, SystemExit) as e:
                    s.disable()
                    t = [
                        'Unable to compile.',
                        'Error message:',
                        '    ' + e.message,
                        'Compiler output:',
                        ]
                    captured = s.text().strip()
                    t.extend(['    ' + x for x in captured.splitlines()])
                    raise myokit.CompilationError('\n'.join(t))
            # Include module (and refresh in case 2nd model is loaded)
            (f, pathname, description) = imp.find_module(name, [d_modul])
            return imp.load_dynamic(name, pathname)
        finally:
            try:
                shutil.rmtree(d_cache)
            except Exception:
                pass
    def _export(self, source, varmap, target=None):
        """
        Exports the given ``source`` to the file ``target`` using the variable
        mapping ``varmap``. If no target is given, the result is returned as a
        string.
        """
        # Test if given module path is writable
        if target is not None:
            if os.path.exists(target):
                if os.path.isdir(target):
                    line = 'Can\'t create output file. A directory exists at '
                    line += format_path(target)
                    raise IOError(line)
            # Open output file
            handle = open(target, 'w')
        # Create source
        p = pype.TemplateEngine()
        if target is not None:
            p.set_output_stream(handle)
        try:
            result = None
            result = p.process(source, varmap)
        except pype.PypeError as e:
            msg = ['An error ocurred while processing the template']
            msg.append(traceback.format_exc())
            d = p.error_details()
            if d:
                msg.append(d)
            raise myokit.GenerationError('\n'.join(msg))
        finally:
            if target is not None:
                handle.close()
        return result
    def _source_file(self):
        """
        Returns a name for the source file created and compiled for this
        module.
        """
        return 'source.c'
class CppModule(CModule):
    """
    Extends the :class:`CModule` class and adds C++ support.
    """
    def _source_file(self):
        return 'source.cpp'
class ProgressReporter(object):
    """
    Interface for progress updates in Simulations. Also allows some job types
    to be cancelled by the user.
    
    Many simulation types take an argument ``progress`` that can be used to
    pass in an object implementing this interface. The simulation will use this
    object to report on its progress.
    
    Note that progress reporters should be re-usable, but the behaviour when
    making calls to a reporter from two different processes (either through
    multi-threading/multi-processing or jobs nested within jobs) is undefined.
    
    An optional description of the job to run can be passed in at construction
    time as `msg`.
    """
    def __init__(self, msg=None):
        pass
    def enter(self, msg=None):
        """
        This method will be called when the job that provides progress updates
        is started.

        An optional description of the job to run can be passed in at
        construction time as `msg`.
        """
        raise NotImplementedError
    def exit(self):
        """
        Called when a job is finished and the progress reports should stop.
        """
        raise NotImplementedError
    def job(self, msg=None):
        """
        Returns a context manager that will enter and exit this
        ProgressReporter using the `with` statement.
        """
        return ProgressReporter._Job(self, msg)
    def update(self, progress):
        """
        This method will be called to provides updates about the current
        progress. This is indicated using the floating point value
        ``progress``, which will have a value in the range ``[0, 1]``.
        
        The return value of this update can be used to cancel a job (if job
        type supports it). Return ``True`` to keep going, ``False`` to cancel
        the job.
        """
        raise NotImplementedError
    class _Job(object):
        def __init__(self, parent, msg):
            self._parent = parent
            self._msg = msg
        def __enter__(self):
            self._parent.enter(self._msg)
        def __exit__(self, type, value, traceback):
            self._parent.exit()
class ProgressPrinter(ProgressReporter):
    """
    Writes progress information to stdout, can be used during a simulation.
    
    For example::
    
        m, p, x = myokit.load('example')
        s = myokit.Simulation(m, p)
        w = myokit.ProgressPrinter(digits=1)
        d = s.run(10000, progress=w)
    
    This will print strings like::
    
        [8.9 minutes] 71.7 % done, estimated 4.2 minutes remaining
        
    To ``stdout`` during the simulation.
    
    Output is only written if the new percentage done differs from the old one,
    in a string format specified by the number of ``digits`` to display. The
    ``digits`` parameter accepts the special value ``-1`` to only print out a
    status every ten percent.
    """
    def __init__(self, digits=1):
        super(ProgressPrinter, self).__init__()
        self._b = myokit.Benchmarker()
        self._f = None
        self._d = int(digits)
    def enter(self, msg=None):
        # Reset
        self._b.reset()
        self._f = None
    def exit(self):
        pass
    def update(self, f):
        if f == 0:
            # First call, reset
            self._b.reset()
            self._f = None
        if self._d < 0:
            f = 10 * round(10 * f, 0)
        else:
            f = round(100 * f, self._d)
        if f != self._f:
            self._f = f
            t = self._b.time()
            if f > 0:
                p = t * (100 / f - 1)
                if p > 60:
                    p = str(round(p / 60, 1))
                    p = ', estimated ' + p + ' minutes remaining'
                else:
                    p = str(int(round(p, 0)))
                    p = ', estimated ' + p + ' seconds remaining'
            else:
                p = ''
            t = str(round(t / 60, 1))
            print('[' + t + ' minutes] ' + str(f)  + ' % done' + p)
            sys.stdout.flush()
        return True
