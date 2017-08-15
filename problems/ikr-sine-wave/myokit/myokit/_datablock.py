#
# Containers for time-series of 1d and 2d rectangular data arrays.
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
from __future__ import division
import os
import sys
import array
import numpy as np
import myokit
README_SAVE_1D = """
Myokit DataBlock1d Binary File
==============================
This zip file contains simulation data in the form of multiple time series.
Zero-dimensional time series, such as time or a global pace variable are
stored, as well as one-dimensional time series, such as the membrane potential
of a linear sequence of cells, as it varies over time.

This file has the following entries:

header_block1d.txt
------------------
A header file containing the following information (line by line):

  - nt     the number of points in time in each entry
  - nx     the width of each 1d block
  - dtype  the used datatype (either "d" or "f")
  - name   the names of all 0d entries, each on its own line
  - 1      the indication that the 1d entries are starting
  - name   the names of all 1d entries, each on its own line

data.bin
--------
A binary file containing the following data, in the data type specified by the
header, and little-endian:

  - The nt time values
  - All 0d entries
  - All 1d entries, reshaped using numpy order='C'

""".strip()
README_SAVE_2D = """
Myokit DataBlock2d Binary File
==============================
This zip file contains simulation data in the form of multiple time series.
Zero-dimensional time series, such as time or a global pace variable are
stored, as well as two-dimensional time series, such as the membrane potential
of a 2d grid of cells, as it varies over time.

This file has the following entries:

header_block2d.txt
------------------
A header file containing the following information (line by line):

  - nt     the number of points in time in each entry
  - ny     the height of each 2d block
  - nx     the width of each 2d block
  - dtype  the used datatype (either "d" or "f")
  - name   the names of all 0d entries, each on its own line
  - 2      the indication that the 2d entries are starting
  - name   the names of all 2d entries, each on its own line

data.bin
--------
A binary file containing the following data, in the data type specified by the
header, and little-endian:

  - The nt time values
  - All 0d entries
  - All 2d entries, reshaped using numpy order='C'

""".strip()
class DataBlock1d(object):
    """
    Container for time-series of 1d rectangular data arrays.
    
    Each ``DataBlock1d`` has a fixed width ``w``, and a 0d time series vector
    containing a sequence of ``n`` times.
    
    One-dimensional time series can be added to the block, provided the data
    also contains ``n`` instances of ``w`` data points. The data should be
    passed in as a numpy array with shape ``(n, w)``.
    
    Zero-dimensional time series can be added, provided they have length ``n``.
    
    A "one-dimensional time-series" is a series of equally sized 1d arrays
    (sequences), such that the first corresponds to a time ``t[0]``, the second
    to ``t[1]``, etc. Each array has shape ``(n, w)``.
    
    A "zero-dimensional time-series" is a series of single values where the
    first corresponds to a time ``t[0]``, the second to ``t[1]``, etc.

    Constructor info:
    
    ``w``
        Each 1d block should have dimension ``w`` by ``1``.
    ``time``
        A sequence of ``n`` times.
    ``copy``
        By default, a copy of the given time sequence will be stored. To
        prevent this copy, set ``copy=False``.
        
    """
    def __init__(self, w, time, copy=True):
        # Width
        w = int(w)
        if w < 1:
            raise ValueError('Minimum w is 1.')
        self._nx = w
        # Time
        time = np.array(time, copy=copy)
        if len(time.shape) != 1:
            raise ValueError('Time must be a sequence.')
        if np.any(np.diff(time) < 0):
            raise ValueError('Time must be non-decreasing.')
        self._time = time
        self._nt = len(time)
        # 0d variables
        self._0d = {}
        # 1d variables
        self._1d = {}
    def block2d(self):
        """
        Returns a :class:`myokit.DataBlock2d` based on this 1d data block.
        """
        b = DataBlock2d(self._nx, 1, self._time)
        for k, v in self._0d.iteritems():
            b.set0d(k, v)
        shape = (self._nt, 1, self._nx)
        for k, v in self._1d.iteritems():
            b.set2d(k, v.reshape(shape))
        return b
    def cv(self, name, threshold=-30, length=0.01, time_multiplier=1e-3,
            border=None):
        """
        Calculates conduction velocity (CV) in a cable.
        
        Accepts the following arguments:
        
        ``name``
            The name (as string) of the membrane potential variable. This
            should be a 1d variable in this datablock.
        ``threshold``
            The start of an action potential is determined as the first time
            the membrane potential crosses this threshold (default=-30mV) and
            has a positive direction.
        ``length``
            The length of a single cell in cm, in the direction of the cable.
            The default is ``length=0.01cm``.
        ``time_multiplier``
            A multiplier used to convert the used time units to seconds. Most
            simulations use milliseconds, so the default value is 1e-3.
        ``border``
            The number of cells to exclude from the analysis on each end of the
            cable to avoid boundary effects. If not given, 1/3 of the number of
            cells will be used, with a maximum of 50 cells on each side.
            
        Returns the approximate conduction velocity in cm/s. If no cv can be
        """
        # Check border
        if border is None:
            border = min(50, self._nx // 3)
        else:
            border = int(border)
            if border < 0:
                raise Exception('The argument `border` cannot be negative.')
            elif border >= self._nx // 2:
                raise Exception('The argument `border` must be less than half'
                    ' the number of cells.')
        # Get indices of selected cells
        ilo = border            # First indice
        ihi = self._nx - border # Last indice + 1
        # Get Vm, reshaped to get each cell's time-series successively.
        v_series = self._1d[name].reshape(self._nt * self._nx, order='F')
        # Split Vm into a series per cell (returns views!)
        v_series = np.split(v_series, self._nx)
        # Find first activation time
        have_crossing = False
        t = []
        for i in xrange(ilo, ihi):
            v = v_series[i]
            # Get indice of first threshold crossing with positive flank
            # Don't include crossings at log indice 0
            itime = np.where((v[1:] > -30) & (v[1:]-v[:-1] > 0))[0]
            if len(itime) == 0 or itime[0] == 0:
                # No crossing found
                if have_crossing:
                    # CV calculation ends here
                    ihi = i
                    break
                else:
                    # Delay CV calculation until first crossing
                    ilo += 1
            else:
                have_crossing = True
                itime = 1 + itime[0]
                # Interpolate to get better estimate
                v0 = v[itime - 1]
                v1 = v[itime]
                t0 = self._time[itime - 1]
                t1 = self._time[itime]
                t.append(t0 + (threshold - v0) * (t1 - t0) / (v1 - v0))
        if not have_crossing:
            return 0
        # Get times in seconds, lengths in cm
        t = np.array(t, copy=False) * time_multiplier
        x = np.arange(ilo, ihi, dtype=float) * length
        # Use linear least squares to find the conduction velocity
        from numpy.linalg import lstsq
        A = np.vstack([t, np.ones(len(t))]).T
        cv = np.linalg.lstsq(A, x)[0][0]
        # Return
        return cv
    @staticmethod
    def fromDataLog(log):
        """
        Creates a DataBlock1d from a :class:`myokit.DataLog`.
        """
        log.validate()
        # Get time variable name
        time = log.time_key()
        if time is None:
            raise ValueError('No time variable set in data log.')
        # Get log info
        infos = log.variable_info()
        # Check time variable
        info = infos[time]
        if info.dimension() != 0:
            raise ValueError('The given time variable should be 0d.')
        # Check if everything is 0d or 1d, get size
        size = None
        for name, info in infos.iteritems():
            d = info.dimension()
            if d not in (0, 1):
                raise ValueError('The given simulation log should only contain'
                    ' 0d or 1d variables. Found <' + str(name) + '> with d = '
                    + str(d) + '.')
            if d == 1:
                if size is None:
                    size = info.size()
                elif info.size() != size:
                    raise ValueError('The given simulation log contains 1d'
                        ' data sets of different sizes.')
        # Get dimensions
        nt = len(log[time])
        nx = size[0]
        # Create data block
        block = DataBlock1d(nx, log[time], copy=True)
        for name, info in infos.iteritems():
            if info.dimension() == 0:
                # Add 0d time series
                if name == time:
                    continue
                block.set0d(name, log[name], copy=True)
            else:
                # Convert to 1d time series
                data = np.zeros(nt * nx)
                # Iterate over info.keys(), this has the correct order!
                for i, key in enumerate(info.keys()):
                    # Copy data into array (really copies)
                    data[i*nt:(i+1)*nt] = log[key]
                # Reshape
                data = data.reshape((nt, nx), order='F')
                # If this is a view of existing data, make a copy!
                if data.base is not None:                    
                    data = np.array(data)
                block.set1d(name, data, copy=False)
        return block
    def get0d(self, name):
        """
        Returns the 0d time-series identified by ``name``. The data is returned
        directly, no copy is made.
        """
        return self._0d[name]
    def get1d(self, name):
        """
        Returns the 1d time-series identified by ``name``. The data is returned
        directly, no copy is made.
        
        The returned data is a 2d array of the shape given by :meth:`shape`.
        """
        return self._1d[name]
    def grid(self, name, transpose=True):
        """
        Returns a 2d grid representation suitable for plotting color maps or
        contours with ``matplotlib.pyplot`` methods such as ``pcolor`` and
        ``pcolormesh``.
        
        When used for example with
        ``pyplot.pcolormesh(*block.grid('membrane.V'))`` this will create a 2d
        plot where the horizontal axis shows time and the vertical axis shows
        the cell index.
        
        Arguments:
        
        ``name``
            The name identifying the 1d data-values to return.
        ``transpose``
            By default (``transpose=True``) the data is returned so that ``x``
            represents time and ``y`` represents space. To reverse this (and
            use the order used internally in the datablocks), set
            ``transpose=False``.

        The returned format is a tuple ``(x, y, z)`` where ``x``, ``y`` and
        ``z`` are all 2d numpy arrays.
        Here, ``x`` (time) and ``y`` (space) describe the x and y-coordinates
        of rectangles, with a color (data value) given by ``z``.
        
        In particular, each rectangle ``(x[i, j], y[i, j])``,
        ``(x[i + 1, j], y[i + 1, j])``, ``(x[i, j + 1], y[i, j + 1])``,
        ``(x[i + 1,j + 1], y[i + 1,j + 1])``, has a color given by ``z[i, j]``.
        
        As a result, for a block of width ``w`` (e.g., ``w`` cells) containing
        ``n`` logged time points, the method returns arrays ``x`` and ``y`` of
        shape ``(w + 1, n + 1)`` and an array ``z`` of shape ``(w, n)``.
        
        See :meth:`image_grid()` for a method where ``x``, ``y`` and ``z`` all
        have shape ``(w, n)``.
        """
        # Append point in time at pos [-1] + ([-1] - [-2])
        ts = np.append(self._time, 2*self._time[-1] - self._time[-2])
        # Append one extra cell or node
        xs = np.arange(0, self._nx + 1)
        # Make grid
        return self._grid(ts, xs, self._1d[name], transpose)
    def image_grid(self, name, transpose=True):
        """
        Returns a 2d grid representation of the data.
        
        The returned format is a tuple ``(x, y, z)`` where ``x``, ``y`` and
        ``z`` are all 2d numpy arrays.
        Here, ``x`` and ``y`` describe the time and space-coordinates of the
        logged points respectively, and ``z`` describes the corresponding data
        value.
        For a block of width ``w`` (e.g., ``w`` cells) containing ``n`` logged
        time points, each returned array has the shape ``(w, n)``.
        
        Arguments:
        
        ``name``
            The name identifying the 1d data-values to return.
        ``transpose``
            By default, the data is transposed so that the ``x`` coordinates
            become time and the ``y`` coordinates become space. Use
            ``transpose=False`` to return untransposed results.
        
        """
        return self._grid(self._time, np.arange(0, self._nx), self._1d[name],
            transpose)
    def _grid(self, ts, xs, vs, transpose):
        """
        Make a grid for the given times, spatial coordinates and data values.
        """
        if transpose:
            x, y = np.meshgrid(ts, xs)
            z = np.reshape(vs, (self._nx * self._nt,), order='F')
            z = np.reshape(z, (self._nx, self._nt), order='C')
        else:
            x, y = np.meshgrid(xs, ts)
            z = np.reshape(vs, (self._nx * self._nt,), order='C')
            z = np.reshape(z, (self._nt, self._nx), order='C')
        # If z is a view, create a copy
        if z.base is not None:
            z = np.array(z, copy=True)
        return x, y, z    
    def keys0d(self):
        """
        Returns an iterator over this block's 0d time series.
        """
        return iter(self._0d)
    def keys1d(self):
        """
        Returns an iterator over this block's 1d time series.
        """
        return iter(self._1d)
    def len0d(self):
        """
        Returns the number of 0d time series in this block.
        """
        return len(self._0d)
    def len1d(self):
        """
        Returns the number of 1d time series in this block.
        """
        return len(self._1d)
    @staticmethod
    def load(filename, progress=None, msg='Loading DataBlock1d'):
        """
        Loads a :class:`DataBlock1d` from the specified file.
        
        To obtain feedback on the simulation progress, an object implementing
        the :class:`myokit.ProgressReporter` interface can be passed in.
        passed in as ``progress``. An optional description of the current
        simulation to use in the ProgressReporter can be passed in as `msg`.
        """
        filename = os.path.expanduser(filename)
        # Load compression modules
        import zipfile
        try:        
            import zlib
        except:
            raise Exception('This method requires the ``zlib`` module to be'
                ' installed.')
        # Get size of single and double types on this machine
        dsize = {'d' : len(array.array('d', [1]).tostring()),
                 'f' : len(array.array('f', [1]).tostring())}
        # Read data from file
        try:
            f = None
            f = zipfile.ZipFile(filename, 'r')
            info = f.infolist()
            if len(info) < 2:
                raise myokit.DataBlockReadError('Invalid DataBlock1d file'
                    ' format: Not enough files in zip.')
            # Get ZipInfo objects
            names = [x.filename for x in info]
            try:
                head = names.index('header_block1d.txt')
            except ValueError:
                raise myokit.DataBlockReadError('Invalid DataBlock1d file'
                    ' format: header not found.')
            try:
                body = names.index('data.bin')
            except ValueError:
                raise myokit.DataBlockReadError('Invalid DataBlock1d file'
                    ' format: data not found.')
            # Read head and body into memory (let's assume it fits...)
            head = f.read(info[head])
            body = f.read(info[body])
        except zipfile.BadZipfile:
            raise myokit.DataBlockReadError('Unable to read DataBlock1d: bad'
                ' zip file.')
        except zipfile.LargeZipFile:
            raise myokit.DataBlockReadError('Unable to read DataBlock1d: zip'
                ' file requires zip64 support and this has not been enabled on'
                ' this system.')
        finally:
            if f:
                f.close()
        # Parse head
        head = head.splitlines()
        try:
            if progress:
                progress.enter(msg)
                # Avoid divide by zero
                fraction = float(len(head) - 3)
                if fraction > 0:
                    fraction = 1.0 / fraction
                iprogress = 0
                progress.update(iprogress * fraction)
            head = iter(head)
            nt = int(head.next())
            nx = int(head.next())
            dtype = str(head.next())[1:-1]
            if dtype not in dsize:
                raise myokit.DataBlockReadError('Unable to read DataBlock1d:'
                    ' unrecognized data type "' + str(dtype) + '".')
            names_0d = []
            names_1d = []
            name = head.next()
            while name != '1':
                names_0d.append(name[1:-1])
                name = head.next()
            for name in head:
                names_1d.append(name[1:-1])
            del(head)
            # Parse body
            start, end = 0, 0
            n0 = dsize[dtype] * nt
            n1 = n0 * nx
            nb = len(body)
            # Read time
            end += n0
            if end > nb:
                raise myokit.DataBlockReadError('Unable to read DataBlock1d:'
                    ' header indicates larger data than found in the body.')
            data = array.array(dtype)
            data.fromstring(body[start:end])
            if sys.byteorder == 'big':
                data.byteswap()
            data = np.array(data)
            if progress:
                iprogress += 1
                if not progress.update(iprogress * fraction):
                    return
            # Create data block
            block = DataBlock1d(nx, data, copy=False)
            # Read 0d data
            for name in names_0d:
                start = end
                end += n0
                if end > nb:
                    raise myokit.DataBlockReadError('Unable to read'
                        ' DataBlock1d: header indicates larger data than found'
                        ' in the body.')
                data = array.array(dtype)
                data.fromstring(body[start:end])
                if sys.byteorder == 'big':
                    data.byteswap()
                data = np.array(data)
                block.set0d(name, data, copy=False)
                if progress:
                    iprogress += 1
                    if not progress.update(iprogress * fraction):
                        return
            # Read 1d data
            for name in names_1d:
                start = end
                end += n1
                if end > nb:
                    raise myokit.DataBlockReadError('Unable to read'
                        ' DataBlock1d: header indicates larger data than found'
                        ' in the body.')
                data = array.array(dtype)
                data.fromstring(body[start:end])
                if sys.byteorder == 'big':
                    data.byteswap()
                data = np.array(data).reshape(nt, nx, order='C')
                block.set1d(name, data, copy=False)
                if progress:
                    iprogress += 1
                    if not progress.update(iprogress * fraction):
                        return
            return block
        finally:
            if progress:
                progress.exit()
    def save(self, filename):
        """
        Writes this ``DataBlock1d`` to a binary file.
        
        The resulting file will be a zip file with the following entries:
        
        ``header_block1d.txt``: A header file containing the following
        information (line by line):
        
        - ``nt`` the number of points in time in each entry
        - ``nx`` the length of each 1d block
        - ``"dtype"`` the used datatype (either "d" or "f")
        - ``"name"`` the names of all 0d entries, each on its own line
        - ``1`` the indication that the 1d entries are starting
        - ``"name"`` the names of all 1d entries, each on its own line
        
        ``data.bin``: A binary file containing the following data, in the data
        type specified by the header, and little-endian:
        
        - The ``nt`` time values
        - All 0d entries
        - All 1d entries, reshaped using numpy order='C'

        """
        # Check filename
        filename = os.path.expanduser(filename)        
        # Load compression modules
        import zipfile
        try:        
            import zlib
        except:
            raise Exception('This method requires the ``zlib`` module to be'
                ' installed.')
        # Data type
        dtype = 'd' # Only supporting doubles right now
        # Create header
        head_str = []
        head_str.append(str(self._nt))
        head_str.append(str(self._nx))
        head_str.append('"' + dtype + '"')
        for name in self._0d:
            head_str.append('"' + name + '"')
        head_str.append(str(1))
        for name in self._1d:
            head_str.append('"' + name + '"')
        head_str = '\n'.join(head_str)
        # Create body
        n = self._nt * self._nx
        body_str = []
        body_str.append(array.array(dtype, self._time))
        for name, data in self._0d.iteritems():
            body_str.append(array.array(dtype, data))
        for name, data in self._1d.iteritems():
            body_str.append(array.array(dtype, data.reshape(n, order='C')))
        if sys.byteorder == 'big':
            for ar in body_str:
                ar.byteswap()
        body_str = ''.join([ar.tostring() for ar in body_str])
        # Write
        head = zipfile.ZipInfo('header_block1d.txt')
        head.compress_type = zipfile.ZIP_DEFLATED
        body = zipfile.ZipInfo('data.bin')
        body.compress_type = zipfile.ZIP_DEFLATED
        read = zipfile.ZipInfo('readme.txt')
        read.compress_type = zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(filename, 'w') as f:
            f.writestr(head, head_str)
            f.writestr(body, body_str)
            f.writestr(read, README_SAVE_1D)
    def set0d(self, name, data, copy=True):
        """
        Adds or updates a zero-dimensional time series ``data`` for the
        variable named by the string ``name``.
        
        The ``data`` must be specified as a sequence of length ``n``, where
        ``n`` is the first value returned by :meth:`DataBlock1d.shape()`.
        
        By default, a copy of the given data will be stored. To prevent this
        and store a reference instead, set ``copy=False``.
        """
        name = str(name)
        if not name:
            raise ValueError('Name cannot be empty.')
        data = np.array(data, copy=copy)
        if data.shape != (self._nt,):
            raise ValueError('Data must be sequence of length ' + str(self._nt)
                + '.')
        self._0d[name] = data
    def set1d(self, name, data, copy=True):
        """
        Adds or updates a one-dimensional time series ``data`` for the variable
        named by the string ``name``.
        
        The ``data`` must be specified as a numpy array with shape ``(n, w)``,
        where ``(n, w)`` is the value returned by :meth:`DataBlock1d.shape()`.
        
        By default, a copy of the given data will be stored. To prevent this
        and store a reference instead, set ``copy=False``.
        """
        name = str(name)
        if not name:
            raise ValueError('Name cannot be empty.')
        data = np.array(data, copy=copy)
        shape = (self._nt, self._nx)
        if data.shape != shape:
            raise ValueError('Data must have shape ' + str(shape) + '.')
        self._1d[name] = data
    def shape(self):
        """
        Returns the required shape for 1d data passed to this data block. Zero
        dimensional series passed in must have length ``shape()[0]``.
        """
        return (self._nt, self._nx)
    def time(self):
        """
        Returns the time data for this datablock. The data is returned
        directly, no copy is made.
        """
        return self._time
    def trace(self, variable, x):
        """
        Returns a 0d time series of the value ``variable``, corresponding to
        the cell at position ``x``. The data is returned directly, no copy is
        made.
        """
        return self._1d[variable][:,x]
class DataBlock2d(object):
    """
    Container for time-series of 2d rectangular data arrays.
    
    Each ``DataBlock2d`` has a fixed width ``w`` and height ``h``, and a 0d
    time series vector containing a sequence of ``n`` times.
    
    Two-dimensional time series can be added to the block, provided the data
    also contains ``n`` instances of ``w`` by ``h`` data points. The
    data should be passed in as a numpy array with shape ``(n, h, w)``.
    
    Zero-dimensional time series can be added, provided they have length ``n``.
    
    A "two-dimensional time-series" is a series of equally sized 2d arrays
    (sequences), such that the first corresponds to a time ``t[0]``, the second
    to ``t[1]``, etc.
    
    A "zero-dimensional time-series" is a series of single values where the
    first corresponds to a time ``t[0]``, the second to ``t[1]``, etc.
    
    Constructor info:
    
    ``w``
        The width of a 2d block. Each block should have shape (n, h, w)
    ``h``
        The height of a 2d block. Each block should have shape (n, h, w)
    ``time``
        A sequence of ``n`` times.
    ``copy``
        By default, a copy of the given time sequence will be stored. To
        prevent this copy, set ``copy=False``.

    """
    def __init__(self, w, h, time, copy=True):
        # Width and height
        w, h = int(w), int(h)
        if w < 1:
            raise ValueError('Minimum width is 1.')
        if h < 1:
            raise ValueError('Minimum height is 1.')
        self._ny = h
        self._nx = w        
        # Time
        time = np.array(time, copy=copy)
        if len(time.shape) != 1:
            raise ValueError('Time must be a sequence.')
        if not np.all(np.diff(time) >= 0):
            raise ValueError('Time must be non-decreasing.')
        self._time = time
        self._nt = len(time)
        # 0d variables
        self._0d = {}
        # 2d variables
        self._2d = {}
    def colors(self, name, colormap='traditional', lower=None, upper=None):
        """
        Converts the 2d series indicated by ``name`` into a list of ``W*H*RGB``
        arrays, with each entry represented as an 8 bit unsigned integer.
        """
        data = self._2d[name]
        # Get color map
        color_map = ColorMap.get(colormap)
        # Get lower and upper bounds for colormap scaling
        lower = np.min(data) if lower is None else float(lower)
        upper = np.max(data) if upper is None else float(upper)
        # Create images
        frames = []
        for frame in data:
            # Convert 2d array into row-strided array
            frame = frame.reshape(self._ny * self._nx, order='C')
            # Apply colormap
            frame = color_map(frame, lower=lower, upper=upper, alpha=False,
                rgb=True)
            # Reshape to nx * ny * 3 color array
            frame = frame.reshape((self._ny, self._nx, 3))
            # Append to list
            frames.append(frame)
        return frames
    @staticmethod
    def combine(block1, block2, map2d, map0d=None, pos1=None, pos2=None):
        """
        Combines two blocks, containing information about different areas, into
        a single :class:`DataBlock2d`.
        
        Both blocks must contain data from the same points in time.
        
        A mapping from old to new variables must be passed in as a dictionary
        ``map2d``. The blocks can have different sizes but must have the same
        time vector. If any empty space is created it is padded with a value
        taken from one of the data blocks or a padding value specified as part
        of ``map2d``.
        
        Positions for the datablocks can be specified as ``pos1`` and ``pos2``,
        the new datablock will have indices ranging from ``(0, 0)`` to
        ``(max(pos1[0] + w1, pos2[0] + w2), max(pos1[0] + w1, pos2[0] + w2))``,
        where ``w1`` and ``w2`` are the widths of ``block1`` and ``block2``
        respectively. Negative indices are not supported and the blocks are not
        allowed to overlap.
        
        Arguments:
        
        ``block1``
            The first DataBlock2d
        ``block2``
            The second DataBlock2d. This must have the same time vector as the
            first.
        ``map2d``
            A dictionary object showing how to map 2d variables from both
            blocks into the newly created datablock. The format must be:
            ``new_name : (old_name_1, old_name_2, padding_value)``. Here,
            ``new_name`` is the name of the new 2d variable, ``old_name_1`` is
            the name of a 2d variable in ``block1``, ``old_name_2`` is the name
            of a 2d variable in ``block2`` and ``padding_value`` is an optional
            value indicating the value to use for undefined spaces in the new
            block.
        ``map0d=None``,
            A dictionary object showing how to map 0d variables from both
            blocks into the newly created datablock. Each entry must take the
            format: ``new_name : (old_name_1, None)`` or
            ``new_name : (None, old_name_2)``.
        ``pos1=None``
            Optional value indicating the position ``(x, y)`` of the first
            datablock. By default ``(0, 0)`` is used.
        ``pos2=None``
            Optional value indicating the position ``(x, y)`` of the first
            datablock. By default ``(w1, 0)`` is used, where ``w1`` is the
            width of ``block1``.
    
        """
        # Check time vector
        time = block1.time()
        if not np.allclose(time, block2.time()):
            raise ValueError('Both datablocks must contain data from the same'
                ' points in time.')
        # Check indices
        nt, h1, w1 = block1.shape()
        nt, h2, w2 = block2.shape()
        if pos1:
            x1, y1 = [int(i) for i in pos1]
            if x1 < 0 or y1 < 0:
                raise ValueError('Negative indices not supported: pos1=('
                    + str(x1) + ', ' + str(y1) + ').')
        else:
            x1, y1 = 0, 0
        if pos2:
            x2, y2 = [int(i) for i in pos2]
            if x2 < 0 or y2 < 0:
                raise ValueError('Negative indices not supported: pos2=('
                    + str(x2) + ', ' + str(y2) + ').')
        else:
            x2, y2 = x1 + w1, 0
        # Check for overlap
        if not (x1 >= x2+w2 or x2 >= x1+w1 or y1 >= y2+h2 or y2 >= y1+h2):
            raise ValueError('The two data blocks indices cannot overlap.')
        # Create new datablock
        nx = max(x1 + w1, x2 + w2)
        ny = max(y1 + h1, y2 + h2)
        block = DataBlock2d(nx, ny, time, copy=True)
        # Enter 0d data
        if map0d:
            for name, old in map0d.iteritems():
                if old[0] is None:
                    b = block2
                    n = old[1]
                elif old[1] is None:
                    b = block1
                    n = old[0]
                else:
                    raise ValueError('The dictionary map0d must map the names'
                        ' of new 0d entries to a tuple (a, b) where either a'
                        ' or b must be None.')
                block.set0d(name, b.get0d(n))
        # Enter 2d data
        for name, source in map2d.iteritems():
            # Get data sources
            name1, name2 = source[0], source[1]
            source1 = block1.get2d(name1)
            source2 = block2.get2d(name2)
            # Get padding value
            try:
                pad = float(source[2])
            except IndexError:
                # Get the first value, of a cell not likely to be paced
                pad = source1[0, int(h1/2), int(w1/2)]
                # Compare to see which side of the mean it's on
                mean = 0.5 * (np.mean(source1) + np.mean(source2))
                if pad > mean:
                    pad = max(np.max(source1), np.max(source2))
                else:
                    pad = min(np.min(source1), np.min(source2))
            # Create new data field
            field = pad * np.ones((nt, ny, nx))
            field[:, y1:y1+h1, x1:x1+w1] = source1
            field[:, y2:y2+h2, x2:x2+w2] = source2
            block.set2d(name, field)
        # Return new block
        return block
    def dominant_eigenvalues(self, name):
        """
        Takes the 2d data specified by ``name`` and computes the dominant
        eigenvalue for each point in time. This only works for datablocks with
        a square 2d grid.
        
        The "dominant eigenvalue" is defined as the eigenvalue with the largest
        magnitude (``sqrt(a + bi)``).
        
        The returned data is a 1d numpy array.
        """
        if self._nx != self._ny:
            raise Exception('Eigenvalues can only be determined for square'
                ' data blocks.')
        data = self._2d[name]
        dominants = []
        for t in xrange(self._nt):
            e = np.linalg.eigvals(data[t])
            dominants.append(e[np.argmax(np.absolute(e))])
        return np.array(dominants)
    def eigenvalues(self, name):
        """
        Takes the 2d data specified as ``name`` and computes the eigenvalues of
        its data matrix at every point in time. This only works for datablocks
        with a square 2d grid.
        
        The returned data is a 2d numpy array where the first axis is time and
        the second axis is the index of each eigenvalue.
        """
        if self._nx != self._ny:
            raise Exception('Eigenvalues can only be determined for square'
                ' data blocks.')
        data = self._2d[name]
        eigenvalues = []
        for t in xrange(self._nt):
            eigenvalues.append(np.linalg.eigvals(data[t]))
        return np.array(eigenvalues)
    @staticmethod
    def fromDataLog(log):
        """
        Creates a DataBlock2d from a :class:`myokit.DataLog`.
        """
        log.validate()
        # Get time variable name
        time = log.time_key()
        if time is None:
            raise ValueError('No time variable set in data log.')
        # Get log info
        infos = log.variable_info()
        # Check time variable
        info = infos[time]
        if info.dimension() != 0:
            raise ValueError('The given time variable should be 0d.')
        # Check if everything is 0d or 2d, get size
        size = None
        for name, info in infos.iteritems():
            d = info.dimension()
            if d not in (0, 2):
                raise ValueError('The given simulation log should only contain'
                    ' 0d or 2d variables. Found <' + str(name) + '> with d = '
                    + str(d) + '.')
            if d == 2:
                if size is None:
                    size = info.size()
                elif info.size() != size:
                    raise ValueError('The given simulation log contains 2d'
                        ' data sets of different sizes.')
        # Get dimensions
        nt = len(log[time])
        nx, ny = size
        # Create data block
        block = DataBlock2d(nx, ny, log[time], copy=True)
        for name, info in infos.iteritems():
            if info.dimension() == 0:
                # Add 0d time series
                if name == time:
                    continue
                block.set0d(name, log[name], copy=True)
            else:
                # Convert to 2d time series
                data = np.zeros(nt * ny * nx)
                # Iterate over info.keys()
                for i, key in enumerate(info.keys()):
                    # Copy data into array (really copies)
                    data[i*nt:(i+1)*nt] = log[key]
                # Reshape
                data = data.reshape((nt, ny, nx), order='F')
                # If this is a view of existing data, make a copy!
                if data.base is not None:                    
                    data = np.array(data)
                block.set2d(name, data, copy=False)
        return block
    def get0d(self, name):
        """
        Returns the 0d time-series identified by ``name``. The data is returned
        directly, no copy is made.
        """
        return self._0d[name]
    def get2d(self, name):
        """
        Returns the 2d time-series identified by ``name``. The data is returned
        directly, no copy is made.
        """
        return self._2d[name]
    def images(self, name, colormap='traditional', lower=None, upper=None):
        """
        Converts the 2d series indicated by ``name`` into a list of 1d arrays
        in a row-strided image format ``ARGB32``.
        """
        data = self._2d[name]
        # Get color map
        color_map = ColorMap.get(colormap)
        # Get lower and upper bounds for colormap scaling
        lower = np.min(data) if lower is None else float(lower)
        upper = np.max(data) if upper is None else float(upper)
        # Create images
        frames = []
        for frame in data:
            # Convert 2d array into row-strided array
            frame = frame.reshape(self._ny * self._nx, order='C')
            frames.append(color_map(frame, lower=lower, upper=upper))
        return frames
    def is_square(self):
        """
        Returns True if this data block's grid is square.
        """
        return self._nx == self._ny
    def items0d(self):
        """
        Returns an iterator over ``(name, value)`` pairs for the 0d series
        stored in this block. The given values are references! No copy of the
        data is made.
        """
        return self._2d.iteritems()
    def items2d(self):
        """
        Returns an iterator over ``(name, value)`` pairs for the 2d series
        stored in this block. The given values are references! No copy of the
        data is made.
        """
        return self._2d.iteritems()
    def keys0d(self):
        """
        Returns an iterator over this block's 0d time series.
        """
        return iter(self._0d)
    def keys2d(self):
        """
        Returns an iterator over this block's 2d time series.
        """
        return iter(self._2d)
    def largest_eigenvalues(self, name):
        """
        Takes the 2d data specified by ``name`` and computes the largest
        eigenvalue for each point in time. This only works for datablocks with
        a square 2d grid.
        
        The "largest eigenvalue" is defined as the eigenvalue with the most
        positive real part. Note that the returned values may be complex.
        
        The returned data is a 1d numpy array.
        """
        if self._nx != self._ny:
            raise Exception('Eigenvalues can only be determined for square'
                ' data blocks.')
        data = self._2d[name]
        largest = []
        for t in xrange(self._nt):
            e = np.linalg.eigvals(data[t])
            largest.append(e[np.argmax(np.real(e))])
        return np.array(largest)
    def len0d(self):
        """
        Returns the number of 0d time series in this block.
        """
        return len(self._0d)
    def len2d(self):
        """
        Returns the number of 2d time series in this block.
        """
        return len(self._2d)
    @staticmethod
    def load(filename, progress=None, msg='Loading DataBlock2d'):
        """
        Loads a :class:`DataBlock2d` from the specified file.
        
        To obtain feedback on the simulation progress, an object implementing
        the :class:`myokit.ProgressReporter` interface can be passed in.
        passed in as ``progress``. An optional description of the current
        simulation to use in the ProgressReporter can be passed in as `msg`.
        
        If the given file contains a :class:`DataBlock1d` this is read and
        converted to a 2d block without warning.
        """
        filename = os.path.expanduser(filename)
        # Load compression modules
        import zipfile
        try:        
            import zlib
        except:
            raise Exception('This method requires the ``zlib`` module to be'
                ' installed.')
        # Get size of single and double types on this machine
        dsize = {'d' : len(array.array('d', [1]).tostring()),
                 'f' : len(array.array('f', [1]).tostring())}
        # Read data from file
        try:
            f = None
            f = zipfile.ZipFile(filename, 'r')
            info = f.infolist()
            if len(info) < 2:
                raise myokit.DataBlockReadError('Invalid DataBlock2d file'
                    ' format: not enough files in zip.')
            # Get ZipInfo objects
            names = [x.filename for x in info]
            try:
                head = names.index('header_block2d.txt')
            except ValueError:
                # Attempt reading as DataBlock1d
                try:
                    head = names.index('header_block1d.txt')
                except ValueError:
                    raise myokit.DataBlockReadError('Invalid DataBlock2d file'
                        ' format: header not found.')
                # It's a DataBlock1d, attempt reading as such
                f.close()
                return DataBlock1d.load(filename, progress, msg).block2d()
            try:
                body = names.index('data.bin')
            except ValueError:
                raise myokit.DataBlockReadError('Invalid DataBlock2d file'
                    ' format: data file not found.')
            # Read head and body into memory (let's assume it fits...)
            head = f.read(info[head])
            body = f.read(info[body])
        except zipfile.BadZipfile:
            raise myokit.DataBlockReadError('Unable to read DataBlock2d: bad'
                ' zip file.')
        except zipfile.LargeZipFile:
            raise myokit.DataBlockReadError('Unable to read DataBlock2d: zip'
                ' file requires zip64 support and this has not been enabled on'
                ' this system.')
        finally:
            if f:
                f.close()
        # Parse head
        head = head.splitlines()
        try:
            if progress:
                progress.enter(msg)
                # Avoid divide by zero
                fraction = float(len(head) - 4)
                if fraction > 0:
                    fraction = 1.0 / fraction
                iprogress = 0
                progress.update(iprogress * fraction)
            head = iter(head)
            nt = int(head.next())
            ny = int(head.next())
            nx = int(head.next())
            dtype = str(head.next())[1:-1]
            if dtype not in dsize:
                raise myokit.DataBlockReadError('Unable to read DataBlock2d:'
                    ' unrecognized data type "' + str(dtype) + '".')
            names_0d = []
            names_2d = []
            name = head.next()
            while name != '2':
                names_0d.append(name[1:-1])
                name = head.next()
            for name in head:
                names_2d.append(name[1:-1])
            del(head)
            # Parse body
            start, end = 0, 0
            n0 = dsize[dtype] * nt
            n2 = n0 * ny * nx
            nb = len(body)
            # Read time
            end += n0
            if end > nb:
                raise myokit.DataBlockReadError('Unable to read DataBlock2d:'
                    ' header indicates larger data than found in the body.')
            data = array.array(dtype)
            data.fromstring(body[start:end])
            if sys.byteorder == 'big':
                data.byteswap()
            data = np.array(data)
            if progress:
                iprogress += 1
                if not progress.update(iprogress * fraction):
                    return
            # Create data block
            block = DataBlock2d(nx, ny, data, copy=False)
            # Read 0d data
            for name in names_0d:
                start = end
                end += n0
                if end > nb:
                    raise myokit.DataBlockReadError('Unable to read'
                        ' DataBlock2d: header indicates larger data than found'
                        ' in the body.')
                data = array.array(dtype)
                data.fromstring(body[start:end])
                if sys.byteorder == 'big':
                    data.byteswap()
                data = np.array(data)
                block.set0d(name, data, copy=False)
                if progress:
                    iprogress += 1
                    if not progress.update(iprogress * fraction):
                        return
            # Read 2d data
            for name in names_2d:
                start = end
                end += n2
                if end > nb:
                    raise myokit.DataBlockReadError('Unable to read'
                        ' DataBlock2d: header indicates larger data than found'
                        ' in the body.')
                data = array.array(dtype)
                data.fromstring(body[start:end])
                if sys.byteorder == 'big':
                    data.byteswap()
                data = np.array(data).reshape(nt, ny, nx, order='C')
                block.set2d(name, data, copy=False)
                if progress:
                    iprogress += 1
                    if not progress.update(iprogress * fraction):
                        return
            return block
        finally:
            if progress:
                progress.exit()
    def save(self, filename):
        """
        Writes this ``DataBlock2d`` to a binary file.
        
        The resulting file will be a zip file with the following entries:
        
        ``header_block2d.txt``: A header file containing the following
        information (line by line):
        
        - ``nt`` the number of points in time in each entry
        - ``ny`` the height of each 2d block
        - ``nx`` the width of each 2d block
        - ``"dtype"`` the used datatype (either "d" or "f")
        - ``"name"`` the names of all 0d entries, each on its own line
        - ``2`` the indication that the 2d entries are starting
        - ``"name"`` the names of all 2d entries, each on its own line
        
        ``data.bin``: A binary file containing the following data, in the data
        type specified by the header, and little-endian:
        
        - The ``nt`` time values
        - All 0d entries
        - All 2d entries, reshaped using numpy order='C'

        """
        # Check filename
        filename = os.path.expanduser(filename)
        # Load compression modules
        import zipfile
        try:        
            import zlib
        except:
            raise Exception('This method requires the ``zlib`` module to be'
                ' installed.')
        # Data type
        dtype = 'd' # Only supporting doubles right now
        # Create header
        head_str = []
        head_str.append(str(self._nt))
        head_str.append(str(self._ny))
        head_str.append(str(self._nx))
        head_str.append('"' + dtype + '"')
        for name in self._0d:
            head_str.append('"' + name + '"')
        head_str.append(str(2))
        for name in self._2d:
            head_str.append('"' + name + '"')
        head_str = '\n'.join(head_str)
        # Create body
        n = self._nt * self._ny * self._nx
        body_str = []
        body_str.append(array.array(dtype, self._time))
        for name, data in self._0d.iteritems():
            body_str.append(array.array(dtype, data))
        for name, data in self._2d.iteritems():
            body_str.append(array.array(dtype, data.reshape(n, order='C')))
        if sys.byteorder == 'big':
            for ar in body_str:
                ar.byteswap()
        body_str = ''.join([ar.tostring() for ar in body_str])
        # Write
        head = zipfile.ZipInfo('header_block2d.txt')
        head.compress_type = zipfile.ZIP_DEFLATED
        body = zipfile.ZipInfo('data.bin')
        body.compress_type = zipfile.ZIP_DEFLATED
        read = zipfile.ZipInfo('readme.txt')
        read.compress_type = zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(filename, 'w') as f:
            f.writestr(head, head_str)
            f.writestr(body, body_str)
            f.writestr(read, README_SAVE_2D)
    def save_frame_csv(self, filename, name, frame, xname='x', yname='y',
            zname='value'):
        """
        Stores a single 2d variable's data at a single point in time to disk,
        using a csv format where each point in the frame is stored on a
        separate line as a tuple ``x, y, value``.
        """
        # Check filename
        filename = os.path.expanduser(filename)
        # Save
        delimx = ','
        delimy = '\n'
        data = self._2d[name]
        data = data[frame]
        text = [delimx.join('"' + str(x) + '"' for x in [xname, yname, zname])]
        for y, row in enumerate(data):
            for x, z in enumerate(row):
                text.append(delimx.join([str(x), str(y), myokit.strfloat(z)]))
        text = delimy.join(text)
        with open(filename, 'w') as f:
            f.write(text)
    def save_frame_grid(self, filename, name, frame, delimx=' ', delimy='\n'):
        """
        Stores a single 2d variable's data at a single point in time to disk,
        using a simple 2d format where each row of the resulting data file
        represents a row of the frame.
        
        Data from 2d variable ``name`` at frame ``frame`` will be stored in
        ``filename`` row by row. Each column is separated by ``delimx`` (by
        default a space) and rows are separated by ``delimy`` (by default this
        will be a newline character).
        """
        # Check filename
        filename = os.expanduser(filename)
        # Save
        data = self._2d[name]
        data = data[frame]
        text = []
        for row in data:
            text.append(delimx.join([myokit.strfloat(x) for x in row]))
        text = delimy.join(text)
        with open(filename, 'w') as f:
            f.write(text)
    def set0d(self, name, data, copy=True):
        """
        Adds or updates a zero-dimensional time series ``data`` for the
        variable named by the string ``name``.
        
        The ``data`` must be specified as a sequence of length ``n``, where
        ``n`` is the first value returned by :meth:`DataBlock2d.shape()`.
        
        By default, a copy of the given data will be stored. To prevent this
        and store a reference instead, set ``copy=False``.
        """
        name = str(name)
        if not name:
            raise ValueError('Name cannot be empty.')
        data = np.array(data, copy=copy)
        if data.shape != (self._nt,):
            raise ValueError('Data must be sequence of length ' + str(self._nt)
                + '.')
        self._0d[name] = data
    def set2d(self, name, data, copy=True):
        """
        Adds or updates a two-dimensional time series ``data`` for the variable
        named by the string ``name``.
        
        The ``data`` must be specified as a numpy array with shape ``(n, w)``,
        where ``(n, w)`` is the value returned by :meth:`DataBlock2d.shape()`.
        
        By default, a copy of the given data will be stored. To prevent this
        and store a reference instead, set ``copy=False``.
        """
        name = str(name)
        if not name:
            raise ValueError('Name cannot be empty.')
        data = np.array(data, copy=copy)
        shape = (self._nt, self._ny, self._nx)
        if data.shape != shape:
            raise ValueError('Data must have shape ' + str(shape) + '.')
        self._2d[name] = data
    def shape(self):
        """
        Returns the required shape for 2d data passed to this data block. Zero
        dimensional series passed in must have length ``shape()[0]``.
        """
        return (self._nt, self._ny, self._nx)
    def time(self):
        """
        Returns the time data for this datablock. The data is returned
        directly, no copy is made.
        """
        return self._time
    def trace(self, variable, x, y):
        """
        Returns a 0d time series of the value ``variable``, corresponding to
        the cell at position ``x``, ``y``. The data is returned directly, no
        copy is made.
        """
        return self._2d[variable][:,y,x]
class ColorMapMeta(type):
    """
    Meta-class for colormap interface.
    """
    def __init__(cls, name, bases, dct):
        if not hasattr(cls, '_colormaps'):
            # Base class. Create empty dict.
            cls._colormaps = {}
        else:
            # Derived class. Add to dict.
            mapname = name[8:].lower()
            cls._colormaps[mapname] = cls
        super(ColorMapMeta, cls).__init__(name, bases, dct)
class ColorMap(object):
    """
    *Abstract class*
    
    Applies colormap transformations to floating point data and returns RGB
    data.
    
    :class:`ColorMaps <ColorMap>` are callable objects and take the following
    arguments:
    
    ``floats``
        A 1-dimensional numpy array of floating point numbers.
    ``lower=None``
        A lower bound for the floats in the input. The ``lower`` and ``upper``
        values are used to normalize the input before applying the colormap. If
        this bound is omitted the lowest value in the input data is used.
    ``upper=None``
        An upper bound for the floats in the input. The ``lower`` and ``upper``
        values are used to normalize the input before applying the colormap. If
        this bound is omitted the highest value in the input data is used.
    ``alpha=True``
        Set to ``False`` to omit an alpha channel from the output.
    ``rgb=None``
        Set to ``True`` to return bytes in the order ``0xARGB``, to ``False``
        to return the order ``0xBGRA`` or to ``None`` to let the system's
        endianness determine the correct order. In the last case, big-endian
        systems will return ``0xARGB`` while little-endian systems use the
        order ``0xBGRA``.
    
    A 1-dimensional array of ``n`` floating point numbers will be converted to
    a 1-dimensional array of ``4n`` ``uints``, or ``3n`` if the alpha channel
    is disabled. The array will be ordered sequentially: the first four (or
    three) bytes describe the first float, the next four (or three) describe
    the second float and so on.
    """
    __metaclass__ = ColorMapMeta
    def __call__(floats, lower=None, upper=None, alpha=True, rgb=None):
        raise NotImplementedError
    @staticmethod
    def exists(name):
        """
        Returns True if the given name corresponds to a colormap.
        """
        return name in ColorMap._colormaps
    @staticmethod
    def get(name):
        """
        Returns the colormap method indicated by the given name.
        """
        try:
            return ColorMap._colormaps[name]()
        except KeyError:
            raise KeyError('Non-existent ColorMap "' + str(name) + '".')
    @staticmethod  
    def hsv_to_rgb(h, s, v):
        """
        Converts hsv values in the range [0,1] to rgb values in the range
        [0,255]. Adapted from Matplotlib.
        """
        r, g, b = np.empty_like(h), np.empty_like(h), np.empty_like(h)
        i = (h * 6).astype(np.int) % 6
        f = (h * 6) - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        idx = (i == 0)
        r[idx], g[idx], b[idx] = v[idx], t[idx], p[idx]
        idx = (i == 1)
        r[idx], g[idx], b[idx] = q[idx], v[idx], p[idx]
        idx = (i == 2)
        r[idx], g[idx], b[idx] = p[idx], v[idx], t[idx]
        idx = (i == 3)
        r[idx], g[idx], b[idx] = p[idx], q[idx], v[idx]
        idx = (i == 4)
        r[idx], g[idx], b[idx] = t[idx], p[idx], v[idx]
        idx = (i == 5)
        r[idx], g[idx], b[idx] = v[idx], p[idx], q[idx]
        out = (
            np.array(r*255, dtype=np.uint8, copy=False),
            np.array(g*255, dtype=np.uint8, copy=False),
            np.array(b*255, dtype=np.uint8, copy=False),
            )
        return out
    @staticmethod
    def image(name, x, y):
        """
        Returns image data (such as returned by :meth:`DataBlock2d.images()`)
        representing the colormap specified by ``name``. The image dimensions
        can be set using ``x`` and ``y``.
        """
        data = np.linspace(1, 0, y)
        data = np.tile(data, (x, 1)).transpose()
        data = np.reshape(data, (1, y, x))
        block = myokit.DataBlock2d(x, y, [0])
        block.set2d('colormap', data, copy=False)
        return block.images('colormap', colormap=name)[0]
    @staticmethod
    def names():
        """
        Returns an iterator over the names of all available colormaps.
        """
        return ColorMap._colormaps.iterkeys()
    @staticmethod
    def normalize(floats, lower=None, upper=None):
        """
        Normalizes the given float data based on the specified lower and upper
        bounds. If no bounds are set, the scaling is deduced from the data.
        """
        floats = np.array(floats, copy=True)
        # Find or enforce lower and upper bounds
        if lower is None:
            lower = np.min(floats)
        else:
            floats[floats<lower] = lower
        if upper is None:
            upper = np.max(floats)
        else:
            floats[floats>upper] = upper
        # Normalize
        n = floats - lower
        r = upper - lower
        if r == 0:
            return n
        else:
            return n / r
class ColorMapBlue(ColorMap):
    """
    A nice red colormap.
    """
    def __call__(self, floats, lower=None, upper=None, alpha=True, rgb=None):
        # Normalize floats
        f = ColorMap.normalize(floats, lower, upper)
        # Calculate h,s,v and convert to rgb
        h = np.zeros(f.shape)
        s = f
        v = np.ones(f.shape)
        b,g,r = ColorMap.hsv_to_rgb(h, s, v)
        # Color order
        rgb = (sys.byteorder == 'big') if rgb is None else rgb
        # Offset for first color in (a)rgb or rgb(a)
        m = 1 if (alpha and rgb) else 0
        # Number of bytes per float
        n = 4 if alpha else 3
        # Create output
        out = 255 * np.ones(n*len(floats), dtype=np.uint8)
        out[m + 0::n] = r if rgb else b
        out[m + 1::n] = g
        out[m + 2::n] = b if rgb else r
        return out
class ColorMapGreen(ColorMap):
    """
    A nice green colormap.
    """
    def __call__(self, floats, lower=None, upper=None, alpha=True, rgb=None):
        # Normalize floats
        f = ColorMap.normalize(floats, lower, upper)
        # Calculate h,s,v and convert to rgb
        h = np.zeros(f.shape)
        s = f
        v = np.ones(f.shape)
        g,r,b = ColorMap.hsv_to_rgb(h, s, v)
        # Color order
        rgb = (sys.byteorder == 'big') if rgb is None else rgb
        # Offset for first color in (a)rgb or rgb(a)
        m = 1 if (alpha and rgb) else 0
        # Number of bytes per float
        n = 4 if alpha else 3
        # Create output
        out = 255 * np.ones(n*len(floats), dtype=np.uint8)
        out[m + 0::n] = r if rgb else b
        out[m + 1::n] = g
        out[m + 2::n] = b if rgb else r
        return out
class ColorMapRed(ColorMap):
    """
    A nice red colormap.
    """
    def __call__(self, floats, lower=None, upper=None, alpha=True, rgb=None):
        # Normalize floats
        f = ColorMap.normalize(floats, lower, upper)
        # Calculate h,s,v and convert to rgb
        h = np.zeros(f.shape)
        s = f
        v = np.ones(f.shape)
        r,g,b = ColorMap.hsv_to_rgb(h, s, v)
        # Color order
        rgb = (sys.byteorder == 'big') if rgb is None else rgb
        # Offset for first color in (a)rgb or rgb(a)
        m = 1 if (alpha and rgb) else 0
        # Number of bytes per float
        n = 4 if alpha else 3
        # Create output
        out = 255 * np.ones(n*len(floats), dtype=np.uint8)
        out[m + 0::n] = r if rgb else b
        out[m + 1::n] = g
        out[m + 2::n] = b if rgb else r
        return out
class ColorMapTraditional(ColorMap):
    """
    Traditional hue-cycling colormap.
    """
    def __call__(self, floats, lower=None, upper=None, alpha=True, rgb=None):
        # Normalize floats
        f = ColorMap.normalize(floats, lower, upper)
        # Calculate h,s,v and convert to rgb
        g = 0.6
        s = g + (1 - g) * np.sin(f * 3.14)
        h = (0.85 - 0.85 * f) % 1
        r,g,b = ColorMap.hsv_to_rgb(h, s, s)
        # Color order
        rgb = (sys.byteorder == 'big') if rgb is None else rgb
        # Offset for first color in (a)rgb or rgb(a)
        m = 1 if (alpha and rgb) else 0
        # Number of bytes per float
        n = 4 if alpha else 3
        # Create output
        out = 255 * np.ones(n*len(floats), dtype=np.uint8)
        out[m + 0::n] = r if rgb else b
        out[m + 1::n] = g
        out[m + 2::n] = b if rgb else r
        return out
