#
# Functions relating to the DataLog class for storing time series data.
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
import re
import sys
import array
import numpy as np
#import operator
from collections import OrderedDict
import myokit
# Function to split keys into dimension-key,qname-key pairs
ID_NAME_PATTERN = re.compile(r'(\d+.)+')
# Readme file for DataLog binary files
README_SAVE_BIN = """
Myokit DataLog Binary File
--------------------------
This zip file contains binary time series data for one or multiple variables.
The file structure.txt contains structural information about the data in plain
text. The first line lists the number of fields. The second line gives the
length of the data arrays. The third line specifies the data type, either
single ("f") or double ("d") precision. The fourth line indicates which entry
corresponds to a time variable, or is blank if no time variable was explicitly
specified. Each following line contains the name of a data field, in the order
its data occurs in the binary data file "data.bin". All data is stored
little-endian.
""".strip()
class DataLog(OrderedDict):
    """
    A dictionary time-series, for example data logged during a simulation or
    experiment.
    
    A :class:`DataLog` is expected but not required to contain a single
    entry indicating time and any number of entries representing a variable
    varying over time.
    
    Single cell data is accessed simply by the variable name::
    
        v = log['membrane.V']
        
    Multi-cell data is accessed by appending the index of the cell before the
    variable name. For example::
    
        v = log['1.2.membrane.V']
        
    This returns the membrane potential for cell (1,2). Another way to obtain
    the same result is::
    
        v = log['membrane.V', 1, 2]
        
    or, finally:
    
        v = log['membrane.V', (1, 2)]
    
    Every array stored in the log must have the same length. This condition can
    be checked by calling the method :meth:`validate`.
    
    A new ``DataLog`` can be created in a number of ways:
    
        # Create an empty DataLog:
        d = myokit.DataLog()
        d['time'] = [1,2,3]
        d['data'] = [2,4,5]
        d.set_time_key('time')
        
        # Create a clone of d
        e = myokit.DataLog(d)
        
        # Create a DataLog based on a dictionary
        d = myokit.DataLog({'time':[1,2,3], 'data':[2,4,5]}, time='time')
        
    Arguments:
    
    ``other``
        A DataLog to clone or a dictionary to use as basis.
    ``time``
        The log key to use for the time variable. When cloning a log, adding
        the ``time`` argument will overwrite the cloned value.
    """
    def __init__(self, other=None, time=None):
        """
        Creates a new DataLog.
        """
        if other is None:
            # Create new
            super(DataLog, self).__init__()
            self._time = None
        else:
            # Clone
            super(DataLog, self).__init__(other)
            try:
                self._time = str(other._time)
            except Exception:
                self._time = None
        if time is not None:
            self.set_time_key(time)
    def apd(self, v='membrane.V', threshold=None):
        """
        Calculates one or more Action Potential Durations (APDs) in a single
        cell's membrane potential.

        *Note: More accuracte apd measurements can be created using the*
        :class:`Simulation` *object's APD tracking functionality. See*
        :meth:`Simulation.run()` *for details.*

        The membrane potential data should be listed in the log under the key
        given by ``v``.

        The APD is measured as the time that the membrane potential exceeds a
        certain, fixed, threshold. It does *not* calculate dynamic thresholds
        like "90% of max(V) - min(V)".

        The returned value is a list of tuples (AP_start, APD).
        """
        def crossings(x, y, t):
            """
            Calculates the ``x``-values where ``y`` crosses threshold ``t``.
            Returns a list of tuples ``(xc, sc)`` where ``xc`` is the ``x``
            coordinate of the crossing and ``sc`` is the slope at this point.
            """
            x = np.asarray(x)
            y = np.asarray(y)
            # Get boolean array of places where v exceeds the threshold
            h = y > t
            # Get boolean array of indices just before a crossing
            c = np.argwhere(h[1:] - h[:-1])
            # Gather crossing times
            crossings = []
            for i in c:
                i = i[0]
                sc = (y[i+1] - y[i]) / (x[i+1] - x[i])
                if y[i] == t:
                    xc = x[i]
                else:
                    xc = x[i] + (t - y[i]) / sc
                crossings.append((xc, sc))
            return crossings
        # Check time variable
        t = np.asarray(self.time())
        # Check voltage variable
        v = np.asarray(self[v])
        # Check threshold
        threshold = float(threshold)
        # Initial status: check if already in AP
        apds = []
        inap = v[0] >= threshold
        last = t[0]
        # Evaluate crossings
        for time, slope in crossings(t, v, threshold):
            if slope > 0:
                # New AP
                inap = True
            elif slope < 0:
                # End of AP
                inap = False
                if last != t[0]:    # Don't inlcude AP started before t[0]
                    apds.append((last, time - last))
            else:
                # This will never happen :)
                if inap and last != t[0]:
                    apds.append((last, time - last))
            last = time
        return apds
    def block1d(self):
        """
        Returns a copy of this log as a :class:`DataBlock1d`.
        """
        return myokit.DataBlock1d.fromDataLog(self)
    def block2d(self):
        """
        Returns a copy of this log as a :class:`DataBlock2d`.
        """
        return myokit.DataBlock2d.fromDataLog(self)
    def clone(self, numpy=False):
        """
        Returns a deep clone of this log.
        
        All lists in the log will be duplicated, but the list contents are
        assumed to be numerical (and thereby immutable) and won't be cloned.
        
        A log with numpy arrays instead of lists can be created by setting
        ``numpy=True``.
        """
        log = DataLog()
        log._time = self._time
        if numpy:
            for k, v in self.iteritems():
                log[str(k)] = np.array(v, copy=True)
        else:
            for k, v in self.iteritems():
                log[str(k)] = list(v)
        return log
    def __contains__(self, key):
        return super(DataLog, self).__contains__(self._parse_key(key))
    def __delitem__(self, key):
        return super(DataLog, self).__delitem__(self._parse_key(key))
    def extend(self, other):
        """
        Extends this :class:`myokit.DataLog` with the data from another.
        
        Both logs must have the same keys and the same time key. The added data
        must be from later time points than in the log being extended.
        """
        time = self._time
        if other._time != time:
            raise ValueError('Both logs must have the same time key.')
        if other[time][0] > self[time][-1]:
            raise ValueError('Cannot extend DataLog with data from an earlier'
                ' time.')
        if set(self.keys()) != set(other.keys()):
            raise ValueError('Both logs must have the same keys.')
        if isinstance(self[time], np.ndarray):
            # Numpy version
            for k, v in self.iteritems():
                self[k] = np.concatenate((np.asarray(v), np.asarray(other[k])))
        else:
            # List / array version
            for k, v in self.iteritems():
                v.extend(other[k])
    def find(self, value):
        """
        Searches for the indice of the first time where
        
            value <= t[i]
        
        If the given value doesn't occur in the log, ``len(t)`` is returned.
        """
        time = self.time()
        # Border cases
        n = len(time)
        if n == 0 or value <= time[0]:
            return 0
        if value > time[-1]:
            return n
        # Find value
        def find(lo, hi):
            # lo = first indice, hi = last indice + 1
            if (lo + 1 == hi):
                return lo + 1
            m = int((lo + hi) / 2)
            if value > time[m]:
                return find(m, hi)
            else:
                return find(lo, m)
        return find(0, n)
    def fold(self, period, discard_remainder=True):
        """
        Creates a copy of the log, split with the given period. Split signals
        are given indexes so that "current" becomes "0.current", "1.current"
        "2.current", etc.
        
        If the logs entries do not divide well by 'period', the remainder will
        be ignored. This happens commonly due to rounding point errors (in
        which case the remainder is a single entry). To disable this behavior,
        set `discard_remainder=False`
        """
        # Note: Using closed intervals can lead to logs of unequal length, so
        # it should be disabled here to ensure a valid log
        logs = self.split_periodic(period, adjust=True, closed_intervals=False)
        # Discard remainder if present
        if discard_remainder:
            if len(logs) > 1:
                n = logs[0].length()
                if logs[-1].length() < n:
                    logs = logs[:-1]
        # Create new log with folded data
        out = myokit.DataLog()
        out._time = self._time
        out[self._time] = logs[0][self._time]
        for i, log in enumerate(logs):
            pre = str(i) + '.'
            for k, v in log.iteritems():
                if k != self._time:
                    out[pre + k] = v
        return out
    def __getitem__(self, key):
        return super(DataLog, self).__getitem__(self._parse_key(key))
    def has_nan(self):
        """
        Returns True if one of the variables in this DataLog has a ``NaN`` as
        its final logged value.
        """
        for k, d in self.iteritems():
            if len(d) > 0 and np.isnan(d[-1]):
                return True
        return False
    def integrate(self, name, *cell):
        """
        Integrates a field from this log and returns it::

            # Run a simulation and calculate the total current carried by INa
            s = myokit.Simulation(m, p)
            d = s.run(1000)
            q = d.integrate('ina.INa')

        Arguments:
        
        ``name``
            The name of the variable to return, for example 'ik1.IK1' or
            '2.1.membrane.V'.
        ``*cell``
            An optional cell index, for easy access to multi-cellular data, for
            example ``log.integrate('membrane.V', 2, 1)``.
            
        """
        # Get data to integrate
        key = [str(x) for x in cell]
        key.append(str(name))
        key = '.'.join(key)
        data = np.array(self[key], copy=True)
        time = np.asarray(self.time())
        # Integration using the midpoint Riemann sum:
        #  At point i=0, the value is 0
        #  At each point i>0, the value increases by step[i, i-1]*mean[i, i-1]
        #data[1:] = 0.5 * (data[1:] + data[:-1]) * (time[1:] - time[:-1])
        #data[0]  = 0
        # For discontinuities (esp. with CVODE), it makes more sense to treat
        # the signal as a zero-order hold, IE use the left-point integration
        # rule:
        data[1:] = data[:-1] * (time[1:] - time[:-1])
        data[0]  = 0        
        return data.cumsum()
    def isplit(self, i):
        """
        Splits this log around the given indice and returns the left and right
        log.
        """
        log1 = DataLog()
        log2 = DataLog()
        log1._time = self._time
        log2._time = self._time
        for k, v in self.iteritems():
            log1[k] = v[:i]
            log2[k] = v[i:]
        return log1, log2
    def itrim(self, a, b, clone=False):
        """
        Removes all entries from this log except those from indices ``a`` to
        ``b`` (anolog to performing ``x = x[a:b]`` on a list).
        
        By default, the log will be modified in place. To leave the log
        unaltered but return a trimmed copy, set ``clone=True``.
        """
        if clone:
            log = DataLog()
            log._time = self._time
            for k, v in self.iteritems():
                if isinstance(v, np.ndarray):
                    log[str(k)] = np.array(v[a:b], copy=True)
                else:
                    log[str(k)] = v[a:b]
            return log
        else:
            for k, v in self.iteritems():
                self[k] = v[a:b]
            return self
    def itrim_left(self, i, clone=False):
        """
        Trims the first ``i`` entries from this log (analog to performing
        ``x = x[i:]`` on a list).

        By default, the log will be modified in place. To leave the log
        unaltered but return a trimmed copy, set ``clone=True``.
        """
        if clone:
            log = DataLog()
            log._time = self._time
            for k, v in self.iteritems():
                if isinstance(v, np.ndarray):
                    log[str(k)] = np.array(v[i:], copy=True)
                else:
                    log[str(k)] = v[i:]
            return log
        else:
            for k, v in self.iteritems():
                self[k] = v[i:]
            return self
    def itrim_right(self, i, clone=False):
        """
        Keeps only the ``i`` leftmost entries of this log (analog to performing
        ``x = x[:i]`` on a list). If ``i`` is negative, the ``abs(i)``
        rightmost entries will be removed.
        
        By default, the log will be modified in place. To leave the log
        unaltered but return a trimmed copy, set ``clone=True``.
        """
        if clone:
            log = DataLog()
            log._time = self._time
            for k, v in self.iteritems():
                if isinstance(v, np.ndarray):
                    log[str(k)] = np.array(v[:i], copy=True)
                else:
                    log[str(k)] = v[:i]
            return log
        else:
            for k, v in self.iteritems():
                self[k] = v[:i]
            return self
    def length(self):
        """
        Returns the length of the entries in this log. If the log is empty,
        zero is returned.
        """
        if len(self) == 0:
            return 0
        return len(self.itervalues().next())
    @staticmethod
    def load(filename, progress=None, msg='Loading DataLog'):
        """
        Loads a :class:`DataLog` from the binary format used by myokit.
        
        The values in the log will be stored in an :class:`array.array`. The
        data type used by the array will be the one specified in the binary
        file. Notice that an `array.array` storing single precision floats will
        make conversions to ``Float`` objects when items are accessed.
        
        To obtain feedback on the simulation progress, an object implementing
        the :class:`myokit.ProgressReporter` interface can be passed in.
        passed in as ``progress``. An optional description of the current
        simulation to use in the ProgressReporter can be passed in as `msg`.
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
        # Get size of single and double types on this machine
        dsize = {'d' : len(array.array('d', [1]).tostring()),
                 'f' : len(array.array('f', [1]).tostring())}
        # Read data
        try:
            f = None
            f = zipfile.ZipFile(filename, 'r')
            # Get ZipInfo objects
            try:
                body = f.getinfo('data.bin')
            except KeyError:
                raise myokit.DataLogReadError('Invalid log file format.')
            try:
                head = f.getinfo('structure.txt')
                old_format = False
            except KeyError:
                try:
                    head = f.getinfo('header.txt')
                    old_format = True
                except KeyError:
                    raise myokit.DataLogReadError('Invalid log file format.')
            # Read file contents
            head = f.read(head)
            body = f.read(body)
        except zipfile.BadZipfile:
            raise myokit.DataLogReadError('Unable to read log: bad zip file.')
        except zipfile.LargeZipFile:
            raise myokit.DataLogReadError('Unable to read log: zip file'
                ' requires zip64 support and this has not been enabled on this'
                ' system.')
        finally:
            if f:
                f.close()
        # Create empty log
        log = DataLog()
        # Parse header
        if old_format:
            # Parse header in old format
            head = head.splitlines()
            if len(head) == 1:
                # Empty file
                return log
            head = iter(head)
            # Skip first line
            head.next()
            # Get field information, data type and size is given redundantly
            fields = []
            for line in head:
                field, data_type, data_size = entry.split(',')
                fields.append(field[1:-1])
            data_size = int(data_size)
            data_type = data_type[1:-1]
        else:
            # Parse header in new format:
            # Number of fields, length of data arrays, data type, time, fields
            head = iter(head.splitlines())
            n = int(head.next())
            data_size = int(head.next())
            data_type = head.next()
            time = head.next()
            if time:
                # Note, this field doesn't have to be present in the log!
                log._time = time
            fields = [x for x in head]
            if len(fields) != n:
                raise DataLogReadError('Invalid number of fields specified.')
        # Get size of each entry on disk
        if data_size < 0:
            raise DataLogReadError('Invalid data size: ' + str(data_size) +'.')
        try:
            data_size *= dsize[data_type]
        except KeyError:
            raise DataLogReadError('Invalid data type: "' + data_type + '".')
        # Parse read data
        fraction = 1.0 / len(fields)
        start, end = 0, 0
        nbody = len(body)
        try:
            if progress:
                progress.enter(msg)
            for k, field in enumerate(fields):
                if progress and not progress.update(k * fraction):
                    return
                # Get new data position
                start = end
                end += data_size
                if end > nbody:
                    raise myokit.DataLogReadError('Header indicates larger'
                        ' data size than found in body.')
                # Read data
                ar = array.array(data_type)
                ar.fromstring(body[start:end])
                if sys.byteorder == 'big':
                    ar.byteswap()
                log[field] = ar
        finally:
            if progress:
                progress.exit()
        return log
    @staticmethod
    def load_csv(filename, precision=myokit.DOUBLE_PRECISION):
        """
        Loads a CSV file from disk and returns it as a :class:`DataLog`.

        The CSV file must start with a header line indicating the variable
        names, separated by commas. Each subsequent row should contain the
        values at a single point in time for all logged variables.
        
        The ``DataLog`` is created using the data type specified by the
        argument ``precision``, regardless of the data type of the stored data.
        
        The log attempts to set a time variable by searching for a strictly
        increasing variable. In the case of a tie the first strictly increasing
        variable is used. This means logs stored with :meth:`save_csv` can
        safely be read.
        """
        log = DataLog()
        # Check filename
        filename = os.path.expanduser(filename)
        # Typecode dependent on precision
        typecode = 'd' if precision == myokit.DOUBLE_PRECISION else 'f'
        # Error raising function
        def e(line, char, msg):
            raise myokit.DataLogReadError('Syntax error on line ' + str(line)
                 + ', character ' + str(1 + char) + ': ' + msg)
        quote = '"'
        delim = ','
        with open(filename, 'r') as f:
            # Read header
            keys = [] # The log keys, in order of appearance
            try:
                line = f.readline()
                # Ignore lines commented with #
                while line.lstrip()[:1] == '#':
                    line = f.readline()
            except EOFError:
                # Empty file
                return log
            # Trim end of line
            if len(line) > 1 and line[-2:] == '\r\n':
                eol = 2
                line = line[:-2]
            else:
                eol = 1
                line = line[:-1]
            # Trim ; at end of line if given
            if line[-1:] == ';':
                eol += 1
                line = line[:-1]
            # Get enumerated iterator over characters
            line = enumerate(line)
            try:
                i, c = line.next()
            except StopIteration:
                # Empty line
                return log
            # Whitespace characters to ignore
            whitespace = ' \f\t'
            # Start parsing header fields
            run1 = True
            while run1:
                text = []
                # Skip whitespace
                try:
                    while c in whitespace:
                        i, c = line.next()
                except StopIteration:
                    break
                if c == quote:
                    # Read quoted field + delimiter or eol
                    run2 = True
                    while run2:
                        try:
                            i, c = line.next()
                        except StopIteration:
                            e(1, i, 'Unexpected end-of-line inside quoted'
                                ' string.')
                        if c == quote:
                            try:
                                i, c = line.next()
                                if c == quote:
                                    text.append(quote)
                                elif c == delim or c in whitespace:
                                    run2 = False
                                else:
                                    e(1, i, 'Expecting double quote, delimiter'
                                        ' or end-of-line. Found "' + c + '".')
                            except StopIteration:
                                run1 = run2 = False
                        else:
                            text.append(c)
                else:
                    # Read unquoted field + delimiter or eol
                    while run1 and c != delim:
                        try:
                            text.append(c)
                            i, c = line.next()
                        except StopIteration:
                            run1 = False
                # Append new field to list
                keys.append(''.join(text))
                # Read next character
                try:
                    i, c = line.next()
                except StopIteration:
                    run1 = False
            if c == delim:
                e(1, i, 'Empty field in header.')
            # Create data structure
            m = len(keys)
            lists = []
            for key in keys:
                x = array.array(typecode)
                lists.append(x)
                log[key] = x
            # Read remaining data
            try:
                n = 0
                while True:
                    row = f.readline()
                    # Ignore blank lines
                    if row.strip() == '':
                        break
                    # Ignore lines commented with #
                    if row.lstrip()[:1] == '#':
                        continue
                    row = row[:-eol]
                    row = row.split(delim)
                    n += 1
                    if len(row) != m:
                        e(n, 0, 'Wrong number of columns found in row '
                            + str(n) + '. Expecting ' + str(m) + ', found '
                            + str(len(row)) +'.')
                    try:
                        for k, v in enumerate(row):
                            lists[k].append(float(v))
                    except ValueError:
                        e(n, 0, 'Unable to convert found data to floats.')
            except StopIteration:
                pass
            # Guess time variable
            for key in keys:
                x = np.array(log[key], copy=False)
                y = x[1:] - x[:-1]
                if np.all(y > 0):
                    log.set_time_key(key)
                    break
            # Return log
            return log
    def npview(self):
        """
        Returns a ``DataLog`` with numpy array views of its data.
        """
        out = DataLog()
        out._time = self._time
        for k, d in self.iteritems():
            out[k] = np.asarray(d)
        return out
    def _parse_key(self, key):
        """
        Parses a key used for __getitem__, __setitem__, __delitem__ and
        __contains__.
        """
        if type(key) == tuple:
            name = str(key[0])
            if len(key) == 2 and type(key[1]) not in [int, float]:
                parts = [str(x) for x in key[1]]
            else:
                parts = [str(x) for x in key[1:]]
            parts.append(str(name))
            key = '.'.join(parts)
        return str(key)
    def regularize(self, dt, tmin=None, tmax=None):
        """
        Returns a copy of this DataLog with data points at regularly spaced
        times.

        *Note: While regularize() can be used post-simulation to create fixed
        time-step data from variable time-step data, it is usually better to
        re-run a simulation with fixed time step logging. See*
        :meth:`Simulation.run()` *for details.*

        The first point will be at ``tmin`` if specified or otherwise the first
        time present in the log. All following points will be spaced ``dt``
        time units apart and the final point will be less than or equal to
        ``tmax``. If no value for ``tmax`` is given the final value in the log
        is used.

        *This function requires scipy to be installed.* It works by

          1. Finding the indices corresponding to ``tmin`` and ``tmax``.
          2. Creating a spline interpolant with all the data from ``tmin`` to
             ``tmax``. If possible, two points to the left and right of
             ``tmin`` and ``tmax`` will be included in the interpolated data
             set (so *only* if there are at least two values before ``tmin`` or
             two values after ``tmax`` in the data respectively).
          3. Evaluating the interpolant at the regularly spaced points.

        As a result of the (first-order) spline interpolation, the function may
        perform poorly on large data sets.
        """
        self.validate()
        from scipy.interpolate import UnivariateSpline as Spline
        # Check time variable
        time = self.time()
        n = len(time)
        # Get left indice for splines
        imin = 0
        if tmin is None:
            tmin = time[0]
        elif tmin > time[0]:
            # Find position of tmin in time list, then add two points to the
            # left so that the spline has 4 points
            imin = np.searchsorted(time, tmin) - 2
            if imin < 0: imin = 0
        # Get right indice for splines
        imax = n
        if tmax is None:
            tmax = time[-1]
        elif tmax < time[-1]:
            imax = np.searchsorted(time, tmax) + 2
            if imax > n: imax = n
        # Get time steps
        steps = 1 + np.floor((tmax - tmin) / dt)
        rtime = tmin + dt * np.arange(0, steps)
        # Create output and return
        out = DataLog()
        out._time = self._time
        out[self._time] = rtime
        time_part = time[imin:imax]
        for key, data in self.iteritems():
            if key != self._time:
                s = Spline(time_part, data[imin:imax], k=1, s=0)
                out[key] = s(rtime)
        return out
    def save(self, filename, precision=myokit.DOUBLE_PRECISION):
        """
        Writes this ``DataLog`` to a binary file.
        
        The resulting file will be a zip file with the following entries:
        
        ``header.txt``
            A csv file with the fields ``name, dtype, len`` for each variable.
        ``data.bin``
            The binary data in the order specified by the header.
        ``readme.txt``
            A text file explaining the file format.
            
        The optional argument ``precision`` allows logs to be stored in single
        precision format, which saves space.
        """
        self.validate()
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
        dtype = 'd' if precision == myokit.DOUBLE_PRECISION else 'f'
        # Create data strings
        head_str = []
        body_str = []
        # Number of fields, length of data arrays, data type, time, fields
        head_str.append(str(len(self)))
        head_str.append(str(len(self.itervalues().next())))
        head_str.append(dtype)
        # Note: the time field might not be present in the log!
        head_str.append(self._time if self._time else '')
        # Write field names and data
        for k, v in self.iteritems():
            head_str.append(k)
            # Create array, ensure it's litte-endian
            ar = array.array(dtype, v)
            if sys.byteorder == 'big':
                ar.byteswap()
            body_str.append(ar.tostring())
        head_str = '\n'.join(head_str)
        body_str = ''.join(body_str)
        # Write
        head = zipfile.ZipInfo('structure.txt')
        head.compress_type = zipfile.ZIP_DEFLATED
        body = zipfile.ZipInfo('data.bin')
        body.compress_type = zipfile.ZIP_DEFLATED
        read = zipfile.ZipInfo('readme.txt')
        read.compress_type = zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(filename, 'w') as f:
            f.writestr(body, body_str)
            f.writestr(head, head_str)            
            f.writestr(read, README_SAVE_BIN)
    def save_csv(self, filename, precision=myokit.DOUBLE_PRECISION, order=None,
            delimiter=',', header=True):
        """
        Writes this ``DataLog`` to a CSV file, following the syntax
        outlined in RFC 4180 and with a header indicating the field names.

        The resulting file will consist of:

          - A header line containing the names of all logged variables,
            separated by commas.
          - Each following line will be a comma separated list of values in the
            same order as the header line. A line is added for each time point
            logged.
            
        Arguments:
        
        ``filename``
            The file to write (existing files will be overwritten without
            warning.
        ``precision``
            If a precision argument (for example ``myokit.DOUBLE_PRECISION``)
            is given, the output will be stored in such a way that this amount
            of precision is guaranteed to be present in the string. If the
            precision argument is set to ``None`` python's default formatting
            is used, which may lead to smaller files.
        ``order``            
            To specify the ordering of the log's arguments, pass in a sequence
            ``order`` with the log's keys.
        ``delimiter``
            This field can be used to set an alternative delimiter. To use
            spaces set ``delimiter=' '``, for tabs: ``delimiter='\\t'``. Note
            that some delimiters (for example '\\n' or '1234') will produce an
            unreadable or invalid csv file.
        ``header``
            Set this to ``False`` to avoid adding a header to the file. Note
            that Myokit will no longer be able to read the written csv file
            without this header.
        
        *A note about locale settings*: On Windows systems with a locale
        setting that uses the comma as a decimal separator, importing CSV files
        into Microsoft Excel can be troublesome. To correctly import a CSV,
        either (1) Change your locale settings to use "." as a decimal
        separator or (2) Use the import wizard under Data > Get External Data
        to manually specify the correct separator and delimiter.
        """
        self.validate()
        # Check filename
        filename = os.path.expanduser(filename)
        # Set precision
        if precision is None:
            fmat = lambda x: str(x)
        elif precision == myokit.DOUBLE_PRECISION:
            fmat = lambda x: myokit.SFDOUBLE.format(x)
        elif precision == myokit.SINGLE_PRECISION:
            fmat = lambda x: myokit.SFSINGLE.format(x)
        else:
            raise ValueError('Precision level not supported.')
        # Write file
        # EOL: CSV files have DOS line endings by convention. On windows, 
        # writing '\n' to a file opened in mode 'w' will actually write '\r\n'
        # which means writing '\r\n' writes '\r\r\n'. To prevent this, open the
        # file in mode 'wb'.
        eol = '\r\n'
        quote = '"'
        escape = '""'
        with open(filename, 'wb') as f:
            # Convert dict structure to ordered sequences
            if order:
                if set(order) != set(self.iterkeys()):
                    raise ValueError('The given `order` sequence must contain'
                        ' all the same keys present in the log.')
                keys = order
                data = [self[x] for x in keys]
            else:
                keys = []
                data = []
                if self._time and self._time in self.keys():
                    # Save time as first variable
                    dat = self[self._time]
                    keys.append(self._time)
                    data.append(dat)
                    for key, dat in sorted(self.iteritems()):
                        if key != self._time:
                            keys.append(key)
                            data.append(dat)    
                else:
                    for key, dat in sorted(self.iteritems()):
                        keys.append(key)
                        data.append(dat)
            # Number of entries
            m = len(keys)
            if m == 0:
                return
            # Get length of entries
            n = self.length()
            # Write header
            if header:
                line = []
                for key in keys:
                    # Escape quotes within strings
                    line.append(quote + key.replace(quote, escape) + quote)
                f.write(delimiter.join(line) + eol)
            # Write data
            data = [iter(x) for x in data]
            for i in xrange(0, n):
                line = []
                for d in data:
                    line.append(fmat(d.next()))
                f.write(delimiter.join(line) + eol)
    def set_time_key(self, key):
        """
        Sets the key under which the time data is stored.
        """
        self._time = None if key is None else str(key)
    def __setitem__(self, key, value):
        return super(DataLog, self).__setitem__(self._parse_key(key),
            value)
    def split(self, value):
        """
        Splits the log into a part before and after the time ``value``::
        
            s = myokit.Simulation(m, p)
            d = s.run(1000)
            d1, d2 = d.split(100)

        In this example, d1 will contain all values up to, but not including,
        t=100. While d2 will contain the values from t=100 and upwards.
        """
        return self.isplit(self.find(value))
    def split_periodic(self, period, adjust=False, closed_intervals=True):
        """
        Splits this log into multiple logs, each covering an equal period of
        time. For example a log covering the time span ``[0, 10000]`` can be
        split with period ``1000`` to obtain ten logs covering ``[0, 1000]``,
        ``[1000, 2000]`` etc.

        The split log files can be returned as-is, or with the time variable's
        value adjusted so that all logs appear to cover the same span. To
        enable this option, set ``adjust`` to ``True``.

        By default, the returned intervals are *closed*, so both the left and
        right endpoint are included (if present in the data). This may involve
        the duplication of some data points. To disable this behaviour and
        return half-closed endpoints (containing only the left point), set
        ``closed_intervals`` to ``False``.
        """
        self.validate()
        # Check time variable
        time = self.time()
        if len(time) < 1:
            raise Exception('DataLog entries have zero length.')
        # Check period
        period = float(period)
        if period <= 0:
            raise ValueError('Period must be greater than zero')
        # Get start, end, etc
        tmin = time[0]
        tmax = time[len(time)-1]
        nlogs = int(np.ceil((tmax - tmin) / period))
        nvars = len(self)
        if nlogs < 2:
            return self
        # Find split points
        tstarts = tmin + np.arange(nlogs) * period
        istarts = [0] * nlogs
        k = 0
        for i, t in enumerate(time):
            while k < nlogs and t >= tstarts[k]:
                istarts[k] = i
                k += 1
        # Create logs
        logs = []
        for i in xrange(0, nlogs - 1):
            log = DataLog()
            log._time = self._time
            # Get indices
            imin = istarts[i]
            imax = istarts[i + 1]
            # Include right point endpoint if needed
            if closed_intervals and time[imax] == tstarts[i + 1]:
                imax += 1
            # Select sections of log and append
            for k, v in self.iteritems():
                d = self[k][imin:imax]
                # Numpy? Then copy data
                if isinstance(d, np.ndarray):
                    d = np.array(d, copy=True, dtype=float)
                log[k] = d
            logs.append(log)
        # Last log
        log = DataLog()
        log._time = self._time
        imin = istarts[-1]
        imax = len(time)
        # Not including right endpoints? Then may be required to omit last pt
        if not closed_intervals and time[-1] >= tmin + nlogs * period:
            imax -= 1
        # Select sections of log and append
        for k, v in self.iteritems():
            d = self[k][imin:imax]
            # Numpy? Then copy data
            if isinstance(d, np.ndarray):
                d = np.array(d, copy=True, dtype=float)
            log[k] = d
        logs.append(log)
        # Adjust
        if adjust:
            if isinstance(time, np.ndarray):
                # Fast method for numpy arrays
                for k, log in enumerate(logs):
                    log[self._time] -= k * period
            else:
                for k, log in enumerate(logs):
                    tlist = log[self._time]
                    tdiff = k * period
                    for i in xrange(len(tlist)):
                        tlist[i] -= tdiff
        return logs
    def time(self):
        """
        Returns this log's time array.
        
        Raises a :class:`myokit.InvalidDataLogError` if the time variable for
        this log has not been specified or an invalid key was given for the
        time variable.
        """
        try:
            return self[self._time]
        except KeyError:
            if self._time is None:
                raise myokit.InvalidDataLogError('No time variable set.')
            else:
                raise myokit.InvalidDataLogError('Invalid key <'
                    + str(self._time) + '> set for time variable.')
    def time_key(self):
        """
        Returns the name of the time variable stored in this log, or ``None``
        if no time variable was set.
        """
        return self._time
    def trim(self, a, b, adjust=False, clone=False):
        """
        Removes log entries before time ``a`` and after time ``b``.
        
        If ``adjust`` is set to ``True``, all logged times will be lowered by
        ``a``.
        
        By default, the log will be modified in place. To leave the log
        unaltered but return a trimmed copy, set ``clone=True``.
        """
        self.validate()
        log = self.itrim(self.find(a), self.find(b), clone=clone)
        if adjust:
            if isinstance(log[self._time], np.ndarray):
                log[self._time] -= a
            else:
                log[self._time] = [x - a for x in log[self._time]]
        return log
    def trim_left(self, value, adjust=False, clone=False):
        """
        Removes all data logged before time ``value``.

        If ``adjust`` is set to ``True``, all logged times will be lowered by
        ``a``.
        
        By default, the log will be modified in place. To leave the log
        unaltered but return a trimmed copy, set ``clone=True``.
        """
        self.validate()
        log = self.itrim_left(self.find(value), clone=clone)
        if adjust:
            if isinstance(log[self._time], np.ndarray):
                log[self._time] -= value
            else:
                log[self._time] = [x - value for x in log[self._time]]
        return log
    def trim_right(self, value, clone=False):
        """
        Removes all data logged after time ``value``.
        
        By default, the log will be modified in place. To leave the log
        unaltered but return a trimmed copy, set ``clone=True``.
        """
        return self.itrim_right(self.find(value), clone=clone)
    def validate(self):
        """
        Validates this ``DataLog``. Raises a
        :class:`myokit.InvalidDataLogError` if the log has inconsistencies.
        """
        if self._time:
            if not self._time in self:
                raise myokit.InvalidDataLogError('Time variable <'
                    + str(self._time) + '> specified but not found in log.')
            dt = np.asarray(self[self._time])
            if np.any(dt[1:] - dt[:-1] < 0):
                raise myokit.InvalidDataLogError('Time must be'
                    ' non-decreasing.')
        if len(self) > 0:
            n = set([len(v) for v in self.itervalues()])
            if len(n) > 1:
                raise myokit.InvalidDataLogError('All entries in a data log'
                    ' must have the same length.')
    def variable_info(self):
        """
        Returns a dictionary mapping fully qualified variable names to
        :class:`LoggedvariableInfo` instances, providing information about the
        logged data.
        
        Comes with the following constraints:
        
        - Per variable, the data must have a consistent dimensionality. For
          example having a key ``0.membrane.V`` and a key ``1.1.membrane.V``
          would violate this constraint.
        - Per variable, the data must be regular accross dimensions. For
          example if there are ``n`` entries ``0.x.membrane.V``, and there are
          also entries of the form ``1.x.membrane.V`` then the values of ``x``
          must be the same for both cases.
          
        An example of a dataset that violates the second constraint is::
        
            0.0.membrane.V
            0.1.membrane.V
            0.2.membrane.V
            1.0.membrane.V
            1.1.membrane.V
            
        If either of the constraints is violated a ``ValueError`` is raised.
        """
        # The algorithm for condition 2 works by creating a set of the unique
        # entries in each column. The product of the sizes of these sets should
        # equal the total number of entries for a variable.
        # For example:
        #   0 1     
        #   0 2     Results in id_sets [(0,1), (1,2,3,4)]
        #   0 3     2 * 4 = 8 != len(id_list)
        #   1 1     So this data must be irregular.
        #   1 2
        #   1 4
        #
        id_lists = {}
        id_sets = {}
        for key in self:
            # Split key into id / name parts
            idx, name = split_key(key)
            # Create tuple version of id
            idx = idx.split('.')
            idx = idx[:-1]
            idx = tuple([int(i) for i in idx])
            # Find or create entry in dict of id lists
            try:
                id_list = id_lists[name]
                id_set = id_sets[name]
            except KeyError:
                # Create entry in id lists dict
                id_lists[name] = id_list = []
                # Create entry in id sets dict (one set for every dimension)
                id_sets[name] = id_set = [set() for x in idx]
            # Check if the dimensions are the same each time a name occurs.
            if id_list and len(id_list[0]) != len(idx):
                key1 = '.'.join([str(x) for x in id_list[0]]) + '.' + name
                key2 = '.'.join([str(x) for x in idx]) + '.' + name
                raise ValueError('Different dimensions used for the same'
                    ' variable. Found: <' + key1 + '> and <' + key2 + '>.')
            # Update the id list
            id_list.append(idx)
            # Update the id set
            for k, i in enumerate(idx):
                id_set[k].add(i)
        # Create variable info objects
        infos = {}
        for name, id_list in id_lists.iteritems():
            id_set = id_sets[name]
            # Check if the data is regular.
            n = len(id_list)
            m = 1
            for x in id_set:
                m *= len(x)
            if n != m:
                raise ValueError('Irregular data used for variable <'
                    + str(name) + '>')
            # Create variable info object
            infos[name] = info = LoggedVariableInfo()
            info._name = name
            info._dimension = len(id_set)
            info._size = tuple([len(x) for x in id_set])
            # Add sorted ids
            if id_list[0]:
                id_list.sort()
                #id_list.sort(key=operator.itemgetter(1,0))
            info._ids = id_list
            # Add sorted keys
            s = '.' + name if id_list[0] else name
            info._keys = ['.'.join([str(x) for x in y]) + s for y in id_list]
        return infos
class LoggedVariableInfo(object):
    """
    Contains information about the log entries for each variable. These objects
    should only be created by :meth:`DataLog.variable_info()`.
    """
    def __init__(self):
        self._dimension = None
        self._ids = None
        self._keys = None
        self._size = None
        self._name = None
    def dimension(self):
        """
        Returns the dimensions of the logged data for this variable, as an
        integer.
        """
        return self._dimension
    def ids(self):
        """
        Returns an iterator over all available ids for this variable, such
        that the second index (y in the simulation) changes fastest. For
        example, for log entries::
        
            0.0.membrane.V
            0.1.membrane.V
            0.2.membrane.V
            1.0.membrane.V
            1.1.membrane.V
            1.2.membrane.V
            
        the returned result would iterate over::
        
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
            
        The keys are returned in the same order as the ids.            
        """
        return iter(self._ids)
    def is_regular_grid(self):
        """
        Returns True if the following conditions are met:
            
        - The data 2 dimensional
        - The data is continuous: along each dimension the first data point is
          indexed as ``0`` and the last as ``Ni-1``, where ``Ni`` is the size
          in that dimension.
          
        """
        nx, ny = self._size
        return (self._dimension == 2
            and self._ids[0][0] == 0
            and self._ids[0][1] == 0
            and self._ids[-1][0] == nx - 1
            and self._ids[-1][1] == ny - 1)
    def keys(self):
        """
        Returns an iterator over all available keys for this variable, such
        that the second index (y in the simulation) changes fastest. For
        example, for log entries::
        
            0.0.membrane.V
            1.0.membrane.V
            0.1.membrane.V
            1.1.membrane.V
            0.2.membrane.V
            1.2.membrane.V
            
        the returned iterator would produce ``"0.0.membrane.V"``, then
        ``"0.1.membrane.V"`` etc.
        
        The ids are returned in the same order as the keys.
        """
        return iter(self._keys)
    def size(self):
        """
        Returns a tuple containing the size i.e. the number of entries for
        the corresponding variable in each dimension.
        
        For example, with the following log entries for `membrane.V`::
        
            0.membrane.V
            1.membrane.V
            2.membrane.V
            
        the corresponding size would be ``(3)``.
        
        A size of ``3`` doesn't guarantee the final entry is for cell number
        ``2``. For example::
        
            0.membrane.V
            10.membrane.V
            20.membrane.V
            
        would also return size ``(3)``
        
        In higher dimensions::
        
            0.0.membrane.V
            0.1.membrane.V
            0.2.membrane.V
            1.0.membrane.V
            1.1.membrane.V
            1.2.membrane.V
            
        This would return ``(2,3)``.
        
        Similarly, in a single cell scenario or for global variables, for
        exmaple::
        
            engine.time
            
        Would have size ``()``.
        """
        return self._size
    def name(self):
        """
        Returns the variable name.
        """
        return self._name
    def to_long_string(self):
        out = [self._name]
        out.append('  dimension: ' + str(self._dimension))
        out.append('  size: ' + ', '.join([str(x) for x in self._size]))
        out.append('  keys:')
        out.extend(['    ' + x for x in self._keys])
        return '\n'.join(out)
def prepare_log(log, model, dims=None, global_vars=None,
        if_empty=myokit.LOG_NONE, allowed_classes=myokit.LOG_ALL,
        precision=myokit.DOUBLE_PRECISION):
    """
    Returns a :class:`DataLog` for simulation classes based on a ``log``
    argument passed in by the user. The model the simulations will be based on
    should be passed in as ``model``.
    as
    
    The ``log`` argument can take on one of four forms:

    An existing simulation log
        In this case, the log is tested for compatibility with the given model
        and simulation dimensions. For single-cell simulations all keys in the
        log must correspond to the qname of a loggable variable (IE not a
        constant). For multi-cellular simulations this means all keys in the
        log must have the form "x.component.variable" where "x" is the cell
        index (for example "1" or "0.3").
    A list (or other sequence) of variable names to log.
        In this case, the list is converted to a DataLog object. All
        arguments in the list must be either strings corresponding to the
        variables' qnames (so "membrane.V") or variable objects from the given
        model.
        In multi-cell scenarios, passing in the qname of a variable (for
        example "membrane.V") will cause every cell's instance of this variable
        to be logged. To log only specific cells' values, pass in the indexed
        name (for example "1.2.membrane.V").
    An integer flag
        One of the following integer flags:
                
        ``myokit.LOG_NONE``
            Don't log any variables.
        ``myokit.LOG_STATE``
            Log all state variables.
        ``myokit.LOG_BOUND``
            Log all variables bound to an external value. The method will
            assume any bound variables still present in the model will be
            provided by the simulation engine.
        ``myokit.LOG_INTER``
            Log all intermediary variables.
        ``myokit.LOG_DERIV``
            Log the derivatives of the state variables.
        ``myokit.LOG_ALL``
            Combines all the previous flags.
                
        Flags can be chained together, for example
        ``log=myokit.LOG_STATE+myokit.LOG_BOUND`` will log all bound variables
        and all states.
    ``None``
        In this case the value from ``if_empty`` will be copied into log before
        the function proceeds to build a log.

    For multi-dimensional logs the simulation dimensions can be passed in as a
    tuple of dimension sizes, for example (10,) for a cable of 10 cells and
    (30,20) for a 30 by 20 piece of tissue.
    
    Simulations can define variables to be either `per-cell` or `global`. Time,
    for example, is typically a global variable while membrane potential will
    be stored per cell. To indicate which is which, a list of global variables
    can be passed in as ``global_vars``.
    
    The argument ``if_empty`` is used to set a default argument if ``log`` is
    is given as ``None``.
    
    The argument ``allowed_classes`` is an integer flag that determines which
    type of variables are allowed in this log.
    
    When a new DataLog is created by this method, the internal storage
    uses arrays from the array module. The data type for these new arrays can
    be specified using the ``precision`` argument.
    """
    # Typecode dependent on precision
    typecode = 'd' if precision == myokit.DOUBLE_PRECISION else 'f'
    # Get all options for dimensionality
    if dims is None:
        dims = ()
    ndims = len(dims)
    if ndims == 0:
        dcombos = ['']
    else:
        dcombos = ['.'.join([str(y) for y in x]) + '.' for x in dimco(*dims)]
    # Check given list of global variables
    if global_vars is None:
        global_vars = []
    else:
        for var in global_vars:
            try:
                v = model.get(var)
            except KeyError as e:
                raise Exception('Unknown variable specified in global_vars <'
                    + str(var) + '>.')
            if v.is_state():
                raise Exception('State cannot be global variable.')
    # Function to check if variable is allowed (doesn't handle derivatives)
    def check_if_allowed_class(var):
        if var.is_constant():
            raise Exception('This log does not support constants.')
        elif var.is_state():
            if not myokit.LOG_STATE & allowed_classes:
                raise Exception('This log does not support state variables.')
        elif var.is_bound():
            if not myokit.LOG_BOUND & allowed_classes:
                raise Exception('This log does not support bound variables.')
        elif not myokit.LOG_INTER & allowed_classes:
            raise Exception('This log does not support intermediary'
                ' variables.')
    #
    # First option, no log argument given, use the "if_empty" option
    #
    if log is None:
        # Check if if_empty matches allowed_classes
        # (AKA test if ``if_empty`` is contained in ``allowed_classes``)
        if if_empty & allowed_classes == if_empty:
            log = if_empty
        else:
            # This one's only for programmers :-)
            raise Exception('if_empty option not contained in allowed_classes')
    #
    # Second option, log given as integer flag: create a simulation log and
    # return it.
    #
    if type(log) == int:
        # Log argument given as flag
        flag = log
        log = myokit.DataLog()
        if flag == myokit.LOG_ALL:
            flag = allowed_classes
        if myokit.LOG_STATE & flag:
            # Check if allowed
            if not (myokit.LOG_STATE & allowed_classes):
                raise Exception('DataLog does not support state variables.')
            # Add states
            for s in model.states():
                name = s.qname()
                for c in dcombos:
                    log[c + name] = array.array(typecode)
            flag -= myokit.LOG_STATE
        if myokit.LOG_BOUND & flag:
            # Check if allowed
            if not (myokit.LOG_BOUND & allowed_classes):
                raise Exception('DataLog does not support bound variables.')
            # Add bound variables
            for label, var in model.bindings():
                name = var.qname()
                if name in global_vars:
                    log[name] = array.array(typecode)
                else:
                    for c in dcombos:
                        log[c + name] = array.array(typecode)
            flag -= myokit.LOG_BOUND
        if myokit.LOG_INTER & flag:
            # Check if allowed
            if not (myokit.LOG_INTER & allowed_classes):
                raise Exception('DataLog does not support intermediary'
                    ' variables.')
            # Add intermediary variables
            for var in model.variables(inter=True, deep=True):
                name = var.qname()
                if name in global_vars:
                    log[name] = array.array(typecode)
                else:
                    for c in dcombos:
                        log[c + name] = array.array(typecode)
            flag -= myokit.LOG_INTER
        if myokit.LOG_DERIV & flag:
            # Check if allowed
            if not (myokit.LOG_DERIV & allowed_classes):
                raise Exception('DataLog does not support time-derivatives.')
            # Add state derivatives
            for var in model.states():
                name = var.qname()
                for c in dcombos:
                    log['dot(' + c + name + ')'] = array.array(typecode)
            flag -= myokit.LOG_DERIV
        if flag != 0:
            raise Exception('One or more unknown flags given as log.')
        # Set time variable
        time = model.time().qname()
        if time in log:    
            log.set_time_key(time)
        # Return
        return log
    #
    # Third option, a dict or DataLog is given. Test if it's suitable for this
    # simulation.
    #
    if isinstance(log, dict):
        # Ensure it's a DataLog
        if not isinstance(log, myokit.DataLog):
            log = myokit.DataLog(log)
        # Set time variable
        time = model.time().qname()
        if time in log:    
            log.set_time_key(time)
        # Ensure the log is valid
        log.validate()
        # Check dict keys
        keys = set(log.keys())
        if len(keys) == 0:
            return log
        for key in keys:
            # Handle derivatives
            deriv = key[0:4] == 'dot(' and key[-1:] == ')'
            if deriv:
                key = key[4:-1]
            # Split of index / name
            kdims, kname = split_key(key)
            # Test name-key
            try:
                var = model.get(kname)
            except KeyError:
                raise Exception('Unknown variable <'+ str(kname) + '> in log.')
            # Check if in class of allowed variables
            if deriv:
                if not myokit.LOG_DERIV & allowed_classes:
                    raise Exception('DataLog does not support derivatives.')
                if not var.is_state():
                    raise Exception('Cannot log time derivative of non-state <'
                        + var.qname() + '>.')
            else:
                check_if_allowed_class(var)
            # Check dimensions
            if kdims:
                # Raise error if global
                if kname in global_vars:
                    raise Exception('Cannot specify a cell index for global'
                        ' logging variable <' + str(kname) + '>.')
                # Test dim key
                if not kdims in dcombos:
                    raise Exception('Invalid index <'+ str(kdims) +'> in log.')
            elif dims:
                # Raise error if non-global variable is used in multi-cell log
                if kname not in global_vars:
                    raise Exception('DataLog contains non-indexed entry for'
                        ' cell-specific variable <' + str(kname) + '>.')
        # Check dict values can be appended to
        m = 'append'
        for v in log.itervalues():
            if not (hasattr(v, m) and callable(getattr(v, m))):
                raise Exception('Logging dict must map qnames to objects'
                    ' that support the append() method.')
        # Return
        return log
    #
    # Check if list interface works
    # If not, then raise exception
    #
    try:
        if len(log) > 0:
            x = log[0]
    except Exception:
        raise Exception('Argument `log` has unexpected type. Expecting None,'
            ' integer flag, sequence of names, dict or DataLog.')
    if isinstance(log, str) or isinstance(log, unicode):
        raise Exception('String passed in as `log` argument, should be list'
            ' or other sequence containing strings.')
    #
    # Fourth option, a sequence of variable names, either global or local.
    #
    lst = log
    log = myokit.DataLog()
    checked_knames = set()
    for key in lst:
        # Allow variable objects and LhsExpressions
        if isinstance(key, myokit.Variable):
            key = key.qname()
        elif isinstance(key, myokit.LhsExpression):
            key = str(key)
        # Handle derivatives
        deriv = key[0:4] == 'dot(' and key[-1:] == ')'
        if deriv:
            key = key[4:-1]            
        # Split off cell indexes
        kdims, kname = split_key(key)
        # Don't re-test multi-dim vars
        if kname not in checked_knames:
            # Test if name key points to valid variable
            try:
                var = model.get(kname)
            except KeyError:
                raise Exception('Unknown variable <' + str(kname) +'> in log.')
            # Check if in class of allowed variables
            if deriv:
                if not myokit.LOG_DERIV & allowed_classes:
                    raise Exception('DataLog does not support derivatives.')
                if not var.is_state():
                    raise Exception('Cannot log time derivative of non-state <'
                        + var.qname() + '>.')
            else:
                check_if_allowed_class(var)
            checked_knames.add(kname)
        # Add key to log
        if kdims:
            # Raise error if global
            if kname in global_vars:
                raise Exception('Cannot specify a cell index for global'
                    ' logging variable <' + str(kname) + '>.')
            # Test dim key
            if not kdims in dcombos:
                raise Exception('Invalid index <' + str(kdims) + '> in log.')
            key = kdims + kname if not deriv else 'dot(' + kdims + kname + ')'
            log[key] = array.array(typecode)
        else:
            if kname in global_vars:
                key = kname if not deriv else 'dot(' + kname + ')'
                log[key] = array.array(typecode)
            else:
                for c in dcombos:
                    key = c + kname if not deriv else 'dot(' + c + kname + ')'
                    log[key] = array.array(typecode)
    # Set time variable
    time = model.time().qname()
    if time in log:    
        log.set_time_key(time)
    # Return
    return log
def dimco(*dims):
    """
    Generates all the combinations of a certain set of integer dimensions. For
    example given ``dims=(2, 3)`` it returns::

        (0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)

    """
    n = len(dims) - 1
    """
    def inner(dims, index, prefix):
        if index == n:
            for i in xrange(0,dims[index]):
                yield prefix + (i,)
        else:
            for i in xrange(0,dims[index]):
                prefix2 = prefix + (i, )
                for y in inner(dims, index + 1, prefix2):
                    yield y
    return inner(dims, 0, ())
    """
    def inner(dims, index, postfix):
        if index == 0:
            for i in xrange(0,dims[index]):
                yield (i,) + postfix
        else:
            for i in xrange(0,dims[index]):
                postfix2 = (i, ) + postfix
                for y in inner(dims, index - 1, postfix2):
                    yield y
    return inner(dims, n, ())
def split_key(key):
    """
    Splits a log entry name into a cell index part and a variable name part.
    
    The cell index will be an empty string for 0d entries or global variables.
    For higher dimensional cases it will be the cell index in each dimension,
    followed by a period, for example: ``15.2.``.
    
    The two parts returned by split_key may always be concatenated to obtain
    the original entry.
    """
    m = ID_NAME_PATTERN.match(key, 0)
    if m:
        return key[:m.end()], key[m.end():]
    else:
        return '', key
