#
# This module reads files in Axon Binary File format v1 or v2 used by Axon
# Technologies and Molecular Devices.
# The v1 format was used until Clampex version 9. Clampex 10 and onwards use
# the v2 format.
#
# WARNING: This file hasn't been extensively tested.
#
# About ABF
# ---------
# pClamp version 10 introduced a new .abf file format ABF2, with format version
#  numbers 2.0 and up. Older version are referred to a version 1 (even though
#  the actual version number may be, for example, 1.6).
# The version 2 format uses a variable sized header making it a little trickier
#  to read but appears to be more structured overal than the first version.
#
# Trials, runs, sweeps and channels
# ---------------------------------
# In pClamp version 10 and higher, each recorded data segment is termed a
#  'sweep'. In older versions the same concept is called an 'episode'.
# The data in a sweep contains the recordings from one or more channels. The
#  number of channels used throughout a single file is constant. In other
#  words, channel 1 in sweep 1 is a recording from the same channel as channel
#  1 in sweep 10.
# A set of sweeps is called a run. Both abf versions 1 and 2 contain a variable
#  indicating the number of sweeps per run. In abf1 this is found in the
#  header, in abf2 it is stored in a separate 'protocol' section.
# It is possible to record multiple runs, each containing an equal number of
#  sweeps, each containing an equal number of channels. However, in this case
#  the data from run to run is averaged and only a single averaged run is
#  saved. This means there is never more than 1 run in a file, even though this
#  data may have been obtained during multiple runs.
# A set of runs is termed a 'trial'. Each file contains a single trial.
#
# Acquisition modes
# -----------------
# pClamp uses five acquisition modes:
#  Gap-free mode
#    Data is recored continuously, without any interruptions
#  Variable-length events mode
#    Data is recorded in bursts of variable length (for example whenever some
#    condition is met)
#  Fixed-length events mode
#    Data is recored in bursts of fixed length, starting whenever some
#    condition is met. Multiple bursts (sweeps, or episodes in pClamp <10
#    terminology) may overlap.
#  High-speed oscilloscope mode
#    Like fixed-length events mode, but sweeps will never overlap
#  Episodic stimulation mode
#    Some stimulation is applied during which the resulting reaction is
#    recorded. The resulting dataset consists of non-overlapping sweeps.
#
# Stimulus waveforms
# ------------------
# A stimulus signal in pClamp is termed a 'waveform'. Each waveform is divided
#  into a series of steps, ramps or pulse trains. Such a subsection is called
#  an 'epoch'. The protocol section of a file defines one or more stimuli, each
#  containing a list of epochs.
#
# Conversion to myokit formats
# ----------------------------
# There is no problem-free mapping of ABF data onto myokit structures, such as
# the simulation log. A fundamental difference is that "sampling" during a
# simulation happens at the same time for every signal. Channels in an ABF file
# each have their own sampling rate.
#
#---------------------------------  license  ----------------------------------
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
#---------------------------------  credits  ----------------------------------
#
# This module is derived in part from code found in the Neo package for
# representing electrophysiology data, specifically from a python module 
# authored by sgarcia and jnowacki.
# Neo can be found at: http://neuralensemble.org/trac/neo
#
# The Neo package is licensed using the following BSD License:
#
#----------------------------------  start  -----------------------------------
# Copyright (c) 2010-2012, Neo authors and contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# Neither the names of the copyright holders nor the names of the contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#-----------------------------------  end  ------------------------------------
# The code used in Neo is itself derived from the publicly contributed matlab
#  script abf2load, again licensed under BSD. The original notice follows
#  below:
#----------------------------------  start  -----------------------------------
# Copyright (c) 2009, Forrest Collman
# Copyright (c) 2004, Harald Hentschke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#-----------------------------------  end  ------------------------------------
# The abf2load script is available from:
#  http://www.mathworks.com/matlabcentral/fileexchange/22114-abf2load
#------------------------------------------------------------------------------
# Information - but no direct code - from the matlab script get_abf_header.m
# was also used: http://neurodata.hg.sourceforge.net/hgweb/neurodata/neurodata/
#------------------------------------------------------------------------------
from collections import OrderedDict
import numpy as np
import traceback
import datetime
import struct
import os
import myokit
class AbfFile(object):
    """
    Represents a read-only Axon Binary Format file (``.abf``), stored at the
    location pointed to by ``filepath``.

    Files in the "ABF" format and the newer "ABF2" format can be read. If the
    given ``filepath`` ends in ``.pro`` a protocol file is assumed. This
    assumption can be overruled by setting the ``is_protocol_file`` argument
    to either ``True`` or ``False``.
    
    The "data" in an AbfFile is recorded (analog-to-digital) data. Any output
    signals from the amplifier to the cell (digital-to-analog) are termed the
    "protocol".
    
    Data in AbfFiles is recorded in episodes called "sweeps". Each sweep
    contains the data from all recorded channels. The number of channels is
    constant: channel 1 in sweep 1 contains data recorded from the same source
    as channel 1 in sweep 10.
    
    The data in an ``AbfFile`` can be read by iterating over it::
    
        f = AbfFile('some_file.abf')
        for sweep in f:
            for channel in sweep:
                pl.plot(channel.times(), channel.values())
    
    Similarly, protocol data can be accessed using::
    
        for sweep in f.protocol():
            for channel in sweep:
                pl.plot(channel.times(), channel.values())
                
    Note that the number of output ("protocol") channels need not equal the
    number of input ("data") channels.
    
    Because each channel can have a different sampling rate, AbfFile data is
    not one-on-one compatible with myokit Simulation logs. To obtain a
    :class:`myokit.DataLog` version of the file's data, use :meth:`myokit_log`.
    
    In some cases, a myokit protocol can be created from a stored stimulus
    protocol. To do this, use the method :meth:`myokit_protocol`.
    """
    def __init__(self, filepath, is_protocol_file=None):
        # The path to the file and its basename
        self._filepath = os.path.abspath(filepath)
        self._filename = os.path.basename(filepath)
        # Abf format version
        self._version = None
        # Protocol info
        self._epoch_functions   = None
        self._numberOfTrials    = None
        self._trialStartToStart = None
        self._runsPerTrial      = None
        self._runStartToStart   = None
        self._sweepsPerRun      = None
        self._sweepStartToStart = None
        # Read as protocol file yes?
        if is_protocol_file is None:
            self._is_protocol_file = os.path.splitext(filepath)[1] == '.pro'
        else:
            self._is_protocol_file = is_protocol_file == True
        # The file header (an ordered dictionary)
        self._header = self._read_header()
        # Date/time of recording
        self._datetime = self._read_datetime()
        # Number of channels, sampling rate (Hz) and acquisition mode
        if self._version < 2:
            self._nc = self._header['nADCNumChannels']
            self._rate = 1e6 / (self._header['fADCSampleInterval'] * self._nc)
            self._mode = self._header['nOperationMode']
        else:
            self._nc = self._header['sections']['ADC']['length']
            self._rate = 1e6 / self._header['protocol']['fADCSequenceInterval']
            self._mode = self._header['protocol']['nOperationMode']
        if not self._mode in acquisition_modes:
            raise NotImplementedError('Unknown mode: ' + str(mode))
        # Conversion factors for integer data in the channels
        self._adc_factors = None
        self._adc_offsets = None
        self._set_conversion_factors()
        # The protocol used (a list of sweeps)
        try:
            self._protocol = self._read_protocol()
        except Exception:
            print('Warning: Unable to read protocol')
            print(traceback.format_exc())
            self._protocol = []
        # The measured data as a list of sweeps
        if not self._is_protocol_file:
            self._sweeps = self._read_sweeps()
    def data_channels(self):
        """
        Returns the number of channels in this file's sweeps.
        """
        return len(self._sweeps[0])
    def extract_channel(self, channel=0):
        """
        Extracts the given data channel and returns it as a list containg:
        
             A time vector
             The first sweep
             The second sweep
             ...
            
        Each vector is returned as a numpy array.
        """
        data = []
        if len(self._sweeps) == 0:
            return data
        data.append(np.array(self._sweeps[0][channel].times()))
        for sweep in self._sweeps:
            data.append(np.array(sweep[channel].values()))
        return data
    def extract_channel_as_myokit_log(self, channel=0):
        """
        Extracts the given data channel and returns it as a myokit
        DataLog.
        
        The log will contain an entry "time" that contains the time vector.
        Each sweep will be in an entry "0.sweep", "1.sweep", "2.sweep" etc.
        """
        import myokit
        log = myokit.DataLog()
        if len(self._sweeps) == 0:
            return log
        log.set_time_key('time')
        log['time'] = np.array(self._sweeps[0][channel].times())
        for k, sweep in enumerate(self._sweeps):
            log['sweep', k] = np.array(sweep[channel].values())
        return log
    def filename(self):
        """
        Returns this AbfFile's filename.
        """
        return os.path.join(self._filepath, self._filename)
    def __getitem__(self, key):
        return self._sweeps.__getitem__(key)
    def _get_conversion_factors(self, channel):
        """
        Returns the conversion factor and shift for the selected channel as a
        tuple of floats ``(factor, shift)``.
        """    
        return self._adc_factors[channel], self._adc_offsets[channel]
    def info(self, show_header=False):
        """
        Returns a string with lots of info on this file.

        The optional argument ``show_header`` can be used to add the full
        header contents to the output.
        """
        out = []
        # Show file info
        out.append('Axon Binary File: ' + self._filename)
        out.append('ABF Format version ' + str(self._version))
        out.append('Recorded on: ' + str(self._datetime))
        # Show protocol info
        out.append('Acquisition mode: ' + str(self._mode) + ': '
            + acquisition_modes[self._mode])
        if self._numberOfTrials:
            out.append('Protocol set for ' + str(self._numberOfTrials)
                + ' trials, measuring ' + str(self._trialStartToStart)
                + 's start-to-start.')
            out.append('    with ' + str(self._runsPerTrial)
                + ' runs per trial, measuring ' + str(self._runStartToStart)
                + 's start-to-start.')
            out.append('     and ' + str(self._sweepsPerRun) 
                + ' sweeps per run, measuring ' + str(self._sweepStartToStart)
                + ' s start-to-start')
        else:
            out.append('Protocol data could not be determined.')
        out.append('Sampling rate: ' + str(self._rate) + ' Hz')
        # Show channel info
        for i, c in enumerate(self._sweeps[0]._channels):
            out.append('Channel ' + str(i) + ': "' + c._name + '"')
            if c._type:
                out.append('  Type: ' + type_mode_names[c._type])            
            out.append('  Unit: ' + c._unit)
            if c._lopass:
                out.append('  Low-pass filter: ' + str(c._lopass) + ' Hz')
            if c._cm:
                out.append('  Cm (telegraphed): ' + str(c._cm) + ' pF')
            if c._rs:
                out.append('  Rs (telegraphed): ' + str(c._rs))
        # Methods
        def show_dict(name, d, tab=''):
            m = 38 - len(tab) - int(0.1 + len(name) / 2.0)
            if m < 0: m = 0
            out.append(tab + '-'*m + '  ' + name + '  ' + '-'*m)
            for n, v in d.iteritems():
                n = str(n)
                if type(v) == OrderedDict:
                    show_dict(n, v, tab + '  ')
                elif type(v) == list:
                    show_list(n, v, tab)
                else:
                    out.append(tab + n + ' : ' + str(v))
            m = 80 - 2 * len(tab)
            if m < 0: m = 0
            out.append(tab + m*'-')
        def show_list(name, d, tab=''):
            for index, item in enumerate(d):
                n = name + '[' + str(index) + ']'
                if type(item) == OrderedDict:
                    show_dict(n, item, tab)
                elif type(item) == list:
                    show_list(n, item, tab)
                else:
                    out.append(tab + n + ' : ' + str(item))
        # Show full header info
        if show_header:
            if self.strings:
                show_dict('Strings', {'strings' : self.strings})
            show_dict('file header', self._header)
        return '\n'.join(out)
    def matplotlib_figure(self):
        """        
        Creates and returns a matplotlib figure of this abf file's contents.
        """        
        import matplotlib.pyplot as pl
        f = pl.figure()
        pl.suptitle(self.filename())
        pl.subplot(2,1,1)
        pl.title('Measured data')
        times = None
        for sweep in self:
            for channel in sweep:
                if times is None:
                    times = channel.times()
                pl.plot(times, channel.values())
        for sweep in self.protocol():
            n = len(sweep)
            times = None
            for i, channel in enumerate(sweep):
                if times is None:
                    times = channel.times()
                pl.subplot(2, n, n + 1 + i)
                pl.title(channel.name())
                pl.plot(times, channel.values())
        return f
    def myokit_log(self):
        """
        Converts the data in this ABF file to a :class:`myokit.DataLog` with an
        entry for every channel. All sweeps will be joined together into a
        single time series.
        
        The log will contain an entry "time" that contains the time vector.
        Channels will be stored using "0.ad", "1.ad" etc for the recorded
        (analog-to-digital) channels and "0.da", "1.da" et for the output
        (digital-to-analog) channels.
        """
        import myokit
        log = myokit.DataLog()
        if self._sweeps:
            # Gather parts of time and channel vectors
            time = []
            ad_channels = []
            da_channels = []
            for i in xrange(self.data_channels()):
                ad_channels.append([])
            for i in xrange(self.protocol_channels()):
                da_channels.append([])
            for sweep in self:
                for channel in sweep:
                    time.append(channel.times())
                    break
                for i, channel in enumerate(sweep):
                    ad_channels[i].append(channel.values())
            for sweep in self.protocol():
                for i, channel in enumerate(sweep):
                    da_channels[i].append(channel.values())
            # Combine into time series, store in log
            log['time'] = np.concatenate(time)
            log.set_time_key('time')
            for i, channel in enumerate(ad_channels):
                log['ad', i] = np.concatenate(channel)
            for i, channel in enumerate(da_channels):
                log['da', i] = np.concatenate(channel)
        return log
    def myokit_protocol(self, channel=None, ms=True):
        """
        Returns a single channel from an embedded protocol as a
        :class:`myokit.Protocol`. The channel to return is specified by setting
        ``channel`` to the correct index.
        
        Only works for episodic stimulation, without user lists.
        
        By default, all times are converted to milliseconds. To disable this
        function, set ``ms=False``.
        """
        import myokit
        # Only episodic stimulation is supported.
        if self._mode != ACMODE_EPISODIC_STIMULATION:
            return myokit.Protocol()
        # Check channel
        if channel is None:
            channel = 0
        else:
            channel = int(channel)
        # User lists are not supported
        if self._version < 2:
            if self._header['nULEnable'][channel]:
                raise NotImplementedError('User lists are not supported.')
        else:
            for userlist in self._header['listUserListInfo']:
                if userlist['nULEnable']:
                    raise NotImplementedError('User lists are not supported.')
        # Create protocol
        p = myokit.Protocol()
        # Get epoch functions set by _read_protocol
        dinfo, einfo_exists, einfo = self._epoch_functions
        out = []
        start = 0
        next_start = 0
        f = 1e3 if ms else 1
        for iSweep in range(self._sweepsPerRun):
            if not einfo_exists(channel):
                raise Exception('Missing protocol data')
            for e in einfo(channel):
                kind = e['type']
                if kind not in epoch_types:
                    raise NotImplementedError('Unknown epoch type: '+str(kind))
                if kind == EPOCH_DISABLED:
                    continue
                elif kind == EPOCH_STEPPED:
                    # Event at step
                    dur = f * e['init_duration'] / self._rate
                    inc = f * e['duration_inc'] / self._rate
                    e_level  = e['init_level'] + e['level_inc'] * iSweep
                    e_start  = start
                    e_length = dur + iSweep * inc
                    p.schedule(e_level, e_start, e_length)
                    start += e_length
                else:
                    raise NotImplementedError('Usupported epoch type: '
                        + epoch_types(kind))
            # Event at holding potential
            next_start += f * self._sweepStartToStart
            e_level = dinfo(channel, 'fDACHoldingLevel')
            e_start = start
            e_length = next_start - start
            p.schedule(e_level, e_start, e_length)
            start = next_start
        return p
    def protocol_channels(self):
        """
        Returns the number of channels in this file's protocol.
        """
        if self._version < 2:
            return len(self._header['sDACChannelName'])
        else:
            return int(self._header['sections']['DAC']['length'])
    def protocol_holding_level(self, channel=0):
        """
        Returns the holding level used by the requested output channel of the
        embedded protocol.
        """
        dinfo, einfo_exists, einfo = self._epoch_functions
        return dinfo(channel, 'fDACHoldingLevel')
    def protocol_steps(self, channel=0):
        """
        For a stepped protocol, this function returns a tuple of lists of the
        successive values (not including the holding value).
        
        For example, for a protocol that has holding value ``-120mV`` and
        performs steps to ``-100mV``, ``-80mV``, and ``-40mV`` the returned
        output will be::
        
            ([-100, -80, -40])
            
        For a more complicated protocol, where each step is followed by a step
        down to ``-140mV``, the output would be::
        
            ([-100, -80, -40], [-140, -140, -140])
            
            
        """
        # Get epoch functions set by _read_protocol
        dinfo, einfo_exists, einfo = self._epoch_functions
        if not einfo_exists(channel):
            raise Exception('Missing protocol data')
        # Create list of step lists
        levels = []
        for e in einfo(channel):
            kind = e['type']
            if kind not in epoch_types:
                raise NotImplementedError('Unknown epoch type: '+str(kind))
            if kind == EPOCH_DISABLED:
                continue
            elif kind == EPOCH_STEPPED:
                levels.append([])
            else:
                raise ValueError('Unsupported epoch type: '
                    + epoch_types(kind))
        levels = tuple(levels)
        # Gather steps
        for i in range(self._sweepsPerRun):
            for k, e in enumerate(einfo(channel)):
                levels[k].append(e['init_level'] + e['level_inc']*i)
        return levels
    def __iter__(self):
        """
        Returns an iterator over all sweeps
        """
        return iter(self._sweeps)
    def __len__(self):
        """
        Returns the number of sweeps in this file.
        """
        return len(self._sweeps)
    def protocol(self):
        """
        Returns an interator over the protocol data.
        """
        return iter(self._protocol)
    def _read_datetime(self):
        """
        Reads the date/time this file was recorded
        """
        # Get date and time
        if self._version < 2:
            t1 = str(self._header['lFileStartDate'])
            t2 = float(self._header['lFileStartTime'])
        else:
            t1 = str(self._header['uFileStartDate'])
            t2 = float(self._header['uFileStartTimeMS']) / 1000
        YY = int(t1[0:4])
        MM = int(t1[4:6])
        DD = int(t1[6:8])
        hh = int(t2 / 3600)
        mm = int((t2 - hh * 3600) / 60)
        ss = t2 - hh * 3600 - mm * 60
        ms = int((ss % 1) * 1e6)
        ss = int(ss)
        return datetime.datetime(YY, MM, DD, hh, mm, ss, ms)
    def _read_header(self):
        """
        Reads the file's header.
        """
        class struct_file(file):
            """
            Struct wrapper around file
            """
            def read_f(self, format, offset = None):
                """
                Read and unpack a file section using the given format.
                """
                if offset is not None:
                    self.seek(offset)
                return struct.unpack(format,self.read(struct.calcsize(format)))
        def ups(val):
            """
            Unpack a single value or, if the given value isn't singular, leave
            it alone.
            """
            if len(val) != 1:
                return val
            val = val[0]
            if type(val) == str and len(val) > 0 and ord(val[0]) == 0:
                return None
            return val
        fid = struct_file(self._filepath, 'rb')
        # Get ABF Format version (pClamp < 10 is version 1, after is version 2)
        sig = fid.read(4)
        if sig == 'ABF ':
            version = 1
        elif sig == 'ABF2':
            version = 2
        else:
            raise NotImplementedError('Unknown ABF Format "' + str(sig) + '".')
        # Gather header fields
        header = OrderedDict()
        for key, offset, format in headerFields[version]:
            header[key] = ups(fid.read_f(format, offset))
        # Get uniform file version number
        if version < 2:
            self._version = np.round(header['fFileVersionNumber'] * 100) / 100.0
        else:
            n = header['fFileVersionNumber']
            self._version = n[3] + 0.1 * n[2] + 0.01 * n[1] + 0.001 * n[0]
        #self._version = version = header['fFileVersionNumber']
        # Get file start time in seconds
        if version < 2:
            header['lFileStartTime'] += header['nFileStartMillisecs'] / 1000
        else:
            header['lFileStartTime'] = header['uFileStartTimeMS'] / 1000
        if version < 2:
            # Version 1: Only read tags
            tags = []
            for i in range(header['lNumTagEntries']):
                fid.seek(header['lTagSectionPtr'] + i*64)
                tag = OrderedDict()
                for key, format in TagInfoDescription:
                    tag[key] = ups(fid.read_f(format))
                tags.append(tag)
            header['tags'] = tags
            self.strings = []
        else:
            # Version 2
            # Find location of file sections
            sections = OrderedDict()
            for i, s in enumerate(abf2FileSections) :
                index, data, length = fid.read_f('IIl', 76 + i * 16)
                sections[s] = OrderedDict()
                sections[s]['index'] = index
                sections[s]['data'] = data
                sections[s]['length'] = length
            header['sections'] = sections
            # String section contains channel names and units
            fid.seek(sections['Strings']['index'] * BLOCKSIZE)
            strings = fid.read(sections['Strings']['data'])
            # Starts with header we need to skip
            # DWORD dwSignature;    4 bytes
            # DWORD dwVersion;      4 bytes
            # UINT  uNumStrings;    4 bytes
            # UINT  uMaxSize;       4 bytes
            # ABFLONG  lTotalBytes; 4 bytes
            # UINT  uUnused[6];     24 bytes
            # Total: 44 bytes
            strings = strings[44:]
            # C-style string termination
            strings = strings.split('\x00')
            self.strings = strings
            # Read tag section
            tags = []
            offs = sections['Tag']['index'] * BLOCKSIZE
            size = sections['Tag']['data']
            for i in range(sections['Tag']['length']):
                fid.seek(offs + i * size)
                tag = OrderedDict()
                for key, format in TagInfoDescription :
                    tag[key] = ups(fid.read_f(format))
                tags.append(tag)
            header['tags'] = tags
            # Read protocol section
            protocol = OrderedDict()
            offs = sections['Protocol']['index'] * BLOCKSIZE
            fid.seek(offs)
            for key, format in protocolFields:
                protocol[key] = ups(fid.read_f(format))
            header['protocol'] = protocol
            # Read analog-digital conversion sections
            adc = []
            offs = sections['ADC']['index'] * BLOCKSIZE
            size = sections['ADC']['data']
            for i in range(sections['ADC']['length']):
                ADC = OrderedDict()
                fid.seek(offs + i * size)
                for key, format in ADCFields:
                    ADC[key] = ups(fid.read_f(format))
                # Get channel name and unit
                ADC['ADCChNames'] = strings[ADC['lADCChannelNameIndex'] - 1]
                ADC['ADCChUnits'] = strings[ADC['lADCUnitsIndex'] - 1]
                adc.append(ADC)
            header['listADCInfo'] = adc
            # Read DAC section
            dac = []
            offs = sections['DAC']['index'] * BLOCKSIZE
            size = sections['DAC']['data']
            for i in range(sections['DAC']['length']):
                fid.seek(offs + size * i)
                DAC = OrderedDict()
                for key, format in DACFields:
                    DAC[key] = ups(fid.read_f(format))
                DAC['sDACChannelName']=strings[DAC['lDACChannelNameIndex']-1]
                DAC['sDACChannelUnits']=strings[DAC['lDACChannelUnitsIndex']-1]
                dac.append(DAC)
            header['listDACInfo'] = dac
            # Read UserList section
            userlists = []
            for i in range(sections['UserList']['length']):
                fid.seek(offs + size * i)
                UserList = OrderedDict()
                for key, format in UserListFields:
                    UserList[key] = ups(fid.read_f(format))
                userlists.append(DAC)
            header['listUserListInfo'] = userlists
            # Read epoch-per-DAC section
            # The resulting OrderedDict has the following structure:
            #  - the first index is the DAC number
            #  - the second index is the epoch number
            header['epochInfoPerDAC'] = OrderedDict()
            offs = sections['EpochPerDAC']['index'] * BLOCKSIZE
            size = sections['EpochPerDAC']['data']
            info = OrderedDict()
            for i in range(sections['EpochPerDAC']['length']) :
                fid.seek(offs + size * i)
                einf = OrderedDict()
                for key, format in EpochInfoPerDACFields:
                    einf[key] = ups(fid.read_f(format))
                DACNum   = einf['nDACNum']
                EpochNum = einf['nEpochNum']
                if not DACNum in info: info[DACNum] = OrderedDict()
                info[DACNum][EpochNum] = einf
            header['epochInfoPerDAC'] = info
        fid.close()
        return header
    def _read_protocol(self):
        """
        Reads the protocol stored in the ABF file and converts it to an analog
        signal.
        
        Only works for episodic stimulation, without any user lists.
        
        The resulting analog signal has the same size as the recorded signals,
        so not the full length of the protocol! This is different from the
        values returned by the Myokit 
        """
        # Only episodic stimulation is supported.
        if self._mode != ACMODE_EPISODIC_STIMULATION:
            return []
        h = self._header
        # Step 1: Gather information about the protocol
        if self._version < 2:
            # Before version 2: Sections are fixed length, locations absolute
            self._numberOfTrials     = h['lNumberOfTrials']
            self._trialStartToStart  = h['fTrialStartToStart']
            self._runsPerTrial       = h['lRunsPerTrial']
            self._runStartToStart    = h['fRunStartToStart']
            self._sweepsPerRun      = h['lSweepsPerRun']
            self._sweepStartToStart = h['fEpisodeStartToStart']
            # Number of samples in a channel for each sweep
            # (Only works for fixed-length, high-speed-osc or episodic)
            nSam = h['lNumSamplesPerEpisode'] / h['nADCNumChannels']
            def dinfo(index, name):
                return h[name][index]
            def einfo_exists(index):
                return True
            def einfo(index):
                lo = index * 8
                hi = index + 8
                for i in range(lo, hi):
                    yield {
                        'type' : h['nEpochType'][i],
                        'init_duration' : h['lEpochInitDuration'][i],
                        'duration_inc' : h['lEpochDurationInc'][i],
                        'init_level' : h['fEpochInitLevel'][i],
                        'level_inc' : h['fEpochLevelInc'][i],
                        }
                raise StopIteration
            self._epoch_functions = (dinfo, einfo_exists, einfo)
        else:
            # Version 2 uses variable length sections
            p = h['protocol']
            # Trials, runs, sweeps
            # (According to the manual, there should only be 1 trial!)
            self._numberOfTrials     = p['lNumberOfTrials']
            self._trialStartToStart  = p['fTrialStartToStart']
            self._runsPerTrial       = p['lRunsPerTrial']
            self._runStartToStart    = p['fRunStartToStart']
            self._sweepsPerRun      = p['lSweepsPerRun']
            self._sweepStartToStart = p['fSweepStartToStart']
            # Number of samples in a channel in a single sweep
            nSam = p['lNumSamplesPerEpisode'] / h['sections']['ADC']['length']
            # Compatibility functions
            def dinfo(index, name):
                return h['listDACInfo'][index][name]
            def einfo_exists(index):
                return index in h['epochInfoPerDAC']
            def einfo(index):
                for e in h['epochInfoPerDAC'][index].itervalues():
                    yield {
                        'type' : e['nEpochType'],
                        'init_duration' : e['lEpochInitDuration'],
                        'duration_inc' : e['lEpochDurationInc'],
                        'init_level' : e['fEpochInitLevel'],
                        'level_inc' : e['fEpochLevelInc'] ,
                        }
                raise StopIteration
            self._epoch_functions = (dinfo, einfo_exists, einfo)
        # Step 2: Generate analog signals corresponding to the waveforms
        # suggested by the 'epochs' in the protocol
        # User lists are not supported
        if self._version < 2:
            if any(self._header['nULEnable']):
                return []
        else:
            for userlist in self._header['listUserListInfo']:
                if userlist['nULEnable']:
                    return []
        sweeps = []
        # Number of DAC channels = number of channels that can be used
        #  to output a stimulation
        nDac = self.protocol_channels()
        start = 0
        for iSweep in range(h['lActualSweeps']):
            sweep = Sweep(nDac)
            # Create channels for this sweep
            for iDac in range(nDac):
                c = Channel(self)
                c._name = dinfo(iDac, 'sDACChannelName').strip()
                c._unit = dinfo(iDac, 'sDACChannelUnits').strip()
                if self._version < 2:
                    c._numb = iDac
                else:
                    c._numb = int(dinfo(iDac, 'lDACChannelNameIndex'))
                c._data = np.ones(nSam) * dinfo(iDac, 'fDACHoldingLevel')
                c._rate = self._rate
                c._start = start
                sweep[iDac] = c
                # No stimulation info for this channel? Then continue
                if not einfo_exists(iDac): continue
                # Save last sample index
                i_last = int(nSam * 15625 / 1e6) #TODO: What's this?
                # For each 'epoch' in the stimulation signal
                for e in einfo(iDac):
                    kind = e['type']
                    if kind not in epoch_types:
                        raise NotImplementedError('Unknown epoch type: '
                            + str(kind))
                    if kind == EPOCH_DISABLED:
                        continue
                    elif kind == EPOCH_STEPPED:
                        dur = e['init_duration']
                        inc = e['duration_inc']
                        i1 = i_last
                        i2 = i_last + dur + iSweep * inc
                        if i2 > nSam:
                            # The protocol may extend beyond the number of
                            # samples in the recording
                            i2 = nSam
                        level = e['init_level'] + e['level_inc'] * iSweep
                        c._data[i1:i2] = level * np.ones(len(range(i2 - i1)))
                        i_last += dur
                        if i_last > nSam:
                            # The protocol may extend beyond the number of
                            # samples in the recording
                            break
                    else:
                        print('Warning: Unsupported epoch type: '
                            + epoch_types(kind))
                        continue
            sweeps.append(sweep)
            start += self._sweepStartToStart
        return sweeps
    def _read_sweeps(self):
        """
        Reads the data from an ABF file and returns a list of sweeps
        """
        header = self._header
        version = self._version
        nc = self._nc
        # Sampling rate is constant for all sweeps and channels
        #TODO: This won't work for 2-rate protocols
        rate = self._rate
        # Get binary integer format
        dt = np.dtype('i2') if header['nDataFormat'] == 0 else np.dtype('f4')
        # Get number of channels, create a numpy memory map
        if version < 2:
            # Old files, get info from fields stored directly in header
            o = header['lDataSectionPtr'] * BLOCKSIZE \
                       + header['nNumPointsIgnored'] * dt.itemsize
            n = header['lActualAcqLength']
        else:
            # New files, get info from appropriate header section
            o = header['sections']['Data']['index'] * BLOCKSIZE
            n = header['sections']['Data']['length']
        data = np.memmap(self._filepath, dt, 'r', shape = (n), offset = o)
        # Load list of sweeps (Sweeps are called 'episodes' in ABF < 2)
        if version < 2:
            n = header['lSynchArraySize']
            o = header['lSynchArrayPtr'] * BLOCKSIZE
        else:
            n = header['sections']['SynchArray']['length']
            o = header['sections']['SynchArray']['index'] * BLOCKSIZE
        if n > 0:
            dt = [('offset', 'i4'), ('len', 'i4')]
            sdata = np.memmap(self._filepath, dt, 'r', shape=(n), offset=o)
        else:
            sdata = np.empty((1), dt)
            sdata[0]['len'] = data.size
            sdata[0]['offset'] = 0
        # Get data
        pos = 0
        # Data structure
        sweeps = []
        # Time-offset at start of sweep
        start = sdata[0]['offset'] / rate
        for j in range(sdata.size):
            # Create a new sweep
            sweep = Sweep(nc)
            # Get the number of data points
            size = sdata[j]['len']
            # Calculate the correct size for variable-length event mode
            if self._mode == ACMODE_VARIABLE_LENGTH_EVENTS:
                if version < 2:
                    f = float(header['fSynchTimeUnit'])
                else:
                    f = float(header['protocol']['fSynchTimeUnit'])
                if f != 0:
                    size /= f
            # Get a memory map to the relevant part of the data
            part = data[pos : pos + size]
            pos += size
            part = part.reshape((part.size / nc, nc)).astype('f')
            # If needed, reformat the integers
            if header['nDataFormat'] == 0:
                # Data given as integers? Convert to floating point
                for i in range(nc):
                    factor, offset = self._get_conversion_factors(i)
                    part[:,i] *= factor
                    part[:,i] += offset
            # Create channel
            if self._mode != ACMODE_EPISODIC_STIMULATION:
                # All modes except episodic stimulation
                start = sdata[j]['offset'] / rate
            for i in range(nc):
                c = Channel(self)
                c._data = part[:,i]
                if version < 2:
                    c._name = str(header['sADCChannelName'][i])
                    c._unit = str(header['sADCUnits'][i])
                    c._numb = int(header['nADCPtoLChannelMap'][i])
                    # Get telegraphed info
                    def get(field):
                        try:
                            return float(header[field][i])
                        except KeyError:
                            return None
                    if get('nTelegraphEnable'):
                        c._type = int(get('nTelegraphMode') or 0)
                        c._cm = get('fTelegraphMembraneCap')
                        c._rs = get('fTelegraphAccessResistance')
                        c._lopass = get('fTelegraphFilter')
                    # Updated low-pass cutoff
                    if header['nSignalType'] != 0:
                        # If a signal conditioner is used, the cutoff frequency
                        # is an undescribed "complex function" of both low-pass
                        # settings...
                        c._lopass = None
                else:
                    c._name = str(header['listADCInfo'][i]['ADCChNames'])
                    c._unit = str(header['listADCInfo'][i]['ADCChUnits'])
                    c._numb = int(header['listADCInfo'][i]['nADCNum'])
                    # Get telegraphed info
                    if header['listADCInfo'][i]['nTelegraphEnable']:
                        c._type = int(header['listADCInfo'][i][
                            'nTelegraphMode'])
                        c._cm = float(header['listADCInfo'][i][
                            'fTelegraphMembraneCap'])
                        c._rs = float(header['listADCInfo'][i][
                            'fTelegraphAccessResistance'])
                        c._lopass = float(header['listADCInfo'][i][
                            'fTelegraphFilter'])
                    # Updated low-pass cutoff
                    if 'nSignalType' in header['protocol']:
                        if header['protocol']['nSignalType'] != 0:
                            # If a signal conditioner is used, the cutoff
                            # frequency is an undescribed "complex function" of
                            # both low-pass settings...
                            c._lopass = None
                c._rate  = rate
                c._start = start
                sweep[i] = c
            if self._mode == ACMODE_EPISODIC_STIMULATION:
                # Increase time according to sweeps in episodic stim. mode
                start += self._sweepStartToStart
            # Store sweep
            sweeps.append(sweep)
        return sweeps
    def _set_conversion_factors(self):
        """
        Calculates the conversion factors to convert integer data from the ABF
        file to floats.
        """
        self._adc_factors = []
        self._adc_offsets = []
        h = self._header
        if self._version < 2:
            for i in range(self._nc):
                # Multiplier
                f = ( h['fInstrumentScaleFactor'][i]
                    * h['fADCProgrammableGain'][i]
                    * h['lADCResolution']
                    / h['fADCRange'])
                # Signal conditioner used?
                if h['nSignalType'] != 0:
                    f *= h['fSignalGain'][i]
                # Additional gain?
                if h['nTelegraphEnable'][i]:
                    f *= h['fTelegraphAdditGain'][i]
                # Set fina gain factor
                self._adc_factors.append(1.0 / f)
                # Shift
                s = h['fInstrumentOffset'][i]
                # Signal conditioner used?
                if h['nSignalType'] != 0:
                    s -= h['fSignalOffset'][i]
                # Set final offset
                self._adc_offsets.append(s)
        else:
            a = h['listADCInfo']
            p = h['protocol']
            for i in range(self._nc):
                # Multiplier
                f = ( a[i]['fInstrumentScaleFactor']
                    * a[i]['fADCProgrammableGain']
                    * p['lADCResolution']
                    / p['fADCRange'])
                # Signal conditioner used?
                if 'nSignalType' in h:
                    if h['nSignalType'] != 0:
                        f *= a[i]['fSignalGain']
                # Additional gain?
                if a[i]['nTelegraphEnable']:
                    f *= a[i]['fTelegraphAdditGain']
                # Set final gain factor
                self._adc_factors.append(1.0 / f)
                # Shift
                s = a[i]['fInstrumentOffset']
                # Signal conditioner used?
                if 'nSignalType' in h:
                    if h['nSignalType'] != 0:
                        s -= a[i]['fSignalOffset']
                # Set final offset
                self._adc_offsets.append(s)
class Sweep(object):
    """
    Represents a single sweep (also called an 'episode')
    
    A sweep is represented as a fixed-size list of channels.
    """
    def __init__(self, n):
        super(Sweep, self).__init__()
        n = int(n)
        if n < 0:
            raise ValueError('Number channels cannot be negative.')
        self._nc = n    # Number of channels
        self._channels = [None]*n
    def __getitem__(self, key):
        return self._channels[key] # Handles slices etc.
    def __iter__(self):
        return iter(self._channels)
    def __len__(self):
        return self._nc
    def __setitem__(self, key, value):
        if type(key) == slice:
            raise ValueError('Assignment with slices is not supported.')
        self._channels[key] = value
class Channel(object):
    """
    Represents an analog signal for a single channel.

    To obtain this channel's formatted data, use times() and trace()
    """
    def __init__(self, parent_file):
        super(Channel, self).__init__()
        self._parent_file = parent_file  # The abf file this channel is from
        self._type = TYPE_UNKNOWN   # Type of recording
        self._name = None   # This channel's name
        self._numb = None   # This channel's index (see note below)
        self._unit = None   # The units this channel's data is in
        self._data = None   # The raw data points
        self._rate = None   # Sampling rate in Hz
        self._start = None  # The signal start time
        self._cm = None     # The reported membrane capacitance        
        self._rs = None     # The reported access resistance
        self._lopass = None # The reported low-pass filter cut-off frequency
        # Note that the channel indices are not necessarily sequential! So a
        # file with 2 channels can have indices 0 and 3.
    def name(self):
        """
        Returns the name set for this channel.
        """
        return self._name
    def number(self):
        """
        Returns the channel index used by pClamp. Note that this does not
        necessarily equal its index in the python sweep data!
        """
        return self._numb
    def __str__(self):
        return 'Channel(' + str(self._numb) + ' "' + str(self._name) \
            + '"); ' + str(len(self._data)) + ' points sampled at ' \
            + str(self._rate) + 'Hz, starts at t=' + str(self._start)
    def times(self):
        """
        Returns a copy of the values on the time axis.
        """
        n = len(self._data)
        f = 1.0 / self._rate
        return np.arange(self._start, self._start + n * f, f)[0:n]
    def values(self):
        """
        Returns a copy of the values on the data axis.
        """
        return np.array(self._data, copy=True)
# Some python struct types:
#   f   float
#   h   short
#   i   int
#   s   string
# Size of block alignment in ABF Files
BLOCKSIZE = 512
# Header fields for versions 1 and 2
# Stored as (key, offset, format) where format corresponds to a struct
#  unpacking format as documented in:
#  http://docs.python.org/library/struct.html#format-characters
headerFields = {
    1 : [
        ('fFileSignature', 0, '4s'),       # Coarse file version indication
        # Group 1, File info and sizes
        ('fFileVersionNumber', 4, 'f'),    # Version number as float
        ('nOperationMode', 8, 'h'),        # Acquisition mode
        ('lActualAcqLength', 10, 'i'),
        ('nNumPointsIgnored', 14, 'h'),
        ('lActualSweeps', 16, 'i'),
        ('lFileStartDate', 20, 'i'),       # Added!
        ('lFileStartTime', 24, 'i'),       #
        ('lStopWatchTime', 28, 'i'),       # Added!
        ('fHeaderVersionNumber', 32, 'f'), # Added!
        ('nFileType', 36, 'h'),            # Added!
        ('nMSBinFormat', 38, 'h'),         # Added!
        # Group 2, file structure
        ('lDataSectionPtr', 40, 'i'),      # And added quite a few more...
        ('lTagSectionPtr', 44, 'i'),
        ('lNumTagEntries', 48, 'i'),
        ('lScopeConfigPtr', 52, 'i'),
        ('lNumScopes', 56, 'i'),
        ('x_lDACFilePtr', 60, 'i'),
        ('x_lDACFileNumEpisodes', 64, 'i'),
        ('lDeltaArrayPtr', 72, 'i'),
        ('lNumDeltas', 76, 'i'),
        ('lVoiceTagPtr', 80, 'i'),
        ('lVoiceTagEntries', 84, 'i'),
        ('lSynchArrayPtr', 92, 'i'),
        ('lSynchArraySize', 96, 'i'),
        ('nDataFormat', 100, 'h'),
        ('nSimultaneousScan', 102, 'h'),
        # Group 3, Trial hierarchy
        ('nADCNumChannels', 120, 'h'),
        ('fADCSampleInterval', 122, 'f'),
        ('fADCSecondSampleInterval', 126, 'f'),
        ('fSynchTimeUnit', 130, 'f'),
        ('fSecondsPerRun', 134, 'f'),
        ('lNumSamplesPerEpisode', 138, 'i'),
        ('lPreTriggerSamples', 142, 'i'),
        ('lSweepsPerRun', 146, 'i'),        # Number of sweeps/episodes per run
        ('lRunsPerTrial', 150, 'i'),
        ('lNumberOfTrials', 154, 'i'),
        ('nAveragingMode', 158, 'h'),
        ('nUndoRunCount', 160, 'h'),
        ('nFirstEpisodeInRun', 162, 'h'),
        ('fTriggerThreshold', 164, 'f'),
        ('nTriggerSource', 168, 'h'),
        ('nTriggerAction', 170, 'h'),
        ('nTriggerPolarity', 172, 'h'),
        ('fScopeOutputInterval', 174, 'f'),
        ('fEpisodeStartToStart', 178, 'f'),
        ('fRunStartToStart', 182, 'f'),
        ('fTrialStartToStart', 186, 'f'),
        ('lAverageCount', 190, 'f'),
        ('lClockChange', 194, 'f'),
        ('nAutoTriggerStrategy', 198, 'h'),
        # Group 4, Display parameters
        # Group 5, Hardware info
        ('fADCRange', 244, 'f'),
        ('lADCResolution', 252, 'i'),
        # Group 6, Environment info
        ('nFileStartMillisecs', 366, 'h'),
        # Group 7, Multi-channel info
        ('nADCPtoLChannelMap', 378, '16h'),
        ('nADCSamplingSeq', 410, '16h'),
        ('sADCChannelName', 442, '10s'*16),
        ('sADCUnits', 602, '8s'*16) ,
        ('fADCProgrammableGain', 730, '16f'),
        ('fADCDisplayAmplification', 794, '16f'),
        ('fADCDisplayOffset', 858, '16f'),
        ('fInstrumentScaleFactor', 922, '16f'),
        ('fInstrumentOffset', 986, '16f'),
        ('fSignalGain', 1050, '16f'),       # The fSignal fields are only
        ('fSignalOffset', 1114, '16f'),     # relevant if a signal conditioner
        ('fSignalLowpassFilter', 1178, '16f'),  # was used.
        ('fSignalHighpassFilter', 1242, '16f'),
        ('sDACChannelName', 1306, '10s'*4),
        ('sDACChannelUnits', 1346, '8s'*4),
        ('fDACScaleFactor', 1378, '4f'),
        ('fDACHoldingLevel', 1394, '4f'),
        ('nSignalType', 1410, 'h'), # 1 if a "CyberAmp 320/380" signal
                                    # conditioner was used
        # Group 8, There doesn't seem to be a group 8
        # Group 9, Wave data
        ('nDigitalEnable', 1436, 'h'),
        ('x_nWaveformSource', 1438, 'h'),
        ('nActiveDACChannel', 1440, 'h'),
        ('x_nInterEpisodeLevel', 1442, 'h'),
        ('x_nEpochType', 1444, '10h'),
        ('x_fEpochInitLevel', 1464, '10f'),
        ('x_fEpochLevelInc', 1504, '10f'),
        ('x_nEpochInitDuration', 1544, '10h'),
        ('x_nEpochDurationInc', 1564, '10h'),
        ('nDigitalHolding', 1584, 'h'),
        ('nDigitalInterEpisode', 1586, 'h'),
        ('nDigitalValue', 2588, '10h'),
        ('lDACFilePtr', 2048, '2i'),            # Pointer to protocol?
        ('lDACFileNumEpisodes', 2056, '2i'),
        ('fDACCalibrationFactor', 2074, '4f'),
        ('fDACCalibrationOffset', 2090, '4f'),
        ('nWaveformEnable', 2296, '2h'),
        ('nWaveformSource', 2300, '2h'),
        ('nInterEpisodeLevel', 2304, '2h'),
        ('nEpochType', 2308, '20h'),       # 2 CMD channels with 10 values each
        ('fEpochInitLevel', 2348, '20f'),
        ('fEpochLevelInc', 2428, '20f'),
        ('lEpochInitDuration', 2508, '20i'),
        ('lEpochDurationInc', 2588, '20i'),
        # Group 10, DAC Output file
        ('fDACFileScale', 2708, 'd'),
        ('fDACFileOffset', 2716, 'd'),
        ('lDACFileEpisodeNum', 2724, 'i'),
        ('nDACFileADCNum', 2732, '2h'),
        ('sDACFilePath', 2736, '256s'*2), # 256 * 2char = utf8? or 2 strings?
        # Group 11,
        # Group 12, User list parameters
        ('nULEnable', 3360, '4h'),
        ('nULParamToVary', 3368, '4h'),
        ('nULParamValueList0', 3376, '256s'*4),
        ('nULRepeat', 4400, '4h'),
        # Group 13,
        # Group 14,
        # Group 15, Leak subtraction
        # Group 16, Misc
        # Group 17, Trains
        # Group 18, Application version data
        # Group 19
        # Group 20
        # Group 21 Skipped
        # Group 22
        # Group 23 Post-processing
        # Group 24 Legacy stuff
        # Group 6 again?
        ('nTelegraphEnable', 4512, '16h'),
        ('fTelegraphAdditGain', 4576, '16f'),
        ],
    2 : [
        ('fFileSignature', 0, '4s'),       # Coarse file version indication
        ('fFileVersionNumber', 4, '4b'),   # Version number as 4 signed chars
        ('uFileInfoSize', 8, 'I'),
        ('lActualSweeps', 12, 'I'),
        ('uFileStartDate', 16, 'I'),       # File start data YYYYMMDD
        ('uFileStartTimeMS', 20, 'I'),     # Time of day in ms ?
        ('uStopwatchTime', 24, 'I'),
        ('nFileType', 28, 'H'),
        ('nDataFormat', 30, 'H'),
        ('nSimultaneousScan', 32, 'H'),
        ('nCRCEnable', 34, 'H'),
        ('uFileCRC', 36, 'I'),
        ('FileGUID', 40, 'I'),
        ('uCreatorVersion', 56, 'I'),
        ('uCreatorNameIndex', 60, 'I'),
        ('uModifierVersion', 64, 'I'),
        ('uModifierNameIndex', 68, 'I'),
        ('uProtocolPathIndex', 72, 'I')
        ]
    }
# ABF2 File sections
abf2FileSections = [
    'Protocol',
    'ADC',
    'DAC',
    'Epoch',
    'ADCPerDAC',
    'EpochPerDAC',
    'UserList',
    'StatsRegion',
    'Math',
    'Strings',
    'Data',
    'Tag',
    'Scope',
    'Delta',
    'VoiceTag',
    'SynchArray',
    'Annotation',
    'Stats',
    ]
# ABF2 Fields in the tag section
TagInfoDescription = [
       ('lTagTime', 'i'),
       ('sComment', '56s'),
       ('nTagType', 'h'),
       ('nVoiceTagNumber_or_AnnotationIndex', 'h'),
   ]
# ABF2 Fields in the protocol section
protocolFields = [
         ('nOperationMode', 'h'),
         ('fADCSequenceInterval', 'f'),
         ('bEnableFileCompression', 'b'),
         ('sUnused1', '3s'),
         ('uFileCompressionRatio', 'I'),
         ('fSynchTimeUnit', 'f'),
         ('fSecondsPerRun', 'f'),
         ('lNumSamplesPerEpisode', 'i'),
         ('lPreTriggerSamples', 'i'),
         ('lSweepsPerRun', 'i'),
         ('lRunsPerTrial', 'i'),
         ('lNumberOfTrials', 'i'),
         ('nAveragingMode', 'h'),
         ('nUndoRunCount', 'h'),
         ('nFirstEpisodeInRun', 'h'),
         ('fTriggerThreshold', 'f'),
         ('nTriggerSource', 'h'),
         ('nTriggerAction', 'h'),
         ('nTriggerPolarity', 'h'),
         ('fScopeOutputInterval', 'f'),
         ('fSweepStartToStart', 'f'),
         ('fRunStartToStart', 'f'),
         ('lAverageCount', 'i'),
         ('fTrialStartToStart', 'f'),
         ('nAutoTriggerStrategy', 'h'),
         ('fFirstRunDelayS', 'f'),
         ('nChannelStatsStrategy', 'h'),
         ('lSamplesPerTrace', 'i'),
         ('lStartDisplayNum', 'i'),
         ('lFinishDisplayNum', 'i'),
         ('nShowPNRawData', 'h'),
         ('fStatisticsPeriod', 'f'),
         ('lStatisticsMeasurements', 'i'),
         ('nStatisticsSaveStrategy', 'h'),
         ('fADCRange', 'f'),
         ('fDACRange', 'f'),
         ('lADCResolution', 'i'),
         ('lDACResolution', 'i'),
         ('nExperimentType', 'h'),
         ('nManualInfoStrategy', 'h'),
         ('nCommentsEnable', 'h'),
         ('lFileCommentIndex', 'i'),
         ('nAutoAnalyseEnable', 'h'),
         ('nSignalType', 'h'),
         ('nDigitalEnable', 'h'),
         ('nActiveDACChannel', 'h'),
         ('nDigitalHolding', 'h'),
         ('nDigitalInterEpisode', 'h'),
         ('nDigitalDACChannel', 'h'),
         ('nDigitalTrainActiveLogic', 'h'),
         ('nStatsEnable', 'h'),
         ('nStatisticsClearStrategy', 'h'),
         ('nLevelHysteresis', 'h'),
         ('lTimeHysteresis', 'i'),
         ('nAllowExternalTags', 'h'),
         ('nAverageAlgorithm', 'h'),
         ('fAverageWeighting', 'f'),
         ('nUndoPromptStrategy', 'h'),
         ('nTrialTriggerSource', 'h'),
         ('nStatisticsDisplayStrategy', 'h'),
         ('nExternalTagType', 'h'),
         ('nScopeTriggerOut', 'h'),
         ('nLTPType', 'h'),
         ('nAlternateDACOutputState', 'h'),
         ('nAlternateDigitalOutputState', 'h'),
         ('fCellID', '3f'),
         ('nDigitizerADCs', 'h'),
         ('nDigitizerDACs', 'h'),
         ('nDigitizerTotalDigitalOuts', 'h'),
         ('nDigitizerSynchDigitalOuts', 'h'),
         ('nDigitizerType', 'h'),
         ]
# ABF2 Fields in the ADC section
ADCFields = [
         ('nADCNum', 'h'),
         ('nTelegraphEnable', 'h'),
         ('nTelegraphInstrument', 'h'),
         ('fTelegraphAdditGain', 'f'),
         ('fTelegraphFilter', 'f'),
         ('fTelegraphMembraneCap', 'f'),
         ('nTelegraphMode', 'h'),
         ('fTelegraphAccessResistance', 'f'),
         ('nADCPtoLChannelMap', 'h'),
         ('nADCSamplingSeq', 'h'),
         ('fADCProgrammableGain', 'f'),
         ('fADCDisplayAmplification', 'f'),
         ('fADCDisplayOffset', 'f'),
         ('fInstrumentScaleFactor', 'f'),
         ('fInstrumentOffset', 'f'),
         ('fSignalGain', 'f'),      # The fSignal fields are only relevant if a
         ('fSignalOffset', 'f'),    # signal conditioner was used
         ('fSignalLowpassFilter', 'f'),
         ('fSignalHighpassFilter', 'f'),
         ('nLowpassFilterType', 'b'),
         ('nHighpassFilterType', 'b'),
         ('fPostProcessLowpassFilter', 'f'),
         ('nPostProcessLowpassFilterType', 'c'),
         ('bEnabledDuringPN', 'b'),
         ('nStatsChannelPolarity', 'h'),
         ('lADCChannelNameIndex', 'i'),
         ('lADCUnitsIndex', 'i'),
         ]
# ABF2 Fields in the DAC section
DACFields = [
       ('nDACNum', 'h'),
       ('nTelegraphDACScaleFactorEnable', 'h'),
       ('fInstrumentHoldingLevel', 'f'),
       ('fDACScaleFactor', 'f'),
       ('fDACHoldingLevel', 'f'),
       ('fDACCalibrationFactor', 'f'),
       ('fDACCalibrationOffset', 'f'),
       ('lDACChannelNameIndex', 'i'),
       ('lDACChannelUnitsIndex', 'i'),
       ('lDACFilePtr', 'i'),
       ('lDACFileNumSweeps', 'i'),
       ('nWaveformEnable', 'h'),
       ('nWaveformSource', 'h'),
       ('nInterEpisodeLevel', 'h'),
       ('fDACFileScale', 'f'),
       ('fDACFileOffset', 'f'),
       ('lDACFileEpisodeNum', 'i'),
       ('nDACFileADCNum', 'h'),
       ('nConditEnable', 'h'),
       ('lConditNumPulses', 'i'),
       ('fBaselineDuration', 'f'),
       ('fBaselineLevel', 'f'),
       ('fStepDuration', 'f'),
       ('fStepLevel', 'f'),
       ('fPostTrainPeriod', 'f'),
       ('fPostTrainLevel', 'f'),
       ('nMembTestEnable', 'h'),
       ('nLeakSubtractType', 'h'),
       ('nPNPolarity', 'h'),
       ('fPNHoldingLevel', 'f'),
       ('nPNNumADCChannels', 'h'),
       ('nPNPosition', 'h'),
       ('nPNNumPulses', 'h'),
       ('fPNSettlingTime', 'f'),
       ('fPNInterpulse', 'f'),
       ('nLTPUsageOfDAC', 'h'),
       ('nLTPPresynapticPulses', 'h'),
       ('lDACFilePathIndex', 'i'),
       ('fMembTestPreSettlingTimeMS', 'f'),
       ('fMembTestPostSettlingTimeMS', 'f'),
       ('nLeakSubtractADCIndex', 'h'),
       ('sUnused', '124s'),
   ]
# ABF2 Fields in the DAC-Epoch section
EpochInfoPerDACFields = [
       ('nEpochNum', 'h'),
       ('nDACNum', 'h'),
       ('nEpochType', 'h'),
       ('fEpochInitLevel', 'f'),
       ('fEpochLevelInc', 'f'),
       ('lEpochInitDuration', 'i'),
       ('lEpochDurationInc', 'i'),
       ('lEpochPulsePeriod', 'i'),
       ('lEpochPulseWidth', 'i'),
       ('sUnused', '18s'),
       ]
UserListFields = [
       ('nListNum', 'h'),
       ('nULEnable', 'h'),
       ('nULParamToVary', 'h'),
       ('nULRepeat', 'h'),
       ('lULParamValueListIndex', 'i'),
       ('sUnused', '52s'),
       ]
# Types of epoch (see head of file for description)
EPOCH_DISABLED  = 0
EPOCH_STEPPED   = 1
EPOCH_RAMPED    = 2
EPOCH_RECTANGLE = 3
EPOCH_TRIANGLE  = 4
EPOCH_COSINE    = 5
EPOCH_UNUSED    = 6 # Legacy issue: "was ABF_EPOCH_TYPE_RESISTANCE"
EPOCH_BIPHASIC  = 7
epoch_types = {
    EPOCH_DISABLED  : 'Disabled',
    EPOCH_STEPPED   : 'Stepped waveform (square pulse)',
    EPOCH_RAMPED    : 'Ramp waveform (fixed-angle in- or decrease)',
    EPOCH_RECTANGLE : 'Rectangular pulse train',
    EPOCH_TRIANGLE  : 'Triangular waveform',
    EPOCH_COSINE    : 'Cosine waveform',
    EPOCH_UNUSED    : 'Unused',
    EPOCH_BIPHASIC  : 'Biphasic pulse train',
    }
# Fields in the epoch section (abf2)
#EpochInfoDescription = [
#       ('nEpochNum', 'h'),
#       ('nDigitalValue', 'h'),
#       ('nDigitalTrainValue', 'h'),
#       ('nAlternateDigitalValue', 'h'),
#       ('nAlternateDigitalTrainValue', 'h'),
#       ('bEpochCompression', 'b'),
#       ('sUnused', '21s'),
#   ]
ACMODE_VARIABLE_LENGTH_EVENTS  = 1
ACMODE_FIXED_LENGTH_EVENTS     = 2
ACMODE_GAP_FREE                = 3
ACMODE_HIGH_SPEED_OSCILLOSCOPE = 4
ACMODE_EPISODIC_STIMULATION    = 5
acquisition_modes = {
    # Variable length sweeps, triggered by some event
    ACMODE_VARIABLE_LENGTH_EVENTS : 'Variable-length events mode',
    # Fixed length sweeps, triggered by some event, may overlap
    ACMODE_FIXED_LENGTH_EVENTS : 'Event-driven fixed-length mode',
    # Continuous recording
    ACMODE_GAP_FREE : 'Gap free mode',
    # Fixed length non-overlapping, sweeps, triggered by some event
    ACMODE_HIGH_SPEED_OSCILLOSCOPE : 'High-speed oscilloscope mode',
    # Fixed length non-overlapping sweeps
    ACMODE_EPISODIC_STIMULATION : 'Episodic stimulation mode'
    }
# DAC channel types
TYPE_UNKNOWN = 0
TYPE_VOLTAGE_CLAMP = 1
TYPE_CURRENT_CLAMP = 2
TYPE_CURRENT_CLAMP_ZERO = 4
type_modes = {
    0 : TYPE_UNKNOWN,
    1 : TYPE_VOLTAGE_CLAMP,
    2 : TYPE_CURRENT_CLAMP,
    4 : TYPE_CURRENT_CLAMP_ZERO,
    }
type_mode_names = { 
    0 : 'Unknown',
    1 : 'Voltage clamp',
    2 : 'Current clamp',
    4 : 'Current clamp zero',
    }
# User list parameter to vary
'''
CONDITNUMPULSES 0
CONDITBASELINEDURATION 1
CONDITBASELINELEVEL 2
CONDITSTEPDURATION 3
CONDITSTEPLEVEL 4
CONDITPOSTTRAINDURATION 5
CONDITPOSTTRAINLEVEL 6
EPISODESTARTTOSTART 7
INACTIVEHOLDING 8
DIGITALINTEREPISODE 9
PNNUMPULSES 10
PARALLELVALUE(0-9) 11-20
EPOCHINITLEVEL(0-9) 21-30
EPOCHINITDURATION(0-9) 31-40
EPOCHTRAINPERIOD(0-9) 41-50
EPOCHTRAINPULSEWIDTH(0-9) 51-60
'''
