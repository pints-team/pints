#
# This module reads files in WCP format
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
import struct
import numpy as np
import myokit
class WcpFile(object):
    """
    Represents a read-only WinWCP file (``.wcp``), stored at the location
    pointed to by ``filepath``.
    
    Only files in the newer file format version 9 can be read. This version of
    the format was introduced in 2010. New versions of WinWCP can read older
    files and will convert them to the new format automatically when opened.
    
    WinWCP is a tool for recording electrophysiological data written by John
    Dempster of Strathclyde University.
    
    WinWCP files contain a number of records ``NR``, each containing data from
    ``NC`` channels. Every channel has the same length, ``NP`` samples.
    Sampling happens at a fixed sampling rate.
    """
    def __init__(self, filepath):
        # The path to the file and its basename
        self._filepath = os.path.abspath(filepath)
        self._filename = os.path.basename(filepath)
        # Records
        self._records = None
        self._channel_names = None
        self._nr = None # Records in file
        self._nc = None # Channels per record
        self._np = None # Samples per channel
        self._dt = None # Sampling interval
        # Time signal
        self._time = None
        # Open the file, extract its contents
        with open(filepath, 'rb') as f:
            self._(f)
    def _(self, f):
        """
        Reads the file header & data.
        """
        # Header size is between 1024 and 16380, depending on number of
        # channels in the file following:
        #   n = (int((n_channels - 1)/8) + 1) * 1024
        # Read first part of header, determine version and number of channels
        # in the file
        data = f.read(1024)
        h = [x.strip().split('=') for x in data.split('\n')]
        h = dict([(x[0].lower(), x[1]) for x in h if len(x) == 2])
        if int(h['ver']) != 9:
            raise ValueError('Only able to read format version 9. Given file'
                ' is in format version ' + str(h['ver']))
        # Get header size
        try:
            # Get number of 512 byte sectors in header
            #header_size = 512 * int(h['nbh'])
            # Seems to be size in bytes!
            header_size = int(h['nbh'])
        except KeyError:
            # Calculate header size based on number of channels
            header_size = (int((int(h['nc']) - 1) / 8) + 1) * 1024
        # Read remaining header data
        if header_size > 1024:
            data += f.read(header_size - 1024)
            h = [x.strip().split('=') for x in data.split('\n')]
            h = dict([(x[0].lower(), x[1]) for x in h if len(x) == 2])
        # Tidy up read data
        header = {}
        header_raw = {}
        for k, v in h.iteritems():
            # Convert to appropriate data type
            try:
                t = HEADER_FIELDS[k]
                if t == float:
                    # Allow for windows locale stuff
                    v = v.replace(',', '.')
                header[k] = t(v)
            except KeyError:
               header_raw[k] = v
        # Convert time
        # No, don't. It's in different formats depending on... the version?
        #if 'ctime' in header:
        #    print(header['ctime'])
        #    ctime = time.strptime(header['ctime'], "%d/%m/%Y %H:%M:%S")
        #    header['ctime'] = time.strftime('%Y-%m-%d %H:%M:%S', ctime)
        # Get vital fields from header
        self._nr = header['nr'] # Records in file
        self._nc = header['nc'] # Channels per record        
        try:
            self._np = header['np'] # Samples per channel
        except KeyError:
            self._np = (header['nbd'] * 512) / (2 * self._nc)
        # Get channel specific fields
        channel_headers = []
        self._channel_names = []
        for i in xrange(self._nc):
            j = str(i)
            c = {}
            for k, t in HEADER_CHANNEL_FIELDS.iteritems():
                c[k] = t(h[k + j])
            channel_headers.append(c)
            self._channel_names.append(c['yn'])
        # Analysis block size and data block size
        # Data is stored as 16 bit integers (little-endian)
        try:
            rab_size = 512 * header['nba']
        except KeyError:
            rab_size = header_size
        try:
            rdb_size = 512 * header['nbd']
        except KeyError:
            rdb_size = 2 * self._nc * self._np
        # Maximum A/D sample value at vmax
        adcmax = header['adcmax']
        # Read data records
        records = []
        offset = header_size
        for i in xrange(self._nr):
            # Read analysis block
            f.seek(offset)
            # Status of signal (Accepted or rejected, as string)
            rstatus = f.read(8)
            # Type of recording, as string
            rtype = f.read(4)
            # Group number (float set by the user)
            group_number = struct.unpack('<f', f.read(4))[0]
            # Time of recording, as float, not sure how to interpret
            rtime = struct.unpack('<f', f.read(4))[0]
            # Sampling interval: pretty sure this should be the same as the
            # file wide one in header['dt']
            rint = struct.unpack('<f', f.read(4))[0]
            # Maximum positive limit of A/D converter voltage range
            vmax = struct.unpack('<' + 'f'*self._nc, f.read(4 * self._nc))
            # String marker set by user
            marker = f.read(16)
            # Increase offset beyond analysis block
            offset += rab_size
            # Get data from data block
            data = np.memmap(self._filepath, np.dtype('<i2'), 'r',
                shape = (self._np, self._nc),
                offset = offset,
                )
            # Separate channels and apply scaling
            record = []
            for j in xrange(self._nc):
                h = channel_headers[j]
                s = float(vmax[j]) / float(adcmax) / float(h['yg'])
                d = np.array(data[:, h['yo']].astype('f4') * s)
                record.append(d)
            records.append(record)
            # Increase offset beyong data block
            offset += rdb_size
        self._records = records
        # Create time signal
        self._time = np.arange(self._np) * header['dt']
    def channels(self):
        """
        Returns the number of channels in this file.
        """
        return self._nc
    def channel_names(self):
        """
        Returns the names of the channels in this file.
        """
        return list(self._channel_names)
    def filename(self):
        """
        Returns the current file's name.
        """
        return self._filename
    def myokit_log(self):
        """
        Creates and returns a :class:`myokit.DataLog` containing all the
        data from this file.
        
        Each channel is stored under its own name, with an indice indicating
        the record it was from. Time is stored under ``time``.
        """
        log = myokit.DataLog()
        log.set_time_key('time')
        log['time'] = np.array(self._time)
        for i, record in enumerate(self._records):
            for j, data in enumerate(record):
                name = self._channel_names[j]
                log[name, i] = np.array(data)
        return log
    def path(self):
        """
        Returns the path to the currently opened file.
        """
        return self._filepath
    def plot(self):
        """
        Creates matplotlib plots of all data in this file.
        """
        import matplotlib.pyplot as pl
        for record in self._records:
            pl.figure()
            for k, channel in enumerate(record):
                pl.subplot(self._nc, 1, 1 + k)
                pl.plot(self._time, channel)
        pl.show()
    def records(self):
        """
        Returns the number of records in this file.
        """
        return self._nr
    def sampling_interval(self):
        """
        Returns the sampling interval used in this file.
        """
        return self._dt
    def times(self):
        """
        Returns the time points sampled at.
        """
        return np.array(self._time)
    def values(self, record, channel):
        """
        Returns the values of channel ``channel``, recorded in record
        ``record``.
        """
        return self._records[record][channel]
HEADER_FIELDS = {
    'ver' : int,        # WCP data file format version number
    'ctime' : str,      # Create date/time
    'nc' : int,         # No. of channels per record
    'nr' : int,         # No. of records in the file.
    'nbh' : int,        # No. of 512 byte sectors in file header block
    'nba' : int,        # No. of 512 byte sectors in a record analysis block
    'nbd' : int,        # No. of 512 byte sectors in a record data block
    'ad' : float,       # A/D converter input voltage range (V)
    'adcmax' : int,     # Maximum A/D sample value
    'np' : int,         # No. of A/D samples per channel
    'dt' : float,       # A/D sampling interval (s)
    'nz' : int,         # No. of samples averaged to calculate a zero level.
    'tu' : str,         # Time units
    'id' : str,         # Experiment identification line
    }
HEADER_CHANNEL_FIELDS = {
    'yn' : str,         # Channel name
    'yu' : str,         # Channel units
    'yg' : float,       # Channel gain factor mV/units
    'yz' : int,         # Channel zero level (A/D bits)
    'yo' : int,         # Channel offset into sample group in data block
    'yr' : int,         # ADCZeroAt, probably for old files
    }
#TODO: Find out if we need to do something with yz and yg
