#
# Imports selected types of protocols from files in Axon Binary Format
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
import myokit.formats
info = \
"""
This importer can read simple square pulse protocols from files in the
Axon Binary Format used by Axon Instruments and Molecular Devices.
"""
class AbfImporter(myokit.formats.Importer):
    """
    This :class:`Importer <myokit.formats.Importer>` can import protocols from
    files in Axon Binary Format.
    """
    def info(self):
        return info
    def supports_protocol(self):
        return True
    def protocol(self, filename, channel=None):
        """
        Attempts to load the protocol from the file at ``filename``.
        
        If specified, the channel index ``channel`` will be used to select
        which channel in the AbfFile to convert to a protocol
        """
        from myokit.formats.axon import AbfFile
        abf = AbfFile(filename)
        return abf.myokit_protocol(channel)
