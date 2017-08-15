#
# Factory methods for pacing protocols
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
"""
This module contains factory methods to create Protocol objects
programmatically.
"""
def blocktrain(period, duration, offset=0, level=1.0, limit=0):
    """
    Creates an train of block pulses.

    Each pulse lasts ``duration`` time units and a pulse is initiated every
    ``period`` time units.

    An optional offset to the first pulse can be specified using ``offset`` and
    the level of each pulse can be set using ``level``. To limit the number of
    pulses generated set limit to a non-zero value.
    """
    p = myokit.Protocol()
    #          level, start, duration, period=0, multiplier=0
    p.schedule(level, offset, duration, period, limit)
    return p
def bpm2bcl(bpm, m=1e-3):
    """
    Converts a beats-per-minute number to a basic cycle length in ms. For
    example a bpm of 60 equals a bcl of 1000ms.

    >>> import myokit
    >>> print(myokit.pacing.bpm2bcl(60))
    1000.0

    To use a different unit scaling, change the optional parameter ``m``. For
    example, with m set to 1 the returned units are seconds:

    >>> print(myokit.pacing.bpm2bcl(120, 1))
    0.5
    """
    return 60.0 / m / bpm
def constant(level):
    """
    Creates a very simple protocol where the pacing variable is held constant
    at a given level specified by the argument ``level``.
    """
    t = 1e9
    p = myokit.Protocol()
    p.schedule(level, 0, t, t, 0)
    return p
def steptrain(vsteps, vhold, tpre, tstep, tpost=0):
    """
    Creates a series of increasing or decreasing steps away from the fixed
    holding potential ``vhold``, towards the voltages listed in ``vsteps``.
  
      1. For the first ``tpre`` time units, the pacing variable is held at the
         value given by ``vhold``.
      2. For the next ``tstep`` time units, the pacing variable is held at a
         value from ``vsteps``
      3. For the next ``tpost`` time units, the pacing variable is held at
         ``vhold`` again.

    These three steps are repeated for each value in the ``vsteps``.
    """
    # Check time arguments
    if tpre < 0:
        raise ValueError('Time tpre can not be negative.')
    if tstep < 0:
        raise ValueError('Time tstep can not be negative.')
    if tpost < 0:
        raise ValueError('Time tpost can not be negative.')
    # Create protocol
    p = myokit.Protocol()
    time = 0
    for vstep in vsteps:
        if tpre > 0:
            p.schedule(vhold, time, tpre)
            time += tpre
        if tstep > 0:
            p.schedule(vstep, time, tstep)
            time += tstep
        if tpost > 0:
            p.schedule(vhold, time, tpost)
            time += tpost
    return p
def steptrain_linear(vmin, vmax, dv, vhold, tpre, tstep, tpost=0):
    """
    Creates a series of increasing or decreasing steps away from a holding
    value (typically a holding potential). This type of protocol is commonly
    used to measure activation or inactivation in ion channel models.

      1. For the first ``tpre`` time units, the pacing variable is held at the
         value given by ``vhold``.
      2. For the next ``tstep`` time units, the pacing variable is held at a
         value from ``vsteps``
      3. For the next ``tpost`` time units, the pacing variable is held at the
         value ``vhold`` again.

    These three steps are repeated for each value in the range from ``vmin``
    up to (but not including) ``vmax``, with an increment specified as
    ``dv``.
    """
    # Check v arguments
    if vmax > vmin:
        if dv <= 0:
            raise ValueError('vmax > vmin so dv must be strictly positive.')
    else:
        if dv >= 0:
            raise ValueError('vmax > vmin so dv must be negative.')
    # Check time arguments
    if tpre < 0:
        raise ValueError('Time tpre can not be negative.')
    if tstep < 0:
        raise ValueError('Time tstep can not be negative.')
    if tpost < 0:
        raise ValueError('Time tpost can not be negative.')
    # Create protocol
    p = myokit.Protocol()
    time = 0
    for i in range(abs((vmax - vmin) / dv)):
        if tpre > 0:
            p.schedule(vhold, time, tpre)
            time += tpre
        if tstep > 0:
            p.schedule(vmin + i * dv, time, tstep)
            time += tstep
        if tpost > 0:
            p.schedule(vhold, time, tpost)
            time += tpost
    return p
#def void():
#    """
#    Creates an empty protocol.
#    """
#    return None
