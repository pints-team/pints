<?
#
# model_wrapper.m :: Creates a wrapper around the model function so that it can
# be used in matlab/octave style ode solvers.
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
?>% Function wrapper
function ydot = model_wrapper(t, y, c)
    if (mod(t - c.stim_offset, c.pcl) < c.stim_duration)
        pace = 1;
    else
        pace = 0;
    end
    ydot = model(t, y, c, pace);
end
