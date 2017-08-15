#
# Creates common graphs
# Uses matplotlib
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
import numpy as np
# Don't import pyplot yet, this will cause a crash if no window environment
# is loaded.
def simulation_times(time=None, realtime=None, evaluations=None, mode='stair',
        axes=None, nbuckets=50, label=None):
    """
    Draws a graph of step sizes used during a simulation.

    Data can be passed in as ``time`` (simulation time) ``realtime``
    (benchmarked time during the simulation) and ``evaluations`` (the number
    of evaluations needed for each step). Which of these fields are required
    dependens on the chosen plot ``mode``:

    ``stair``
        Draws ``time`` on the x-axis, and the step number on the y-axis.
        In this plot, a high slope means the integrator is taking lots of
        steps.
    ``stair_inverse``
        Draws ``time`` on the y-axis, and the step number on the x-axis.
    ``load``
        Draws ``time`` on the x-axis, and log(1 / step size) on the y-axis.
        In this plot, high values on the y-axis should be found near difficult
        times on the x-axis
    ``histo``
        Lumps ``time`` into buckets (whose number can be specified using the
        argument ``nbuckets``) and counts the number of steps in each bucket.
        In the final result, the times corresponding to the buckets are plotted
        on the x axis and the number of evaluations in each bucket is plotted
        on the y axis.
    ``time_per_step``
        Uses the ``realtime`` argument to calculate the time taken to advance
        the solution each step.
        In the resulting plot, the step count is plotted on the x-axis, while
        the y-axis shows the time spent at this point.
    ``eval_per_step``
        Uses the ``evaluations`` entry to calculate the number of rhs
        evaluations required for each step.
        In the resulting plot, the step number is plotted on the x-axis, and
        the number of rhs evaluations for each step is plotted on the y-axis.

    The argument ``axes`` can be used to pass in a matplotlib axes object to be
    used for the plot. If none are given, the current axes obtained from
    ``pyplot.gca()`` are used.

    Returns a matplotlib axes object.
    """
    import matplotlib.pyplot as pl
    if axes is None:
        axes = pl.gca()
    def stair(ax, time, realtime, evaluations):
        if time is None:
            raise ValueError('This plotting mode requires "time" to be set.')
        time = np.array(time, copy=False)
        step = np.arange(0, len(time))
        ax.step(time, step, label=label)
    def stair_inverse(ax, time, realtime, evaluations):
        if time is None:
            raise ValueError('This plotting mode requires "time" to be set.')
        time = np.array(time, copy=False)
        step = np.arange(0, len(time))
        ax.step(step, time, label=label)
    def load(ax, time, realtime, evaluations):
        if time is None:
            raise ValueError('This plotting mode requires "time" to be set.')
        time = np.array(time, copy=False)
        size = np.log(1.0 / (time[1:] - time[:-1]))
        ax.step(time[1:], size, label=label)
    def histo(ax, time, realtime, evaluations):
        if time is None:
            raise ValueError('This plotting mode requires "time" to be set.')
        time = np.array(time, copy=False)
        zero = float(time[0])
        bucket_w = (time[-1] - zero) / nbuckets
        bucket_m = bucket_w * 0.5
        bucket_x = np.zeros(nbuckets)
        bucket_y = np.zeros(nbuckets)
        hi = zero
        for k in xrange(nbuckets):
            lo = hi
            hi = zero + (k + 1) * bucket_w
            bucket_x[k] = lo
            bucket_y[k] = np.sum((lo < time)*(time <= hi))
        bucket_y[0] += 1  # First bucket contains point lo == time
        ax.step(bucket_x, bucket_y, where='post', label=label)
    def time_per_step(ax, time, realtime, evaluations):
        if realtime is None:
            raise ValueError('This plotting mode requires "realtime" to be'
                ' set.')
        real = np.array(realtime) # Will be manipulated
        real = real[1:] - real[:-1]
        step = np.arange(1, 1+len(real))
        ax.step(step, real, where='mid', label=label)
    def eval_per_step(ax, time, realtime, evaluations):
        if evaluations is None:
            raise ValueError('This plotting mode requires "evaluations" to be'
                ' set.')
        evls = np.array(evaluations) # Will be manipulated
        evls = evls[1:] - evls[:-1]
        step = np.arange(1, 1+len(evls))
        ax.step(step, evls, where='mid', label=label)
    modes = {
        'stair'         : stair,
        'stair_inverse' : stair_inverse,
        'load'          : load,
        'histo'         : histo,
        'time_per_step' : time_per_step,
        'eval_per_step' : eval_per_step,
        }
    try:
        fn = modes[mode]
    except KeyError:
        raise ValueError('Selected mode not found. Avaiable modes are: '
            + ', '.join(['"'+x+'"' for x in modes.iterkeys()]))
    return fn(axes, time, realtime, evaluations)
def current_arrows(log, voltage, currents, axes=None):
    """
    Draws a graph of voltage versus time with arrows to indicate which currents
    are active at which stage of the action potential.

    The argument, ``log`` should be a :class:`myokit.DataLog` containing
    the data needed for the plot. The argument ``voltage`` should be the key in
    ``log`` that maps to the membrane potential.

    The list ``currents`` should contain all keys of currents to display.

    Returns a matplotlib axes object.
    """
    import matplotlib.pyplot as pl
    # Get currents, normalize with respect to total current at each time
    log = log.npview()
    traces = [log[x] for x in currents]
    times = log.time()
    memv = log[voltage]
    # Get sum of _absolute_ traces!
    I_total = np.zeros(len(traces[0]))
    for I in traces:
        I_total += abs(I)
    # Create axes
    ax = axes if axes is not None else pl.gca()
    # Plot membrane potential
    ax.plot(times, memv)
    ax.set_title(voltage)
    # Get width of time steps
    n = len(times)
    steps = np.concatenate((times[0:1], times, times[-1:]))
    steps = 0.5 * steps[2:] - 0.5 * steps[0:-2]
    # Find "zero" points, points of interest
    threshold_abs = 0.1
    threshold_int = 0
    for ii, I in enumerate(traces):
        # Capture parts where abs(I) is greather than the threshold and the
        # sign doesn't change
        parts = []
        indices = None
        sign = (I[0] >= 0)
        for k, i in enumerate(I):
            if abs(i) < threshold_abs or sign != (i >= 0):
                # Do nothing
                if indices is not None:
                    parts.append(indices)
                    indices = None
            else:
                # Store indices
                if indices is None:
                    indices = []
                indices.append(k)
            sign = (i >= 0)
        if indices is not None:
            parts.append(indices)
        # For each part, calculate
        #  the weighted midpoint in time
        #  the total charge transferred
        #  the average current
        #  the peak current
        #  the total charge transferred / the total sum charge transferred in
        #  that same time. This last measure can be used as a secondary
        #  threshold
        for part in parts:
            q_total = 0 # Sum of charge transferred
            t_total = 0 # Total time elapsed
            s_total = 0 # Sum of all currents in this time frame
            i_peak  = 0 # Max absolute current
            t_mid   = 0 # Weighted midpoint in time
            for k in part:
                t_total += steps[k]
                q_total += steps[k] * I[k]
                s_total += steps[k] * I_total[k]
                t_mid   += steps[k] * I[k] * times[k]
                i_peak = max(i_peak, abs(I[k]))
            # Test if relative total transferred charge is above threshold
            if abs(q_total / s_total) < threshold_int:
                continue
            # Weighted midpoint in time (weight is height * width)
            t_mid /= q_total
            # Average charge transferred = total current transferred
            i_total = q_total / t_total
            # Average current
            i_mean  = i_total / t_total
            # Add sign to peak current
            if sum(I) < 0: i_peak *= -1.0
            #if log is not None:
            #    log.append('-- ' + currents[ii] + ' '
            #        + '-'*(76-len(currents[ii])))
            #    log.append('Transferred charge (abs) :', q_total)
            #    log.append('Transferred charge (rel) :', q_total / s_total)
            #    log.append('Start    :', times[part[0]])
            #    log.append('End      :', times[part[-1]])
            #    log.append('Duration :', t_total)
            #    log.append('Midpoint :', t_mid)
            #    log.append('Peak current  :', i_peak)
            #    log.append('Mean current  :', i_mean)
            #    log.append('Total current :', i_total)
            # Add arrow
            k = np.nonzero(times >= t_mid)[0][0]
            ars = 'rarrow'
            arx = t_mid
            if k + 1 == len(times):
                ary = memv[k]
                arr = 0
            else:
                t1 = times[k]
                t2 = times[k+1]
                ary = (memv[k]*(t2 - t_mid) + memv[k+1]*(t_mid - t1)) / (t2-t1)
                arr = np.arctan2(t1-t2, memv[k+1]-memv[k]) * 180 / np.pi
                if sum(I) > 0: arr += 180
                if abs(arr) > 90:
                    arr = 180 + arr
                    ars = 'larrow'
            bbox_props = dict(boxstyle=ars+',pad=0.3', fc='w', ec='black',lw=1)
            ax.annotate(currents[ii], xy=(arx, ary), ha='center', va='center',
                        rotation=arr, size=14, bbox=bbox_props)
    return ax
def cumulative_current(log, currents, axes, labels=None, colors=None,
        integrate=False):
    """
    Plots a number of currents, one on top of the other, with the positive and
    negative parts of the current plotted separately.
    
    The advantage of this type of plot is that it shows the relative size of
    each current versus the others, and gives an indication of the total
    positive and negative current in a model.
    
    Accepts the following arguments:
    
    ``log``
        A :class:`myokit.DataLog` containing all the data to plot.
    ``currents``
        A list of keys, where each key corresponds to a current stored in
        ``log``.
    ``axes``
        The matplotlib axes to create the plot on.
    ``labels``
        Can be used to pass in a list containing the label to set for each
        current.
    ``colors``
        Can be used to pass in a list containing the colors to set for each
        current.
    ``integrate``
        Set this to ``True`` to plot total carried charge instead of currents.

    The best results are obtained if relatively constant currents are specified
    early. Another rule of thumb is to specify the currents roughly in the
    order they appear during an AP.    
    """
    import matplotlib
    import matplotlib.pyplot as pl
    # Get numpy version of log
    log = log.npview()
    # Get time
    t = log.time()
    # Get currents or charges
    if integrate:
        signals = [log.integrate(c) for c in currents]
    else:
        signals = [log[c] for c in currents]
    # Colors
    n = len(currents)
    if colors:
        while len(colors) < n:
            colors.extend(colors)
        custom = colors[0:n]
    else:
        # Colormap
        cmap = matplotlib.cm.get_cmap(name='spectral')
        colors = [cmap(i) for i in np.linspace(0.9, 0.1, len(currents))]
    # Offsets
    op = on = 0
    # Plot
    for k, c in enumerate(currents):
        # Get color
        color = colors[k]
        # Get label
        if labels:
            label = labels[k]
        else:
            if integrate:
                label = 'Q(' + c[c.find('.')+1:] + ')'
            else:
                label = c[c.find('.')+1:]
        # Split signal
        s = signals[k]
        p = np.maximum(s, 0) + op
        n = np.minimum(s, 0) + on
        # Plot!
        axes.fill_between(t, p, op, facecolor=color)
        axes.fill_between(t, n, on, facecolor=color)
        axes.plot(t, p, color=color, label=label)
        axes.plot(t, p, color='k', lw=1)
        axes.plot(t, n, color='k', lw=1)
        on = n
        op = p
