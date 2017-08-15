#
# Exports to Ansi C, using the CVODE libraries for integration
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
DIRNAME = 'ansic'
class AnsiCExporter(myokit.formats.TemplatedRunnableExporter):
    """
    This :class:`Exporter <myokit.formats.Exporter>` generates a runnable ansic
    C model implementation and integrator. The integration is based on the
    Sundials CVODE library, which is required to run the program.

    Both the model definition and pacing protocol are exported and the file
    is set up ready to run a simulation. No post-processing is included.

    Provides the following external variables:

    ``time``
        The current simulation time
    ``pace``
        The current value of the pacing system, implemented using the given
        protocol.

    Example plot script::
    
        import matplotlib.pyplot as pl
        with open('v.txt', 'r') as f:
          T,V = [], []
          for line in f:
            t,v = [float(x) for x in line.split()]
            T.append(t)
            V.append(v)
        pl.figure()
        pl.plot(T, V)
        pl.show()

    To compile in gcc, use::

        gcc -Wall -lm -lsundials_cvode -lsundials_nvecserial sim.c -o sim

    """
    def info(self):
        import inspect
        return inspect.getdoc(self)
    def _dir(self, root):
        return os.path.join(root, DIRNAME)
    def _dict(self):
        return {'sim.c' : 'sim.c'}
    def _vars(self, model, protocol):
        return {'model':model, 'protocol':protocol}
class AnsiCEulerExporter(myokit.formats.TemplatedRunnableExporter):
    """
    This :class:`Exporter <myokit.formats.Exporter>` generates an ansic C
    implementation using a simple explicit forward-Euler scheme.

    Both the model definition and pacing protocol are exported and the file
    is set up ready to run a simulation. No post-processing is included.

    Provides the following external variables:

    ``time``
        The simulation time
    ``pace``
        The value of the pacing variable, implemented using the given protocol.
    
    No labeled variables are required.
    
    Example plot script::
    
        import matplotlib.pyplot as pl
        with open('v.txt', 'r') as f:
          T,V = [], []
          for line in f:
            t,v = [float(x) for x in line.split()]
            T.append(t)
            V.append(v)
        pl.figure()
        pl.plot(T, V)
        pl.show()
    
    To compile using gcc::

        gcc -Wall -lm euler.c -o euler
       
    """
    def info(self):
        import inspect
        return inspect.getdoc(self)
    def _dir(self, root):
        return os.path.join(root, DIRNAME)
    def _dict(self):
        return {'euler.c' : 'euler.c'}
    def _vars(self, model, protocol):
        return {'model':model, 'protocol':protocol}
class AnsiCCableExporter(myokit.formats.TemplatedRunnableExporter):
    """
    This :class:`Exporter <myokit.formats.Exporter>` generates a 1d cable
    simulation using a simple forward-Euler scheme in ansi-C.

    Both the model definition and pacing protocol are exported and the file
    is set up ready to run a simulation. No post-processing is included.

    Provides the following external variables:

    ``time``
        The simulation time
    ``pace``
        The value of the pacing variable, implemented using the given protocol.
    ``diffusion_current``
        The current flowing from each cell to its neighbours. This will be
        positive if the cell is acting as a source, negative when it's acting
        as a sink.
    
    Requires the following labels to be set in the model:
    
    ``membrane_potential``
        The membrane potential.
        
    Variables are linked using ``diffusion_current``, this is calculated from 
    the membrane potentials as:
    
        i = g * ((V - V_next) - (V_last - V))
    
    At the boundaries, V is substituted for V_last or V_next.

    Example plot script::

        import myokit    
        import numpy as np
        import matplotlib.pyplot as pl
        from mpl_toolkits.mplot3d import axes3d
        d = myokit.load_csv('data.txt')
        n = 50 # Assuming 50 cells
        f = pl.figure()
        x = f.gca(projection='3d')
        z = np.ones(len(d['time']))
        for i in xrange(0, n):
            x.plot(d['time'], z*i, d.get(str(i)+'_V'))
        pl.show()

    To compile using gcc::

        gcc -Wall -lm cable.c -o cable
     
    """
    def info(self):
        import inspect
        return inspect.getdoc(self)
    def _dir(self, root):
        return os.path.join(root, DIRNAME)
    def _dict(self):
        return {'cable.c' : 'cable.c'}
    def _vars(self, model, protocol):
        return {'model':model, 'protocol':protocol}
