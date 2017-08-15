#
# Exports to Ansi-C using OpenCL for parallelization
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
class OpenCLExporter(myokit.formats.TemplatedRunnableExporter):
    """
    This :class:`Exporter <myokit.formats.Exporter>` creates a cable simulation
    that can run on any OpenCL environment (GPU or CPU).
    
    A value must be bound to ``diffusion_current`` which represents the current
    flowing from cell to cell. This is defined as positive when the cell is 
    acting as a source, negative when it acts like a sink. In other words, it 
    is defined as an outward current.
    
    The membrane potential must be marked as ``membrane_potential``.
    
    By default, the simulation is set to log all state variables. This is nice
    to explore results with, but quite slow...
    
    Please keep in mind that CellML and other downloaded formats are typically
    not directly suitable for GPU simulation. Specifically, when simulating
    on single-precision devices a lot of divide-by-zero errors might crop up
    that remain hidden when using double precision single cell simulations on
    the CPU.
    """
    def info(self):
        import inspect
        return inspect.getdoc(self)
    def _dir(self, root):
        return os.path.join(root, 'opencl', 'template')
    def _dict(self):
        return {
            'cable.c'    : 'cable.c',
            'kernel.cl'  : 'kernel.cl',
            'plot.py'    : 'plot.py',
            'minilog.py' : 'minilog.py',
            'test'       : 'test',
        }
    def _vars(self, model, protocol):
        from myokit.formats.opencl import keywords
        # Clone model, merge interdependent components
        model = model.clone()
        model.merge_interdependent_components()
        # Process bindings, remove unsupported bindings, get map of bound
        # variables to internal names.
        bound_variables = model.prepare_bindings({
            'time' : 'time',
            'pace' : 'pace',
            'diffusion_current' : 'idiff',
            })
        # Reserve keywords
        model.reserve_unique_names(*keywords)
        model.reserve_unique_names(
            *['calc_' + c.name() for c in model.components()])
        model.reserve_unique_names(
            'cid',
            'dt',
            'g',
            'idiff',
            'idiff_vec'
            'n_cells',
            'offset',
            'pace',
            'pace_vec',
            'state',
            'time',
            )
        model.create_unique_names()
        # Return variables
        return {
            'model'           : model,
            'precision'       : myokit.SINGLE_PRECISION,
            'native_math'     : True,
            'bound_variables' : bound_variables,
            }
