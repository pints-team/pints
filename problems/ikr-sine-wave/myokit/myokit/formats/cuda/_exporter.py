#
# Exports to a CUDA kernel
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
class CudaKernelExporter(myokit.formats.TemplatedRunnableExporter):
    """
    This :class:`Exporter <myokit.formats.Exporter>` creates an unoptimised
    CUDA cell kernel that calculates a forward Euler step for a single cell.
    
    Only a kernel file is created, no driver class is included and no support
    for protocol export is provided.
    
    A value must be bound to ``diffusion_current`` which represents the
    current flowing from cell to cell. This is defined as positive when the
    cell is acting as a source, negative when it acts like a sink. In other
    words, it is defined as an outward current.
    
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
        return os.path.join(root, 'cuda')
    def _dict(self):
        return {'kernel.cu' : 'kernel.cu'}
    def _vars(self, model, protocol):
        return {'model' : model}
