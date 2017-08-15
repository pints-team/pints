#
# Exports to matlab/octave
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
class MatlabExporter(myokit.formats.TemplatedRunnableExporter):
    """
    This :class:`Exporter <myokit.formats.Exporter>` generates a matlab/octave
    based model implementation and solving routine.

    Only the model definition is exported. A basic pacing mechanism is
    added that will need to be customized to match the model's needs.
    No post-processing is included.
    
    The following inputs are provided:

    ``time``
        The current simulation time
    ``pace``
        The current value of the pacing system, implemented using a very simple
        pacing mechanism.

    To run simulations, run the file ``main.m``.
    """
    def info(self):
        import inspect
        return inspect.getdoc(self)
    def _dir(self, root):
        return os.path.join(root, 'matlab', 'template')
    def _dict(self):
        return {
            'main.m' : 'main.m',
            'model.m' : 'model.m',
            'constants.m' : 'constants.m',
            'ifthenelse.m' : 'ifthenelse.m',
            'model_wrapper.m' : 'model_wrapper.m',
            }
    def _vars(self, model, protocol):
        import myokit.formats.matlab as matlab
        # Pre-process model
        model.reserve_unique_names(*matlab.keywords)
        model.reserve_unique_names(
            't',
            'c',
            'y', 'y0',
            'nBeats', 'iBeat',
            't1', 't2', 'T', 'Y',
            'flags',
            'pace',
            'model', 'model_wrapper',
            'pcl',
            'stim_duration',
            'stim_offset'
        )
        model.create_unique_names()
        # Variable naming function
        def v(var):
            if isinstance(var, myokit.Derivative):
                return 'ydot(' + str(1 + var.var().indice()) + ')'
            if isinstance(var, myokit.Name):
                var = var.var()
            if var.is_constant():
                return 'c.' + var.uname()
            else:
                return var.uname()
        # Expression writer
        ew = matlab.MatlabExpressionWriter()
        ew.set_lhs_function(v)
        ew.set_condition_function('ifthenelse')
        # Process bound variables
        bound_variables = model.prepare_bindings({
            'time' : 't',
            'pace' : 'pace',
            })
        # Common variables
        equations = model.solvable_order()
        components = []
        for comp in equations:
            if comp != '*remaining*':
                components.append(model[comp])
        # Return variables
        return {
            'v' : v,
            'e' : ew.eq,
            'model' : model,
            'components' : components,
            'equations' : equations,
            'bound_variables' : bound_variables,
            }
