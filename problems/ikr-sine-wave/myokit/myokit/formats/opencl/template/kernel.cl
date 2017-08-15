<?
#
# cell.cl
#
# A pype template for an OpenCL kernel
#
# Required variables
# -------------------------------------------------------------
# model           A model (cloned) with independent components
# native_math     True or False
# precision       A myokit precision constant
# bound_variables
# -------------------------------------------------------------
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
import myokit.formats.opencl as opencl
   
# Get equations
equations = model.solvable_order()

# Delete "*remaning*" group, guaranteed to be empty with independent components
del(equations['*remaining*'])

# Get component order
comp_order = equations.keys()
comp_order = [model.get(c) for c in comp_order]

# Get component inputs/output arguments
comp_in, comp_out = model.map_component_io(
    omit_states = True,
    omit_derivatives = False,
    omit_constants = True,
    )

# Bound variables will be passed in to every function as needed, so they can be
# removed from the input/output lists
def clear_io_list(comp_list):
    for comp, clist in comp_list.iteritems():
        for var in bound_variables:
            lhs = var.lhs()
            while lhs in clist:
                clist.remove(lhs)
clear_io_list(comp_in)
clear_io_list(comp_out)

# Components that use one of the bound variables should get it as an input
# variable.
for comp, clist in comp_in.iteritems():
    for bound in bound_variables:
        lhs = bound.lhs()
        if lhs in clist:
            continue        
        for var in comp.variables(deep=True):
            if var.rhs().depends_on(lhs):
                clist.append(lhs)
                break

# Get expression writer
w = opencl.OpenCLExpressionWriter(precision=precision, native_math=native_math)

# Define var/lhs function
ptrs = []
def set_pointers(names=None):
    """
    Tells the expression writer to write the given variable names (given as
    LhsExpression objects) as pointers.
    
    Calling set_pointers a second time clears the first list. Calling with
    ``names=None`` unsets all pointers.
    """
    global ptrs
    ptrs = []
    if names is not None:
        ptrs = list(names)
def v(var):
    """
    Accepts a variable or a left-hand-side expression and returns its C
    representation.
    """
    if isinstance(var, myokit.Derivative):
        # Explicitly asked for derivative
        pre = '*' if var in ptrs else ''        
        return pre + 'D_' + var.var().uname()
    if isinstance(var, myokit.Name):
        var = var.var()
    if var in bound_variables:
        return bound_variables[var]
    pre = '*' if myokit.Name(var) in ptrs else ''        
    return pre + var.uname()
w.set_lhs_function(v)

# Tab
tab = '    '

# To render last function inline, set "last_component" to the last component
#last_component = None
last_component = comp_order[-1]

?>/*
OpenCL kernel for <?= model.name() ?>

Generated on <?= myokit.date() ?> by myokit opencl export

*/
#define n_state <?=str(model.count_states())?>

<?
if precision == myokit.SINGLE_PRECISION:
    print('/* Using single precision floats */')
    print('typedef float Real;')
else:
    print('/* Using double precision floats */')
    print('typedef double Real;')

print('')
print('/* Constants */')
for group in equations.itervalues():
    for eq in group.equations(const=True):
        if isinstance(eq.rhs, myokit.Number):
            print('#define ' + v(eq.lhs) + ' ' + w.ex(eq.rhs))

print('')
print('/* Calculated constants */')
for group in equations.itervalues():
    for eq in group.equations(const=True):
        if not isinstance(eq.rhs, myokit.Number):
            print('#define ' + v(eq.lhs) + ' (' + w.ex(eq.rhs) + ')')

print('')
print('/* Aliases of state variables. */')
for var in model.states():
    print('#define ' + var.uname() + ' state[offset + ' + str(var.indice()) + ']')

print('')
for comp, ilist in comp_in.iteritems():
    if comp == last_component:
        continue
    olist = comp_out[comp]
    if len(olist) == 0:
        continue

    # Comment
    print('/*')
    print('Component: ' + comp.name())
    if 'desc' in comp.meta:
        print(comp.meta['desc'])
    print('*/')

    # Function header
    args = ['const int cid', 'const int offset', '__global Real *state']
    args.extend(['Real '  + v(lhs) for lhs in ilist])
    args.extend(['__private Real *' + v(lhs) for lhs in olist])
    set_pointers(olist)
    name = 'calc_' + comp.name()
    print('void ' + name + '(' + ', '.join(args) + ')')
    print('{')

    # Equations
    for eq in equations[comp.name()].equations(const=False):
        var = eq.lhs.var()
        pre = tab
        if not (eq.lhs in ilist or eq.lhs in olist):
            pre += 'Real '
        if var not in bound_variables:
            print(pre + w.eq(eq) + ';')

    print('}')
    print('')
    set_pointers(None)
?>
/*
 * Cell kernel.
 * Computes a single Euler-step for a single cell.
 */
__kernel void cell_step(const int n_cells, const Real time, const Real dt, __global Real *state, __global const Real* idiff_vec)
{
    const int cid = get_global_id(0);
    const int offset = cid * n_state;
    if(cid >= n_cells) return;
    
    /* Diffusion */
    Real idiff = idiff_vec[cid];
    
    /* Pacing */
    Real pace;
    if(cid < 4) {
        pace = (time > 10 && time < 10.5) ? 1 : 0;
    } else {
        pace = 0;
    }
    
<?
print(tab + '/* Evaluate derivatives */')
for comp in comp_order:
    ilist = comp_in[comp]
    olist = comp_out[comp]

    # Skip components without output
    if len(olist) == 0:
        continue

    # Skip last component (if in inline mode)
    if comp == last_component:
        continue

    # Declare any output variables
    for var in comp_out[comp]:
        print(tab + 'Real ' + v(var) + ' = 0;')

    # Function header
    args = ['cid', 'offset', 'state']
    args.extend([v(lhs) for lhs in ilist])
    args.extend(['&' + v(lhs) for lhs in olist])
    print(tab + 'calc_' + comp.name() + '(' + ', '.join(args) + ');')

if last_component:
    print(tab)
    print(tab + '/* Evaluate ' + last_component.name() + ' */')
    olist = comp_out[last_component]
    ilist = comp_in[last_component]
    for eq in equations[last_component.name()].equations(const=False):
        var = eq.lhs.var()
        pre = tab
        if not eq.lhs in ilist:
            pre += 'Real '
        if var not in bound_variables:
            print(pre + w.eq(eq) + ';')
            
?>
    /* Perform update */
<?
for var in model.states():
    print(tab + v(var) + ' += dt * ' + v(var.lhs()) + ';')
?>
}

/*
 * Diffusion kernel program
 * Performs a single diffusion step
 */
__kernel void diff_step(const int n_cells, const Real g, __global Real *state, __global Real* idiff)
{
    const int cid = get_global_id(0);
    const int offset = cid * n_state;
    if(cid >= n_cells) return;

    if(n_cells > 1) {
        if(cid == 0) {
            /* Diffusion, first cell */
            idiff[cid] = g * (state[0] - state[n_state]);
        } else if (cid == n_cells - 1) {
            /* Diffusion, last cell */
            idiff[cid] = g * (state[cid*n_state] - state[(cid-1)*n_state]);
        } else {
            /* Diffusion, middle cells */
            idiff[cid] = g * (2*state[cid*n_state] - state[(cid-1)*n_state] - state[(cid+1)*n_state]);
        }
    }
}

<?
print('/* Remove aliases of state variables. */')
for var in model.states():
    print('#undef ' + var.uname())
print('')
print('/* Remove constant definitions */')
for group in equations.itervalues():
    for eq in group.equations(const=True):
        print('#undef ' + v(eq.lhs))
?>
#undef n_state
