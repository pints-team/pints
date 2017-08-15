#
# Dependency graphing: creates graphical representations of myokit models
# Uses matplotlib.
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
import random
import collections
import myokit
# Don't import pyplot yet, this will crash if no window environment is loaded
def create_state_dependency_matrix(model, direct=False, knockout=[]):
    """
    Creates a matrix showing state variable dependency distances.

    Dependencies can either be direct (a relies on b) or indirect (a relies on
    b which relies on c, but a does not rely on c directly). To show only
    direct dependencies, set ``direct=True``, in this case the returned matrix
    will show the shape of the system's Jacobian.

    Variables can be "knocked out" by adding them to the list in the
    ``knockout`` parameter. If x depends on y and y depends on z, knocking out
    y will prevent the method from findind x's dependency on z.
    """
    # Create matrix of direct dependencies
    s = [x for x in model.states()]
    n = len(s)
    m = [0] * n
    deep = model.map_deep_dependencies(collapse=True, omit_states=False)
    i = 0
    for var in s:
        m[i] = [0] * n
        for dep in deep[var.lhs()]:
            if dep.var().is_state():
                m[i][dep.var().indice()] = 1
        i += 1
    if direct:
        return m
    # Create two copies of dependency matrix, apply knockouts
    p = []
    q = []
    for i in xrange(0, n):
        pp = []
        qq = []
        for j in xrange(0, n):
            v = m[i][j] if s[i] not in knockout else 0
            pp.append(v)
            qq.append(v)
        p.append(pp)
        q.append(qq)
    del(m)
    # The distance at step+1 will be set in q, calculated from p
    # At the end of each iteration, q is copied into p
    # At the end of this operation, p = q = the distances
    # Repeat this trick n-1 times...
    for index in xrange(1, n):
        for i in xrange(0, n):
            for j in xrange(0, n):
                # ... for every item
                if p[i][j] == index:
                    # ... where the value is the current index 1,2,3...
                    for k in xrange(0, n):
                        # ... update the entries
                        # ... unless a previous value was set
                        if q[i][k] == 0 and p[j][k] > 0:
                            q[i][k] = p[i][j] + p[j][k]
        for i in xrange(0, n):
            for j in xrange(0, n):
                p[i][j] = q[i][j]
    return p
def plot_state_dependency_matrix(model, direct=False, knockout=[],
    axes=None):
    """
    Creates a matrix showing state variable dependency distances.

    To show only direct (first-order) dependencies, set the optional argument
    ``direct`` to ``True``.

    Variables can be "knocked out" by adding them to the list in the
    ``knockout`` parameter. If x depends on y and y depends on z, knocking out
    y will prevent the method from findind x's dependency on z.

    Returns a matplotlib axes object.
    """
    import matplotlib.pyplot as pl
    # Create dependency matrix
    m = create_state_dependency_matrix(model, direct, knockout)
    n = len(m)
    # Configure axes
    #a = f.add_subplot(111, aspect='equal')
    a = axes if axes is not None else pl.gca()
    a.set_aspect('equal')
    a.set_xlim(0, n + 2)
    a.set_ylim(0, n)
    # Create matrix plot
    from matplotlib.patches import Rectangle
    w = 1.0 / (max(max(m))) # color weight
    p1 = 0.1                # Padding
    p2 = 1.0 - 2 * p1       # Square size
    def c(v):
        # Colormap
        if v == 1:
            return (0, 0, 0)
        else:
            vv = v * w
            if (vv > 1.0):
                return (0, 1.0, 0)
            return (vv, vv, 1.0)
    for y,row in enumerate(m):
        y = n - y - 1
        for x,v in enumerate(row):
            r = Rectangle((x+p1, y+p1), p2, p2, color=c(v))
            a.add_patch(r)
            r.set_clip_box(a.bbox)
        # Add colorbar
        r = Rectangle((1+n+p1, y+p1), p2, p2, color=c(1+y))
        a.add_patch(r)
        r.set_clip_box(a.bbox)
        a.text(1.5+n, y+0.5, str(y+1), horizontalalignment='center',
            verticalalignment='center')
    # Set tick labels
    names = [i.qname() for i in model.states()]
    a.set_xticks([i + 0.5 for i in xrange(0, n)])
    a.set_xticklabels(names)
    from matplotlib.transforms import Affine2D
    r = Affine2D.identity()
    a.set_yticks([i + 0.5 for i in xrange(0, n)])
    rnames = list(names)
    rnames.reverse()
    a.set_yticklabels(rnames)
    for tick in a.xaxis.iter_ticks():
        tick[0].label2On = True
        tick[0].label1On = False
        tick[0].label2.set_rotation('vertical')
    # Add axes labels
    a.annotate('Affecting variable -->', xy=(0,-0.05),
        ha='left', va='center', xycoords='axes fraction',
        textcoords='offset points')
    # Adjust subplot margins
    return a
class DiGraph(object):
    """
    A simple directed graph implementation.

    If desired, a digraph can be created from an n-by-n connectivity matrix,
    for example ``matrix=[[0, 1, 1], [0, 1, 0], [0, 0, 0]]``
    """
    def __init__(self, matrix=None):
        super(DiGraph, self).__init__()
        if isinstance(matrix, DiGraph):
            # Clone
            self.nodes = collections.OrderedDict()
            for node in matrix:
                self.add_node(node.uid)
            for node in matrix:
                for edge in node.edgo:
                    self.add_edge(node.uid, edge.uid)
        else:
            # Create new
            if matrix is None:
                matrix = []
            self.build_from_matrix(matrix)
    def __len__(self):
        return len(self.nodes)
    def __iter__(self):
        return self.nodes.itervalues()
    def __getitem__(self, key):
        return self.nodes.__getitem__(key)
    def add_edge(self, node1, node2):
        """
        Adds an edge from node1 to node2
        """
        node1 = self.uid_or_node(node1)
        node2 = self.uid_or_node(node2)
        node1.add_edge_to(node2)
    def add_node(self, node):
        """
        Adds a node. You can pass an existing ``Node`` object or an object to
        use as a new node's id.
        """
        if not isinstance(node, Node):
            node = Node(node)
        uid = node.uid
        if uid in self.nodes:
            raise Exception('Duplicate node id: "' + node.label() + '"')
        if node.graph is not None and node.graph != self:
            raise Exception('Node already in another graph: "'
                + node.label() + '"')
        self.nodes[uid] = node
        node.graph = self
        return node
    def build_from_matrix(self, matrix, edges_only=False):
        """
        Replaces this graph's structure with the graph defined by the given
        n by n connectivity matrix.

        If ``edges_only`` is set to True, the matrix must have the same size
        as the current number of nodes. In this, all existing edges will be
        removed and replaced by the ones given in the connectivity matrix.
        """
        # Check matrix structure
        n = len(matrix)
        for row in matrix:
            if len(row) != n:
                raise ValueError('Matrix must be n by n')
        if edges_only:
            # Test if matrix size is same as current graph
            if n != len(self.nodes):
                raise ValueError('Matrix must have same size as graph.')
            # Clear existing edges
            for node in self.nodes.itervalues():
                node.clear_edges()
        else:
            # Delete nodes
            self.nodes = collections.OrderedDict()
            # Create nodes
            for i in xrange(n):
                self.add_node(i)
        # Add edges
        nodes = self.nodes.values()
        for i, row in enumerate(matrix):
            for j, edge in enumerate(row):
                if edge:
                    self.add_edge(nodes[i], nodes[j])
    def cg_layers_dag(graph):
        """
        Returns a layering according to the Coffman-Graham ordering scheme.

        Returns a list of lists, where each inner list represents a consecutive
        layer.

        Will raise an exception if this digraph has cycles.
        """
        def isless(n, m):
            """
            Returns True if n < m

            Given two decreasing lists of integers::

                n = [n1, n2, ..., nt]
                m = [m1, m2, ..., ms]

            We say that n < m if either

                1. for some i >= 1, we have nj = mj for all j in [1,i-1] and
                    ni < mi
                    (So if all elements up to i are equal and the i-th element
                    of n is smaller)
                2. t < s and ni = mi for i in [1,t]
                    (m is equal to n + some extra values)

            Example::

                n = [9]
                m = [10, 9, 8, 7]
                -----------------
                The first 0 elements are equal and the 1st element of n is
                smaller than the first element of m, so n < m

                n = [10, 9, 7]
                m = [10, 9, 8, 7, 2]
                --------------------
                The first 2 elements are equal and the 3d element of n is
                smaller than the 3d element of m, so n < m

                n = [10, 9, 7, 6, 2]
                m = [10, 9, 8]
                --------------------
                The first 2 elements are equal and the 3d element of n is
                smaller than the 3d element of m, so n < m

                n = [10, 9, 8]
                m = [10, 9, 8, 7, 2]
                --------------------
                n is shorter than m and all values are equal: n < m

                n = [10, 9, 8]
                m = [10, 9, 8]
                --------------------
                Neither criterium met: n is not smaller than m

            """
            ni = iter(n)
            mi = iter(m)
            try:
                v = 0
                while v == 0:
                    v = ni.next() - mi.next()
                return v < 0
            except StopIteration:
                return len(n) < len(m)
        # Get minimal equivalent graph
        original = graph
        graph = graph.meg_dag()
        # Number of nodes
        n = len(graph)
        # Values of nodes
        P = collections.OrderedDict()
        for node in graph:
            P[node] = 0
        # Pick a random node with no leaving edges (successors)
        first = None
        for node in graph:
            if len(node.edgo) == 0:
                first = node
                break
        if first is None:
            raise Exception('Graph is not acyclical: no node without'
                ' successors found.')
        P[first] = 1
        # Assign remaining orders
        for i in xrange(2, n+1):
            kmin = None
            nmin = None
            for node in graph:
                # Only test unassigned node
                if P[node] > 0:
                    continue
                # Get values of successors
                k = [0] * len(node.edgo)
                for j, s in enumerate(node.edgo):
                    p = P[s]
                    if p == 0:
                        k = None
                        break
                    k[j] = p
                # Only test nodes whose successors have all been assigned
                if k is None:
                    continue
                k.sort(reverse=True)
                if kmin is None or k < kmin:
                    kmin = k
                    nmin = node
            # Assign value to node with minimum value sequence
            P[nmin] = i
        # Assign levels
        L = []      # Layers
        D = set()   # Contains nodes already assigned
        E = set()   # Nodes on this level
        level = 0
        for node in sorted(P, key=P.get):
            if not node.edgo.issubset(D):
                D = D.union(E)
                L.append([original[x.uid] for x in E])
                E.clear()
            E.add(node)
        L.append([original[x.uid] for x in E])
        # Convert to lists
        layers = []
        for layer in L:
            layers.append(list(layer))
        return layers
    def meg_dag(self):
        """
        Finds a Minimal Equivalent Graph (MEG) of a Directed Acyclic Graph
        (DAG) using the algorithm by Harry Hsu [1].

        Will raise an exception if this digraph has cycles.

        [1] An algorithm for finding a minimal equivalent graph of a digraph.
            Harry T. Hsu (1975) Journal of the Assoclatlon for Computing
            Machinery, Vol 22, No. 1, January 1975, pp 11-16
        """
        m = self.path_matrix()
        n = len(m)
        for i in xrange(n):
            if m[i][i]:
                raise Exception('Graph is not acyclical.')
        for j in xrange(n):
            for i in xrange(n):
                if m[i][j]:
                    for k in xrange(n):
                        if m[j][k]:
                            m[i][k] = 0
        graph = DiGraph(self)
        graph.build_from_matrix(m, edges_only = True)
        return graph
    def matrix(self):
        """
        Returns a connectivity matrix for this graph.
        """
        nodes = self.nodes.values()
        n = len(nodes)
        m = [0] * n
        for i, node in enumerate(nodes):
            row = [0] * n
            for j in xrange(0, n):
                if node.has_edge_to(nodes[j]):
                    row[j] = 1
            m[i] = row
        return m
    def node(self, uid):
        """
        Returns the node with this id
        """
        if not uid in self.nodes:
            raise IndexError('Node not found: "' + str(uid) + '"')
        return self.nodes[uid]
    def path_matrix(self):
        """
        Returns a path matrix for this graph, showing which nodes are reachable
        from which.
        """
        p = self.matrix()
        n = len(p)
        for i in xrange(0, n):
            for j in xrange(0, n):
                if i == j:
                    continue
                if p[j][i]:
                    for k in xrange(0, n):
                        if p[j][k] == 0:
                            p[j][k] = p[i][k]
            return p
    def text(self, matrix=False):
        """
        Returns an ascii view of this graph
        """
        if matrix:
            m = self.matrix()
            o = []
            for row in m:
                o.append(', '.join([str(x) for x in row]))
            return '\n'.join(o)
        else:
            if self.nodes:
                out = []
                for node in self.nodes.itervalues():
                    out.append('Node "' + node.label() + '"')
                    for edge in node.edgo:
                        out.append('  > Node "' + edge.label() + '"')
                    #out.append('')
                return '\n'.join(out)
            else:
                return '(empty graph)'
    def layout_layered(self):
        """
        Changes the x,y coordinates of this graph's node resulting in a layered
        layout.
        """
        # Get layers
        layers = self.cg_layers_dag()
        # Set initial positions
        fy = 1.0 / len(layers)
        for i, layer in enumerate(layers):
            y = fy * i
            n = len(layer)
            if n == 1:
                layer[0].x = 0.5
                layer[0].y = y
            else:
                fx = 1.0 / len(layer)
                for j, node in enumerate(layer):
                    node.y = y
                    node.x = j * fx
        # Iteratively update x coordinates using median rule
        max_runs = 100
        for n_runs in xrange(max_runs):
            change = False
            for i, layer in enumerate(layers):
                # Set nodes to median of neighbours
                for j, node in enumerate(layer):
                    n = len(node.edgi) + len(node.edgo)
                    if n > 0:
                        nodes = list(node.edgi)
                        nodes.extend(list(node.edgo))
                        nodes.sort(key=lambda snode: snode.x)
                        node.x = nodes[n/2].x
                # Fix possible issues with new coordinates
                org = list(layer)
                layer.sort(key=lambda snode: snode.x)
                if layer != org:
                    change = True
                # Layer-wide operations
                if i == 0:
                    # Bottom layer? Then redistribute nodes
                    fx = 1.0 / len(layer)
                    f1 = 0.8
                    f2 = 1.0 - f1
                    for k, node in enumerate(layer):
                        node.x = f1 * k * fx + f2 * node.x
                else:
                    # Resolve equal x positions
                    n = len(layer)
                    too_close = 0.1 / n
                    last = None
                    for j, node in enumerate(layer):
                        if last is None:
                            last = node
                            continue
                        if abs(node.x - last.x) < too_close:
                            eq = [last, node]
                            # Get last x
                            x1 = 0.0 if j == 1 else layer[j - 2].x
                            # Get next x
                            x2 = None
                            for k in xrange(j+1, n):
                                if abs(layer[k].x - node.x) >= too_close:
                                    x2 = layer[k].x
                                    break
                                eq.append(layer[k])
                            if x2 is None:
                                x2 = 1.0
                            # Space nodes between x1 and x2
                            dx = (x2 - x1) / (1.0 + len(eq))
                            for k, enode in enumerate(eq):
                                enode.x = x1 + dx * (1 + k)
                        last = node
            if not change:
                #print('Done in ' + str(n_runs) + ' runs.')
                break
    def remove_node(self, node):
        """
        Removes a node from this graph.
        """
        node = self.uid_or_node(node)
        node.clear_edges()
        del(self.nodes[node.uid])
    def uid_or_node(self, test):
        """
        Safely turns 'test' into a node. Throws exceptions if it can't.
        """
        if not isinstance(test, Node):
            return self.node(test)
        if test.graph != self:
            raise Exception('Node from other graph: "'+ test.label() +'"')
        if test.uid not in self.nodes:
            raise IndexError('Node not in graph: "' + test.label() + '"')
        if self.nodes[test.uid] != test:
            raise Exception('Different node found at the same id for: "'
                + test.label() + '"')
        return test
class Node(object):
    """
    Defines a node in a graph
    """
    def __init__(self, uid):
        """
        Creates a new, graphless node with the given identifier
        """
        super(Node, self).__init__()
        self.graph = None
        self.uid  = uid
        self.edgo = set()
        self.edgi = set()
        # Graphical
        self.x     = 0
        self.y     = 0
        self.rgba  = (0, 0, 1, 1)
        self.label = None
    def add_edge_to(self, node):
        """
        Ensures an edge from this node to another
        """
        node = self.graph.uid_or_node(node)
        self.edgo.add(node)
        node.edgi.add(self)
    def clear_edges(self):
        """
        Removes any edges leading from this node.
        """
        for node in self.edgo:
            node.edgi.remove(self)
        self.edgo.clear()
    def label(self):
        """
        Returns a label derived from this node
        """
        return str(self.uid)
    def outgoing(self):
        """
        Returns a list of nodes pointed at by this one
        """
        return list(self.edgo)
    def uid(self):
        """
        Returns this node's unique identifier
        """
        return self.uid
    def has_edge_from(self, test):
        """
        Returns true if test points at this node
        """
        test = self.graph.uid_or_node(test)
        return test in self.edgi
    def has_edge_to(self, test):
        """
        Returns true if this node points at test
        """
        test = self.graph.uid_or_node(test)
        return test in self.edgo
    def remove_edge_to(self, node):
        """
        Removes the edge from this node to the given node. An error is given
        if no such edge exists.
        """
        node = self.graph.uid_or_node(node)
        node.edgi.remove(self)
        self.edgo.remove(node)
    def __str__(self):
        return str(self.uid)
def plot_digraph(graph, axes=None, r_node=None):
    """
    Returns a DiGraph object to a set of matplotlib axes.

    Returns a matplotlib axes object.
    """
    import matplotlib.pyplot as pl
    import matplotlib.lines as lines
    import matplotlib.patches as patches
    limits = [None, None, None, None]
    size = None
    # Create axes
    ax = axes if axes is not None else pl.gca()
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    xmin = ymin = 1
    xmax = ymax = 0
    # Node height and width
    if r_node is None:
        w = h = 0.85 / len(graph)    # Just a heuristic
    else:
        w = r_node
    #r = self.GetSize()
    # Draw edges, arrows
    fArrow = 0.8
    for node in graph:
        for mode in node.edgo:
            x1, y1, x2, y2 = node.x, node.y, mode.x, mode.y
            line = lines.Line2D((x1,x2), (y1, y2), lw=2,
                color=(0,0,0), alpha=.5, zorder=0)
            ax.add_line(line)
            ax.add_patch(patches.FancyArrowPatch(
                (x1,y1),(x1+fArrow*(x2-x1), y1+fArrow*(y2-y1)),
                arrowstyle='->',mutation_scale=60,
                color=(0,0,0), alpha=.5, zorder=0))
        if node.x < xmin: xmin = node.x
        if node.x > xmax: xmax = node.x
        if node.y < ymin: ymin = node.y
        if node.y > ymax: ymax = node.y
    p = 0.8  # use 0.5 to show the whole node, anything higher is padding
    if limits[0] is None: limits[0] = xmin - p*w
    if limits[1] is None: limits[1] = xmax + p*w
    if limits[2] is None: limits[2] = ymin - p*w
    if limits[3] is None: limits[3] = ymax + p*w
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    # Draw nodes
    for node in graph:
        # Draw ellipse
        e = patches.Ellipse(xy=(node.x, node.y), width=w, height=w, angle=0,
            zorder=1)
        e.set_clip_box(ax.bbox)
        e.set_alpha(node.rgba[3])
        e.set_facecolor(node.rgba[0:3])
        ax.add_artist(e)
        # Draw label
        if node.label is not None:
            ax.text(node.x, node.y, node.label,
                horizontalalignment='center', zorder=2)
    # Return axes
    return ax
def create_component_dependency_graph(model, omit_states=True,
    omit_constants=False):
    """
    Creates and returns a component dependency graph.
    """
    # Create graph
    g = DiGraph()
    # Add nodes
    deps = model.map_component_dependencies(omit_states=omit_states,
        omit_constants=omit_constants)
    for c in deps:
        node = g.add_node(c)
        node.x, node.y = (random.random(), random.random())
        node.rgba = (random.random(), random.random(), random.random(), 0.9)
        node.label = c.name()
    # Add edges, get sorted dict of nodes sorted by #deps
    order = {}
    for c, cdeps in deps.iteritems():
        n = 0
        for d in cdeps:
            g.add_edge(c, d)
            n += 1
        if n not in order:
            order[n] = []
        order[n].append(c)
    # Layout
    try:
        g.layout_layered()
    except Exception:
        # Cyclical graph, create simple layout
        nOrder = len(order)
        p = 0.1
        dy = (1 - 2*p) / (nOrder - 1) if nOrder > 1 else 0
        y = p
        if False:
            dy = -dy
            y = 1 - y
        for key in sorted(order.iterkeys()):
            level = order[key]
            nLevel = len(level)
            dx = (1 - 2*p) / (nLevel - 1) if nLevel > 1 else 0
            x = 0.5 * (1 - dx * (nLevel - 1))
            for c in level:
                node = g[c]
                node.x, node.y = (x, y)
                x += dx
            y += dy
    return g
def plot_component_dependency_graph(model, axes=None, omit_states=True,
    omit_constants=False):
    """
    Draws a graph showing the dependencies between a model's components.

    Returns a matplotlib axes object.
    """
    return plot_digraph(create_component_dependency_graph(model,
        omit_states=omit_states, omit_constants=omit_constants), axes)
def create_variable_dependency_graph(model):
    """
    Creates a dependency graph from the given model

    (Doesn't include constants)
    """
    g = DiGraph()
    # Get shallow dependencies
    deps = model.map_shallow_dependencies(collapse=True, omit_states=True,
        omit_constants=True)
    # Create nodes
    for lhs, dps in deps.iteritems():
        # Create node
        node = g.add_node(lhs)
        # Set node label
        node.label = lhs.var().qname()
        if type(lhs) == myokit.Derivative:
            node.label = 'd(' + node.label + ')'
        # Set color
        node.rgba = (random.random(), random.random(), random.random(), 0.9)
    # Create edges
    used = set()
    for lhs, dps in deps.iteritems():
        for dep in dps:
            g.add_edge(lhs, dep)
            used.add(lhs)
            used.add(dep)
    # Remove unused nodes
    for node in set(g.nodes).difference(used):
        g.remove_node(node)
    del(used)
    # Layout graphs
    g.layout_layered()
    # Return
    return g
def plot_variable_dependency_graph(model, axes=None):
    """
    Draws a graph showing the dependencies between a model's variables.

    Returns a matplotlib axes object.
    """
    return plot_digraph(create_variable_dependency_graph(model), axes=axes,
        r_node=0.03)
