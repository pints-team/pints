#
# Parser for .mmt files
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
import re
import unicodedata
from collections import OrderedDict
import myokit
from myokit import ParseError, ProtocolParseError
def parse(source):
    """
    Parses strings in ``mmt`` format and returns a tuple ``(model,
    protocol, embedded script)``.

    ``None`` may be returned for any segment not appearing in the source.

    The source to parse can be given as a plain string, a sequence of lines
    or a stream of lines.
    """
    # Get raw stream
    raw = source
    if type(raw) in [str, unicode]:
        raw = raw.splitlines()
    try:
        raw.next
    except AttributeError:
        raw = iter(raw)
    # Create tokenizer
    stream = Tokenizer(raw)
    # Start parsing
    model = protocol = script = None
    # Get segments
    token = expect(stream.peek(), SEGMENT_HEADER)
    if token[1][2:-2] not in ['protocol', 'script']:
        model = parse_model_from_stream(stream)
        token = expect(stream.peek(), [SEGMENT_HEADER, EOF])
    if token[0] == SEGMENT_HEADER and token[1][2:-2] == 'protocol':
        protocol = parse_protocol_from_stream(stream)
        token = expect(stream.peek(), [SEGMENT_HEADER, EOF])
    if token[0] == SEGMENT_HEADER:
        if token[1][2:-2] != 'script':
            if protocol is None:
                raise ParseError('Invalid segment header', token[2], token[3],
                    'Expecting [[protocol]] or [[script]]')
            else:
                raise ParseError('Invalid segment header', token[2], token[3],
                    'Expecting [[script]]')
        script = parse_script_from_stream(stream, raw)
    else:
        expect(stream.next(), EOF)
    # Return
    return (model, protocol, script)
def parse_model(source):
    """
    Parses a model in ``mmt`` format.
    """
    # Get raw stream
    raw = source
    if type(raw) in [str, unicode]:
        raw = raw.splitlines()
    try:
        raw.next
    except AttributeError:
        raw = iter(raw)
    # Parse and return
    stream = Tokenizer(raw)
    token = expect(stream.peek(), SEGMENT_HEADER)
    if token[1] != '[[model]]':
        raise ParseError('Invalid segment header', token[2], token[3],
            'Expecting [[model]]')
    model = parse_model_from_stream(stream)
    expect(stream.next(), EOF)
    return model
def parse_protocol(source):
    """
    Parses a protocol in ``mmt`` format.
    """
    # Get raw stream
    raw = source
    if type(raw) in [str, unicode]:
        raw = raw.splitlines()
    try:
        raw.next
    except AttributeError:
        raw = iter(raw)
    # Parse and return
    stream = Tokenizer(raw)
    token = expect(stream.peek(), SEGMENT_HEADER)
    if token[1] != '[[protocol]]':
        raise ParseError('Invalid segment header', token[2], token[3],
            'Expecting [[protocol]]')
    protocol = parse_protocol_from_stream(stream)
    expect(stream.next(), EOF)
    return protocol
def parse_script(source):
    """
    Parses a script in ``mmt`` format.
    """
    # Get raw stream
    raw = source
    if type(raw) in [str, unicode]:
        raw = raw.splitlines()
    try:
        raw.next
    except AttributeError:
        raw = iter(raw)
    # Parse and return
    stream = Tokenizer(raw)
    token = expect(stream.peek(), SEGMENT_HEADER)
    if token[1] != '[[script]]':
        raise ParseError('Invalid segment header', token[2], token[3],
            'Expecting [[script]]')
    script = parse_script_from_stream(stream)
    expect(stream.next(), EOF)
    return script
def parse_state(state):
    """
    Parses an initial state declaration given in one of two formats::

        <var_name_1> = <value>
        <var_name_1> = <value>
        ...

    Blank lines, whitespace and indentation are ignored. All variable names
    must be given fully qualified. Parsed data from this format is returned as
    a dictionary mapping variable names to values.

    Alternatively, the input can be given as a list of floating point numbers
    separated by spaces, commas, semi-colons, line breaks etc::

        <value>
        <value>
        ...

    Parsed data from this input format is returned as a list of floating point
    numbers.
    """
    # Get raw stream
    raw = state
    if type(raw) in [str, unicode]:
        raw = raw.splitlines()
    try:
        raw.next
    except AttributeError:
        raw = iter(raw)
    # Create tokenizer
    stream = Tokenizer(raw, check_indenting=False)
    # Start parsing
    if stream.peek()[0] == NAME:
        # Parse name = value format
        pairs = {}
        while stream.peek()[0] != EOF:
            name = expect(stream.next(), NAME)[1]
            t = stream.peek()
            if t[0] != DOT:
                raise ParseError('Unexpected token, expecting "."', t[2], t[3],
                    'All variable names must be fully qualified: comp.var')
            expect(stream.next(), DOT)
            name += '.' + expect(stream.next(), NAME)[1]
            expect(stream.next(), EQUAL)
            pairs[name] = parse_expression_stream(stream).eval()
            t = expect(stream.next(), EOL)
        return pairs
    else:
        # Parse list-of-floats format
        expr = [EOF, INTEGER, FLOAT, PLUS, MINUS, PAREN_OPEN, FUNC_NAME]
        state = []
        while True:
            while stream.peek()[0] not in expr:
                stream.next()
            if stream.peek()[0] == EOF:
                return state
            state.append(parse_expression_stream(stream).eval())
def split(source):
    """
    Attempts to split the ``source`` into model, protocol and script segments.
    Any content before the [[model]] tag is discarded.
    """
    # Get raw stream
    raw = source
    if type(raw) in [str, unicode]:
        raw = raw.splitlines()
    try:
        raw.next
    except AttributeError:
        raw = iter(raw)
    segments = ['', '', '']
    prot_or_script = ('[[protocol]]','[[script]]')
    # Gather lines in stream
    lines = []
    for line in raw:
        lines.append(line)
    # Create glue string for joining up lines again
    glue = ''
    if lines:
        glue = '' if lines[0][-1] == '\n' else '\n'
    # Create stream of lines
    i = 0
    # Try parsing model
    try:
        stream = Tokenizer(iter(lines))
        while stream.peek()[0] == EOL: stream.next()
        token = stream.peek()
        if token[0] == SEGMENT_HEADER and token[1] == '[[model]]':
            parse_model_from_stream(stream, syntax_only=True)
    except ParseError:
        pass
    # Next up should be a protocol or script, anything else is junk that will
    #  be added to the model code
    if stream:
        i = stream.peek()[2] - 1
    for line in lines[i:]:
        if line.strip() in prot_or_script: break
        i += 1
    # Get model code
    segments[0] = glue.join(lines[:i])
    # Nothing found? Then everything is model code. Return
    if i == len(lines):
        return tuple(segments)
    # Chop off first part, continue hopeful parsing
    lines = lines[i:]
    # This will work because the next line is [[protocol]] or [[script]]
    stream = Tokenizer(iter(lines))
    # Try parsing protocol
    try:
        while stream.peek()[0] == EOL: stream.next()
        token = stream.peek()
        if token[0] == SEGMENT_HEADER and token[1] == '[[protocol]]':
            parse_protocol_from_stream(stream)
            token = stream.peek()
            line = token[2]-1 if token[0] == SEGMENT_HEADER else token[2]
            segments[1] = glue.join(lines[:line])
            lines = lines[line:]
            stream = Tokenizer(iter(lines))
    except ParseError as e:
        # Reasonable guess: the protocol extends at least until e.line
        # Try finding [[script]]
        i = e.line
        for line in lines[e.line:]:
            if line.strip() == '[[script]]':
                break
            i += 1
        segments[1] = glue.join(lines[:i])
        if i == len(lines):
            return tuple(segments)
        lines = lines[i:]
    # Remainder is script
    segments[2] = glue.join(lines)
    return tuple(segments)
def expect(token, expected):
    """
    Checks the type of the given ``token``. If the token type equals
    ``expected`` or is one of the types in the sequence ``expected``, the
    token is returned. If not, an ParseError is raised.
    """
    code, text, line, char = token
    # Expected token type can be integer or collection
    if type(expected) == int:
        expected = [expected]
    if code in expected:
        return token
    unexpected_token(token, expected)
def unexpected_token(token, expected):
    """
    Raises a formatted unexpected token error (ParseError).
    """
    code, text, line, char = token
    # Parse gotten token
    got  = token_map[code]
    hide = [EOL, EOF, INDENT, DEDENT, AND, OR, NOT]
    if not code in hide:
        got += ' "' + text + '"'
    # Parse expected token(s) or string
    if type(expected) != str:
        if len(expected) > 2:
            expected = 'one of [ ' \
                + ', '.join([token_str[i] for i in expected]) + ' ]'
        elif len(expected) == 2:
            expected = token_str[expected[0]] + ' or ' + token_str[expected[1]]
        else:
            expected = token_str[expected[0]]
    raise ParseError('Syntax error', line, char,
        'Unexpected token ' + got + ' expecting ' + expected)
def reg_token(info, token, obj):
    """
    Registers this token for the given object.

    Creates a variable _token in the object, pointing to the token
    Creates an entry in the model's token list, containing a tuple (token, obj)
    """
    # Register token with object
    obj._token = token
    # Register (token, obj) in model
    if info and info.model:
        if token[2] not in info.model._tokens:
            d = info.model._tokens[token[2]] = {}
        else:
            d = info.model._tokens[token[2]]
        d[token[3]] = (token, obj)
class ParseInfo(object):
    def __init__(self):
        self.model = None
        self.initial_values = OrderedDict()
        self.alias_map = {}
        self.user_functions = {}
def parse_model_from_stream(stream, syntax_only=False):
    """
    Parses a model.

    By setting the optional argument ``syntax_only`` to ``True`` the function
    can be used as a syntax check: In this case only the early stages of
    parsing are handled but variables aren't resolved, the model isn't
    validated etc. In this use-case, True is returned if the parsed text had no
    syntax errors.
    """
    # Create parse info object
    info = ParseInfo()
    # Parse header definition
    token = expect(stream.next(), SEGMENT_HEADER)
    if token[1][2:-2] != 'model':
        raise ParseError('Invalid segment header', token[2], token[3],
            'Expecting [[model]]')
    expect(stream.next(), EOL)
    # Create model
    model = info.model = myokit.Model()
    reg_token(info, token, model)
    # Parse header data
    token = stream.peek()
    while token[0] in (NAME, FUNC_NAME, META_NAME):
        if token[0] == FUNC_NAME:
            # User function
            parse_user_function(stream, info)
        elif token[0] == META_NAME:
            # Meta data
            t = stream.next()
            meta_key = t[1].strip()
            if meta_key in model.meta:
                raise ParseError('Duplicate meta-data property', t[2], t[3],
                    'The meta-data property "' + meta_key + '" was already'
                    ' specified for this model.')
            expect(stream.next(), COLON)
            next = expect(stream.next(), [TEXT, EOL])
            meta_val = ''
            if next[0] == TEXT:
                meta_val = next[1].strip()
                expect(stream.next(), EOL)
            model.meta[meta_key] = meta_val
        else:
            # Initial value
            t0 = stream.next()
            t1 = expect(stream.next(), DOT)
            t2 = expect(stream.next(), NAME)
            name = t0[1] + t1[1] + t2[1]
            if name in info.initial_values:
                raise ParseError('Duplicate initial value', t0[2], t0[3],
                    'A value for <' + name + '> was already specified.')
            expect(stream.next(), EQUAL)
            expr = parse_expression_stream(stream)
            expect(stream.next(), EOL)
            info.initial_values[name] = expr
            reg_token(info, t0, expr)
        token = stream.peek()
    # Save order of state variables
    state_order = info.initial_values.keys()
    # Parse components
    while stream.peek()[0] == BRACKET_OPEN:
        parse_component(stream, info)
    expect(stream.peek(), (EOF, BRACKET_OPEN, SEGMENT_HEADER))
    # Syntax checking mode
    if syntax_only:
        return True
    # All initial variables must have been used
    for qname, e in info.initial_values.iteritems():
        raise ParseError('Unused initial value', 0, 0,
            'An unused initial value was found for "' + str(qname) + '".')
    # Re-order the model state
    model.reorder_state(state_order)
    # Order encountered tokens
    m = model._tokens
    model._tokens = {}#OrderedDict()
    for line in sorted(m.iterkeys()):
        model._tokens[line] = {}#OrderedDict()
        for char in sorted(m[line].iterkeys()):
            model._tokens[line][char] = m[line][char]
    # Resolve alias map
    resolve_alias_map_names(info)
    # Resolve variable references to objects
    # A.K.A. Parse proto expression into Myokit.Expression
    for var in model.variables(deep=True):
        if var._proto_rhs is not None:
            # No need to check for IllegalReferenceErrors here, since these
            # won't have been resolved in the first place.
            var.set_rhs(convert_proto_expression(var._proto_rhs, var, info))
        del(var._proto_rhs)
    # Check the semantics of the model
    try:
        model.validate()
    except myokit.IntegrityError as e:
        t = e.token()
        if t:
            raise ParseError('IntegrityError', t[2], t[3], e.message,
                cause=e)
        raise ParseError('IntegrityError', 0, 0, e.message, cause=e)
    # Return
    return model
def parse_user_function(stream, info):
    """
    Parses a user function.
    """
    # Parse name
    token, name, line, char = expect(stream.next(), FUNC_NAME)
    # Parse argument list
    args = []
    expect(stream.next(), PAREN_OPEN)
    token = expect(stream.next(), [PAREN_CLOSE, NAME])
    while token[0] == NAME:
        args.append(token[1])
        token = expect(stream.next(), [COMMA, PAREN_CLOSE])
        if token[0] == COMMA:
            token = expect(stream.next(), NAME)
    # Parse template
    expect(stream.next(), EQUAL)
    expr = convert_proto_expression(parse_proto_expression(stream, info))
    expect(stream.next(), EOL)
    # Create user function
    try:
        info.model.add_function(name, args, expr)
    except (myokit.DuplicateFunctionName ,
            myokit.DuplicateFunctionArgument,
            myokit.InvalidNameError,
            myokit.InvalidFunction) as e:
        raise ParseError('Invalid function declaration', line, char,
            e.message, cause=e)
def parse_component(stream, info=None):
    """
    Parses a component
    """
    if info is None or info.model is None:
        raise Exception('parse_component requires at least a model to be'
            ' present in the ParseInfo.')
    # Parse component declaration
    expect(stream.next(), BRACKET_OPEN)
    token = expect(stream.next(), NAME)
    expect(stream.next(), BRACKET_CLOSE)
    expect(stream.next(), EOL)
    code, name, line, char = token
    try:
        component = info.model.add_component(name)
    except myokit.DuplicateName as e1:
        raise ParseError('Duplicate component name', line, char, e1.message,
            cause=e1)
    except myokit.InvalidNameError as e2:
        raise ParseError('Illegal component name', line, char, e2.message,
            cause=e2)
    reg_token(info, token, component)
    # Add alias map
    info.alias_map[component] = {}
    # Parse fields
    token = stream.peek()
    while token[0] not in [BRACKET_OPEN, SEGMENT_HEADER, EOF]:
        if token[0] == USE:
            # Alias
            parse_alias(stream, info, component)
        elif token[0] == META_NAME:
            # Meta data
            t = stream.next()
            meta_key = t[1].strip()
            if meta_key in component.meta:
                raise ParseError('Duplicate meta-data property', t[2], t[3],
                    'The meta-data property "' + meta_key + '" was already'
                    ' specified for this component.')
            expect(stream.next(), COLON)
            next = expect(stream.next(), [TEXT, EOL])
            meta_val = ''
            if next[0] == TEXT:
                meta_val = next[1].strip()
                expect(stream.next(), EOL)
            component.meta[meta_key] = meta_val
        else:
            # Variable
            parse_variable(stream, info, component)
        token = stream.peek()
def parse_alias(stream, info, component):
    """
    Parses an alias definition
    ::

        alias_map = {
            comp1 : {
                alias_as_string : (use_token, comp_token, var_token),

    """
    # Ensure valid parse info and alias map
    if info is None:
        info = ParseInfo()
    try:
        amap = info.alias_map[component]
    except KeyError:
        amap = info.alias_map[component] = {}
    # Parse
    token = expect(stream.next(), [USE])
    while True:
        # Get referenced variable
        token_c = expect(stream.next(), NAME)   # component
        token_v = stream.next()                 # dot
        if token_v[0] != DOT:
            raise ParseError('Invalid reference', token_v[2], token_v[3],
                'Aliassed variables must be specified using their fully'
                ' qualified name (component.variable)')
        token_v = expect(stream.next(), NAME)   # variable
        # Get reference name
        if stream.peek()[0] == AS:
            expect(stream.next(), AS)
            code, name, line, char = expect(stream.next(), NAME)
        else:
            code, name, line, char = token_v
        # Create alias
        amap[name] = (token, token_c, token_v)
        # Expecting end of line or next alias
        next = expect(stream.next(), [COMMA, EOL])
        if next[0] == EOL: break
def parse_variable(stream, info, parent, convert_proto_rhs=False):
    """
    Parses a variable.
    
    In normal operation (when parsing models) the variable's expression is
    stored in Variable._proto_rhs until the full model has been parsed. This
    allows the rhs to be set with a fully resolved expression. For debugging
    purposes, it might be necessary to convert the proto_rhs immediatly. To do
    so, set ``convert_proto_rhs`` to ``True``.
    """
    # Ensure valid parse info and alias map
    if info is None:
        info = ParseInfo()
    try:
        amap = info.alias_map[parent]
    except KeyError:
        amap = info.alias_map[parent] = {}
    # List of tokens to register with the final variable
    toreg = []
    # Parse variable declaration
    token = expect(stream.next(), [NAME, FUNC_NAME])
    toreg.append(token)
    code, name, line, char = token
    # Allow dot() function on lhs
    if code == FUNC_NAME:
        if name != 'dot':
            raise ParseError('Illegal lhs', line, char, 'Only variable names'
                ' or the dot() function may appear on the left-hand side of an'
                ' equation.')
        if type(parent) != myokit.Component:
            raise ParseError('Illegal variable declaration', line, char,
                'State variable declarations may not be nested.')
        toreg.append(expect(stream.next(), PAREN_OPEN))
        token = expect(stream.next(), NAME)
        toreg.append(expect(stream.next(), PAREN_CLOSE))
        toreg.append(token)
        code, name, line, char = token
        is_state = True
    else:
        is_state = False
    # Check if name is already in use for an alias
    component = parent
    if not isinstance(component, myokit.Component):
        component = component.parent(myokit.Component)
    # Create variable
    try:
        var = parent.add_variable(name)
    except myokit.DuplicateName as e1:
        raise ParseError('Duplicate variable name', line, char, e1.message,
            cause=e1)
    except myokit.InvalidNameError as e2:
        raise ParseError('Illegal variable name', line, char, e2.message,
            cause=e2)
    # Register tokens
    for token in toreg:
        reg_token(info, token, var)
    del(toreg)
    # Set initial value for states
    if is_state:
        if not var.qname() in info.initial_values:
            raise ParseError('Missing initial value', line, char,
                'No initial value was found for "' + var.qname() + '"')
        state_value = info.initial_values[var.qname()]
        try:
            var.promote(state_value)
        except myokit.NonLiteralValueError as e:
            t = state_value._token
            raise ParseError('Illegal state value', t[2],t[3], e.message,
                cause=e)
        del(info.initial_values[var.qname()])
    # Parse definition, quick unit, bind, label and description syntax
    # These token must occur in a fixed order!
    unit = None
    token = expect(stream.peek(), (EQUAL, IN, BIND, LABEL, COLON, EOL))
    if token[0] == EQUAL:
        # Parse variable definition
        stream.next()
        # Save proto-expression for right-hand side
        var._proto_rhs = parse_proto_expression(stream, info)
        # Get rest of line
        token = expect(stream.peek(), (IN, BIND, LABEL, COLON, EOL))
    else:
        # No right hand side set!
        var._proto_rhs = None
    if token[0] == IN:
        # Parse variable unit
        stream.next()
        unit = parse_bracketed_unit(stream)
        token = expect(stream.peek(), (BIND, LABEL, COLON, EOL))
    if token[0] == BIND:
        # Parse bind
        parse_binding(stream, info, var)
        token = expect(stream.peek(), (LABEL, COLON, EOL))
    if token[0] == LABEL:
        # Parse label
        parse_label(stream, info, var)
        token = expect(stream.peek(), (COLON, EOL))        
    if token[0] == COLON:
        # Colon found, rest of line is description
        stream.next()
        desc = expect(stream.next(), TEXT)[1].strip()
        var.meta['desc'] = desc
    expect(stream.next(), EOL)
    # Parse indented fields (temp vars, meta or unit)
    if stream.peek()[0] == INDENT:
        stream.next()
        while stream.peek()[0] != DEDENT:
            token = expect(stream.peek(), [NAME, META_NAME, IN, BIND, LABEL])
            code, name, line, char = token
            if token[0] == NAME:
                # Nested variable
                parse_variable(stream, info, var)
            elif token[0] == META_NAME:
                # Meta property
                meta_key = name.strip()
                if meta_key in var.meta:
                    raise ParseError('Duplicate meta-data property', line,
                        char, 'The meta-data property "' + meta_key + '" was'
                        ' already specified for this variable.')
                stream.next()
                expect(stream.next(), COLON)
                next = expect(stream.next(), [TEXT, EOL])
                meta_val = ''
                if next[0] == TEXT:
                    meta_val = next[1].strip()
                    expect(stream.next(), EOL)
                var.meta[meta_key] = meta_val
            elif token[0] == IN:
                if unit:
                    raise ParseError('Duplicate variable unit', line, char,
                        'Unit already specified for this variable.')
                # Variable unit
                stream.next()
                unit = parse_bracketed_unit(stream)
                expect(stream.next(), EOL)
            elif token[0] == BIND:
                # Binding to external value
                parse_binding(stream, info, var)
                expect(stream.next(), EOL)
            elif token[0] == LABEL:
                # Labelled as unique value
                parse_label(stream, info, var)
                expect(stream.next(), EOL)
            else:
                raise Exception('Unhandled case.')
            # Next line
            token = stream.peek()[0]
        expect(stream.next(), DEDENT)
    # Set unit
    var.set_unit(unit)
    # Normal operation is to leave the ._proto_rhs untouched until the full
    # model has been parsed and variables have been resolved. For debugging, it
    # may be nice to resolve at this point already.
    if convert_proto_rhs:
        var.set_rhs(convert_proto_expression(var._proto_rhs))
        del(var._proto_rhs)
def parse_binding(stream, info, var):
    """
    Parses the "bind" part of a variable definition.
    """
    token, name, line, char = expect(stream.next(), BIND)
    # Get binding label
    label = expect(stream.next(), NAME)[1]
    # Bind variable
    try:
        var.set_binding(label)
    except myokit.InvalidBindingError as e:
        raise ParseError('Illegal binding', line, char, e.message, cause=e)
def parse_label(stream, info, var):
    """
    Parses the "label" part of a variable definition.
    """
    token, name, line, char = expect(stream.next(), LABEL)
    # Get label
    label = expect(stream.next(), NAME)[1]
    # Register label
    try:
        var.set_label(label)
    except myokit.InvalidLabelError as e:
        raise ParseError('Illegal label', line, char, e.message, cause=e)
def resolve_alias_map_names(info):
    """
    Resolves the variables pointed to by the alias dictionary

    Updates the alias map in each component from a string:string dictionary to
    a string:variable dictionary and checks if
     - The variables are in another component
     - No two aliasses are tied to the same variable (within a single
      component)
    """
    for comp, amap in info.alias_map.iteritems():
        for name, (t_use, t_comp, t_var) in amap.iteritems():
            var_name = t_comp[1] + '.' + t_var[1]
            (var, sug, msg) = info.model.suggest_variable(var_name)
            if var is None:
                raise ParseError('Variable not found', t_comp[2], t_comp[3],
                    msg)
            try:
                comp.add_alias(name, var)
            except myokit.DuplicateName as e:
                raise ParseError('Duplicate name error', t_comp[2], t_comp[3],
                    e.message, cause=e)
            except myokit.InvalidNameError as e2:
                raise ParseError('Illegal alias name', t_comp[2], t_comp[3],
                    e2.message, cause=e)
def parse_unit(stream):
    """
    Parses a unit expression and returns a Unit object.
    """
    # Parse first unit in unit expression
    token = expect(stream.next(), [NAME, INTEGER])
    if token[0] == NAME:
        try:
            unit = myokit.Unit.parse_simple(token[1])
        except KeyError as ke:
            raise ParseError('Unit not recognized', token[2], token[3],
                ke.message, cause=ke)
        if stream.peek()[0] == POWER:
            stream.next()
            expo = expect(stream.next(), (INTEGER, FLOAT, MINUS))[1]
            if expo == '-':
                unit **= -float(expect(stream.next(), (INTEGER, FLOAT))[1])
            else:
                unit **= float(expo)
    else:
        x = float(token[1])
        if x != 1.0:
            raise ParseError('Invalid unit specification', token[2], token[3],
                'Unit specification must start with unit name or "1"')
        unit = myokit.Unit.parse_simple('1')
    # Parse remaining units (* or /)
    while stream.peek()[0] in [STAR, SLASH]:
        op = stream.next()[0]
        token = expect(stream.next(), NAME)
        try:
            part = myokit.Unit.parse_simple(token[1])
        except KeyError as ke:
            raise ParseError('Unit not recognized', token[2], token[3],
                ke.message, cause=ke)
        if stream.peek()[0] == POWER:
            stream.next()
            expo = expect(stream.next(), (INTEGER, FLOAT, MINUS))[1]
            if expo == '-':
                part **= -float(expect(stream.next(), (INTEGER, FLOAT))[1])
            else:
                part **= float(expo)
        if op == STAR:
            unit *= part
        else:
            unit /= part
    # Parse multiplier
    if stream.peek()[0] == PAREN_OPEN:
        stream.next()
        token = stream.peek()
        e = parse_expression_stream(stream)
        if not e.is_literal():
            raise ParseError('Invalid unit multiplier', token[2], token[3],
                'Unit multipliers cannot contain variables.')
        unit *= e.eval()
        expect(stream.next(), PAREN_CLOSE)
    return unit
def parse_unit_string(string):
    """
    Parses string data into a :class:`myokit.Unit`.
    """
    s = Tokenizer(string)
    e = parse_unit(s)
    expect(s.next(), EOL)
    expect(s.next(), EOF)
    try:
        s.next()
        raise ParseError('Unused tokens', 0, 0, 'Expecting a string containing'
            ' only a single unit expression.')
    except StopIteration:
        return e
def parse_bracketed_unit(stream):
    """
    Parses a unit wrapped in brackets, catches the unit text and adds a fake
    UNIT token to the unit, allowing later maniplation.
    """
    unit_text = stream.start_catching()
    token = expect(stream.next(), BRACKET_OPEN)
    unit = parse_unit(stream)
    expect(stream.next(), BRACKET_CLOSE)
    unit_text = stream.stop_catching(unit_text)
    unit._token = (UNIT, unit_text, token[2], token[3])
    return unit
def parse_protocol_from_stream(stream):
    """
    Parses a protocol segment, requires a Tokenstream
    """
    # Number parsing function
    def parse_number(stream):
        t = stream.peek()
        e = parse_expression_stream(stream)
        if not e.is_literal():
            raise ProtocolParseError('Invalid expression', t[2], t[3],
                'Protocol expressions cannot contain variables.')
        return e.eval()
    # Check segment header
    if stream.peek()[0] == SEGMENT_HEADER:
        token = stream.next()
        if token[1][2:-2] != 'protocol':
            raise ParseError('Invalid segment header', token[2], token[3],
                'Expecting [[protocol]]')
        expect(stream.next(), EOL)
    # Create protocol
    protocol = myokit.Protocol()
    # Parse lines
    t_last = None
    t_next = 0
    n = stream.peek()
    while(n[0] not in (EOF, SEGMENT_HEADER)):
        # Parse level
        v = parse_number(stream)
        # Parse starting time
        # Allow 'next' to mean "after the previous event ends"
        if stream.peek()[1] == 'next':
            stream.next()
            if t_next is None:
                raise ProtocolParseError('Invalid next', n[2], n[3],
                    'Unable to determine end of previous event, "next" cannot'
                    ' be used here.')
            else:
                t = t_next
        else:
            t = parse_number(stream)
        if t_last is None:
            t_last == t
        else:
            if t > t_last:
                t_last = t
            elif t == t_last:
                raise ProtocolParseError('Simultaneous stimuli', n[2],
                    n[3], 'Stimuli may not occur at the same time')
            else:
                raise ProtocolParseError('Non-consecutive stimuli', n[2], 
                    n[3], 'Stimuli must be listed in chronological order')
        # Parse duration
        d = parse_number(stream)
        if d <= 0:
            raise ProtocolParseError('Non-positive duration', n[2], n[3],
                'The duration of a stimulus must be strictly positive')
        # Parse period
        p = parse_number(stream)
        if p < 0:
            raise ProtocolParseError('Negative period', n[2], n[3],
                'Stimuli cannot occur with a negative period')
        # Parse multiplier
        r = int(parse_number(stream))
        if r < 0:
            raise ProtocolParseError('Negative multiplier', n[2], n[3],
                'Stimulus cannot occur a negative number of times')
        elif r > 0 and p == 0:
            raise ProtocolParseError('Invalid multiplier', n[2], n[3],
                'Non-periodic event cannot occur more than once')
        # Determine next event end
        if p == 0:
            t_next = t + d
        else:
            t_next = None
        # Parse end of line
        expect(stream.next(), EOL)
        # Schedule event
        try:
            protocol.schedule(v, t, d, p, r)
        except myokit.ProtocolEventError as e:
            raise ProtocolParseError('Invalid event specification', n[2], 0,
                e.message)
        n = stream.peek()
    return protocol
def parse_script_from_stream(stream, raw_stream):
    """
    Parses a script segment
    """
    token = expect(stream.next(), SEGMENT_HEADER)
    if token[1][2:-2] != 'script':
        raise ParseError('Invalid segment header', token[2], token[3],
            'Expecting [[script]]')
    token = expect(stream.peek(), EOL)
    raw = []
    for line in raw_stream:
        raw.append(line)
    return ''.join(raw)
def strip_expression_units(model_text, skip_literals=True):
    """
    Takes the text of a valid model as input and returns a version stripped of
    any expression units. Variable units defined with ``in`` are preserved.
    Only the model part should be passed in, no script or protocol segments.
    
    By default, constants defined for variables whose RHS is a single number
    will keep their units. To disable this, set ``skip_literals=False``.
    
    This method will raise a :class:`myokit.ParseError` if the given code
    cannot be parsed to a valid model.
    """
    if type(model_text) in (str, unicode):
        lines = model_text.splitlines()
    else:
        lines = model_text
    stream = Tokenizer(lines)
    m = parse_model_from_stream(stream)
    # Collect positions of fake unit tokens
    cuts = []
    toks = []
    for var in m.variables(deep=True):
        rhs = var.rhs()
        if skip_literals and type(rhs) == myokit.Number:
            continue
        for e in rhs.walk(allowed_types=myokit.Number):
            u = e.unit()
            if u is not None:
                if not u._token:
                    raise Exception('Token not set for number.')
                token, text, line, char = u._token
                toks.append(u._token)
                # Lines start at 1, chars start at 0...
                cuts.append((line-1, char, char + len(text)))
    # Create stripped version of lines
    if cuts:
        stripped = []
        cuts.sort()
        cuts = iter(cuts)
        cut = cuts.next() 
        for k, line in enumerate(lines):
            if cut and cut[0] == k:
                # Gather non-unit parts of line
                line2 = ''
                offset = 0
                while cut and cut[0] == k:
                    x, char1, char2 = cut
                    # Adjust spaces
                    if line[char1-1:char1] == ' ':
                        # Remove leading space if:
                        #  - unit is followed by a space: 1 [ms] * 10
                        #  - unit is followed by a bracket: (1 + 1 [ms])
                        #  - unit is at end of line: 1 [ms]
                        if line[char2:char2+1] in (' ', ')', ''):
                            char1 -= 1
                    line2 += line[offset:char1]
                    offset = char2
                    try:
                        cut = cuts.next()
                    except StopIteration:
                        cut = None
                line = line2 + line[offset:]
            stripped.append(line)
        lines = stripped
    return '\n'.join(lines)
# Tokens
token_map = {}
token_str = {}
def add_token(name, nice):
    """
    Defines a token in the global namespace of this module

    Arguments:
    name
        The variable name
    nice
        A human-readable version of the variable name
    """
    code = 10 + len(token_map)
    globals()[name] = code
    token_map[code] = name
    token_str[code] = nice
add_token('EOL', 'End of line')
add_token('EOF', 'End of file')
add_token('INDENT', 'Indentation increase')
add_token('DEDENT', 'Indentation decrease')
add_token('WHITESPACE', 'Whitespace between tokens')
add_token('NAME', 'Name (function, variable etc)')
add_token('INTEGER', 'Integer')
add_token('FLOAT', 'Float')
add_token('TEXT', 'Text')
add_token('FUNC_NAME', 'Function name')
add_token('META_NAME', 'Meta property name')
add_token('PAREN_OPEN', 'Left parentheses "("')
add_token('PAREN_CLOSE', 'Right parentheses ")"')
add_token('BRACKET_OPEN', 'Left bracket "["')
add_token('BRACKET_CLOSE', 'Right bracket "]"')
add_token('SEGMENT_HEADER', 'Segment header "[[segment_name]]"')
add_token('EQUAL', 'Assignment "="')
add_token('COLON', 'Colon ":"')
add_token('DOT', 'Dot "."')
add_token('COMMA', 'Comma ","')
add_token('PLUS', 'Plus "+"')
add_token('MINUS', 'Minus "-"')
add_token('STAR', 'Star "*"')
add_token('SLASH', 'Slash "/"')
add_token('POWER', 'Power "^"')
add_token('REMAINDER', 'Remainder "%"')
add_token('QUOTIENT', 'Quotient "//"')
add_token('EQEQUAL', 'Equality "=="')
add_token('NOTEQUAL', 'Inequality "!="')
add_token('MORE', 'More ">"')
add_token('LESS', 'Less "<"')
add_token('MOREEQUAL', 'More than or equal ">="')
add_token('LESSEQUAL', 'Less than or equal "<="')
add_token('AND', 'and')
add_token('OR', 'or')
add_token('NOT', 'not')
add_token('IN', 'in')       # Unit
add_token('USE', 'use')     # Alias
add_token('AS', 'as')       # Alias
add_token('BIND', 'bind')   # External value
add_token('LABEL', 'label') # Special value
add_token('UNIT', 'unit')   # Fake token, used to given Units a useful token.
# Reserved keywords
KEYWORD_MAP = {
    'and'   : AND,
    'or'    : OR,
    'not'   : NOT,
    'in'    : IN,
    'use'   : USE,
    'as'    : AS,
    'bind'  : BIND,
    'label' : LABEL,
    }
KEYWORDS = KEYWORD_MAP.keys()
# Tabsize is used to interpret the column a tab will move you to, this affects
#  the indenting rules
tabsize = 8
# Define token recognising regex
# The syntax r'' is used for raw strings
_rTOKEN = re.compile('|'.join([
    # Whitespace
    r'[ \f\t]+',
    # Function opening (must come before name in list)
    r'[a-zA-Z]\w*\(',
    # Meta property name (must come before name in list)
    r'([a-zA-Z]\w*:)+',
    # Names
    r'[a-zA-Z]\w*',
    # Floating point numbers
    r'(([0-9]*\.[0-9]+)|([0-9]+\.?[0-9]*))([eE][+-]?[0-9]+)?',
    # Integers
    r'[0-9]+',
    # Comparison 1
    r'[<>!=]=',
    # Comparison 2
    r'[<>]',
    # Quotient and remainder
    r'[/]{2}|[%]',
    # Operators, * must come first (regex reasons...)
    r'[*+-/^]',
    # Segments
    r'\[\[[a-zA-Z]{1}[a-zA-Z0-9_]*\]\]',
    # Parentheses, brackets
    r'[()[\]{}]',
    # Assignment
    r"[=:]",
    # Dot, comma
    r'[.,]',
    # Line joining backslash
    r'[\\]',
    ]))
_sWHITE = ' \f\t'
_rWHITE = re.compile(r'[ \f\t]*')
_rSPACE = re.compile(r'[ \f\t]{1}')
# Recognizable characters
_sEOL = '\n'
_sNUMBERS = '0123456789'
# Tell floats from integers
_rFloat = re.compile(r'[.eE]+')
# Map of single characters
_SINGLE = "[]<>=.,+-*/^%"
_SINGLE_MAP = [
    BRACKET_OPEN, BRACKET_CLOSE,
    LESS, MORE, EQUAL, DOT, COMMA,
    PLUS, MINUS, STAR, SLASH, POWER, REMAINDER]
# Map of equals-comparators (==, !=, >=, <=)
_COMPEQ = '=!><'
_COMPEQ_MAP = [EQEQUAL, NOTEQUAL, MOREEQUAL, LESSEQUAL]
# Textual operators
_TEXT_OP = {'and':AND, 'or':OR, 'not':NOT}
class Tokenizer:
    """
    Takes a stream of lines as input and provides a stream interface returning
    tokens.

    Strips comments and concatenates lines with open parentheses (anything
    between round brackets is counted as a single line). Blank lines (or lines
    with nothing but comments) are ignored. Leading whitespace is converted
    to INDENT and DEDENT tokens, other whitespace is ignored.

    The tokens are returned as a 4-tuple with the following members:

    * The token type
    * The string parsed into a token
    * The line number the token was found on
    * The position in the line the token was found at

    The current token can be obtained using :meth:`current()`, the tokenizer
    can be advanced to the next token with :meth:`next()` and a preview of the
    next token can be obtained using :meth:`peek()`. In other words; ``peek()``
    and ``next()`` return the same token but ``next()`` also advances the
    stream while ``peek()`` does not.

    For memory efficiency, the given stream is read one line at a time and no
    reference to the read lines or returned tokens is kept. To obtain a list of
    all tokens returned between time A and B, the "catcher" mechanism can be
    used. For details, see :meth:`start_catching()`.

    For files that are to be parsed without indenting rules, the optional
    argument ``check_indenting`` can be set to False. In this mode, the
    tokenizer won't yield ``INDENT`` or ``DEDENT`` tokens.
    """
    def __init__(self, stream_of_lines, check_indenting=True):
        # Set next value and peek value
        self._next = None
        self._peek = None
        # At end of stream?
        self._has_last_value = False
        # Catchers and catcher handle index
        self._catchers = {}
        self._catcheri = 0
        # String given instead of stream of lines? Convert
        if type(stream_of_lines) in (str, unicode):
            stream_of_lines = iter(stream_of_lines.splitlines())
        # Create tokenizer
        self._tokenizer = self._tizer(stream_of_lines, check_indenting)
        # Grab first token
        self._advance()
    def __iter__(self):
        return self
    def _advance(self):
        """
        Advances to the next token.
        """
        self._next = self._peek
        for c in self._catchers.itervalues():
            c.append(self._next[1])
        try:
            self._peek = self._tokenizer.next()
            while self._peek[0] == WHITESPACE:
                for c in self._catchers.itervalues():
                    c.append(self._peek[1])
                self._peek = self._tokenizer.next()
        except StopIteration:
            self._has_last_value = True
    def current(self):
        """
        Returns the current token.
        """
        return self._next
    def next(self):
        """
        Takes the next token from the stream and returns it.
        """
        if self._has_last_value:
            raise StopIteration
        self._advance()
        return self._next
    def peek(self):
        """
        Peeks ahead and returns the next token it sees.
        """
        if self._has_last_value:
            raise IndexError
        return self._peek
    def start_catching(self):
        """
        Creates a new buffer that catches the string parts of each token.
        The method returns a handle used to identify this buffer.
        """
        self._catcheri += 1
        self._catchers[self._catcheri] = []
        return self._catcheri
    def stop_catching(self, handle):
        """
        Closes the buffer with the given handle and returns the caught
        input.
        """
        txt = ''.join(self._catchers[handle])
        del(self._catchers[handle])
        return txt
    def _tizer(self, stream, check_indenting):
        # All columns are stored to determine the level of indenting
        #  (This may require some extra checks...)
        columns = []
        # Indenting
        dents = 0
        # Parentheses must line up, parentheses can join lines together
        bracket_depth = 0
        bracket_lines = []
        # Lines can be appended with () or \, or """ for meta-values
        append_next_line = False
        in_multi_string = False
        # Block comments can be made by starting a line with """
        in_block_comment = False
        # Comment was found on the current line
        comment = False
        # TEXT also follows the \ rule
        text_buffer = False
        text_start = None
        # Loop through lines
        numb = 0
        ln = 0
        for line in stream:
            # Unicode conversion
            numb += 1
            if type(line) != str:
                line = unicodedata.normalize('NFKD', line).encode(
                    'ascii','ignore')
            # Handle multi-line meta-property strings
            if in_multi_string:
                line = line.rstrip()
                p = line.find('"""')
                if p < 0:
                    text_buffer.append(line)
                    continue
                elif p != len(line) - 3:
                    raise ParseError('Unexpected character(s)',
                        numb, p + 3, 'Text found after closing of multi-line'
                        ' string')
                text_buffer.append(line[0:p])
                text = self._post_process_multiline_string(text_buffer)
                yield TEXT, text, text_start[0], text_start[1]
                yield EOL, _sEOL, numb, len(line)
                in_multi_string = False
                text_buffer = False
                continue
            # Handle block comments
            if in_block_comment:
                p = line.find('"""')
                if p < 0:
                    continue
                else:
                    line = line[p+3:]
                    in_block_comment = False
            elif line[0:3] == '"""':
                #line = line.rstrip()
                p = line.find('"""', 3)
                if p < 0:
                    in_block_comment = True
                    continue
                else:
                    line = line[p+3:]
            # Ordinary lines, strip comments
            z = line.find('#')
            comment = (z >= 0)
            if comment:
                line = line[0:z]
            # Trim whitespace from end of string
            line = line.rstrip()
            ln = len(line)
            # Skip empty lines
            if ln == 0:
                if append_next_line:
                    # Handle multi-line situation
                    if bracket_depth > 0:
                        # Continue until bracket is closed
                        continue
                    if comment:
                        # Line with comment only doesn't end multi-line
                        # situation!
                        continue
                    if text_buffer:
                        # Multiline TEXT ends on blank line: yield!
                        yield TEXT, ''.join(text_buffer), text_start[0], \
                            text_start[1]
                        text_buffer = False
                    yield EOL, _sEOL, numb, ln
                    append_next_line = False
                continue
            # Index of character in character array
            pos = 0
            # Position in line (tab = 8 columns)
            column = 0
            # Initial whitespace is indenting
            countColumns = True
            # Append this line to previous?
            if append_next_line:
                # Skip whitespace
                m = _rWHITE.match(line)
                if m:
                    pos = m.end()
                countColumns = False
                if text_buffer:
                    if line[-1] == '\\':
                        text_buffer.append(line[pos:-1])
                    else:
                        text_buffer.append(line[pos:])
                        yield TEXT, ''.join(text_buffer), text_start[0], \
                            text_start[1]
                        yield EOL, _sEOL, numb, ln
                        append_next_line = False
                        text_buffer = False
                    continue
            append_next_line = False
            # Loop over token matches
            while pos < ln:
                m = _rTOKEN.match(line, pos)
                if m:
                    # Get token, token bounds, first character, move pointer
                    char = line[pos]
                    start, end = m.span()
                    token = line[start:end]
                    size = end - start
                    pos = end
                    # New line? convert whitespace to indent/dedent
                    if countColumns and check_indenting:
                        if char in _sWHITE:
                            for char in token:
                                if char == ' ':
                                    column += 1
                                elif char == '\t':
                                    column = (1 + column // tabsize) * tabsize
                                elif char == '\f':
                                    # Form-feed aka page-down, used by emacs,
                                    #  apparently
                                    column = 0
                            continue
                        else:

                            if not columns:
                                # First non-whitespace of new file
                                columns.append(column)
                            else:
                                # Differing whitespace?
                                if column > columns[-1]:
                                    # Yield indent token
                                    yield INDENT, line[0:start], numb, start
                                    columns.append(column)
                                    dents += 1
                                elif column < columns[-1]:
                                    # Yield dedent token(s)
                                    n = len(columns)
                                    while column < columns[-1]:
                                        yield DEDENT, line[0:start], numb,start
                                        dents -= 1
                                        columns.pop()
                                    if column != columns[-1]:
                                        raise ParseError(
                                            'Unexpected indenting level', numb,
                                            start, 'Indenting doesn\'t match'
                                            ' with previous level')
                            # Finished counting columns for this line
                            countColumns = False
                    # Ignore whitespace between tokens
                    if char == ' ' or char == '\t' or char == '\f':
                        yield WHITESPACE, token, numb, start
                        continue
                    if size == 1:
                        if char == ':':
                            # Colon? Treat remainder of line as text
                            yield COLON, token, numb, start
                            m = _rWHITE.match(line, end)
                            if m:
                                end = m.end()
                            # Get text, strip whitespace from right
                            text = line[end:].rstrip()
                            if text[0:3] == '"""':
                                # Triple quoted string
                                text = text[3:]
                                end += 3
                                p = text.find('"""')
                                if p < 0:
                                    # Multi-line string
                                    text_buffer = [text]
                                    text_start = numb, end
                                    in_multi_string = True
                                else:
                                    # Single-line string
                                    text = text[:-3]
                                    if p != len(text):
                                        raise ParseError('Unexpected'
                                            ' character(s)', numb, end + p + 3,
                                            'Text found after closing of'
                                            ' multi-line string')
                                    text = text.rstrip()
                                    yield TEXT, text, numb, end
                            elif text[-1:] == '\\':
                                # Multi-line string
                                text_buffer = [text[0:-1]]
                                text_start = numb, end
                                append_next_line = True
                            elif text != '':
                                # Single-line string
                                yield TEXT, text, numb, end
                            pos = ln
                            break
                        elif char == '\\':
                            # Append next line to this one
                            if end == ln:
                                append_next_line = True
                                continue
                            else:
                                raise ParseError('Illegal character',
                                    numb, start,
                                    'Backslash must be last character in line')
                        elif char == '(':
                            bracket_lines.append((numb, start))
                            bracket_depth += 1
                            yield PAREN_OPEN, token, numb, start
                            continue
                        elif char == ')':
                            bracket_depth -= 1
                            if bracket_depth < 0:
                                raise ParseError('Parentheses mismatch',
                                    numb, start,
                                    'Closing bracket without opening bracket')
                            bracket_lines.pop()
                            yield PAREN_CLOSE, token, numb, start
                            continue
                        else:
                            # Known single character token?
                            index = _SINGLE.find(char)
                            if index >= 0:
                                yield _SINGLE_MAP[index], token, numb, start
                                continue
                    elif size == 2:
                        if token[1] == '=':
                            # Comparison with ?=
                            index = _COMPEQ.find(char)
                            if index >= 0:
                                yield _COMPEQ_MAP[index], token, numb, start
                                continue
                        if token == '//':
                            # Quotient (Integer division)
                            yield QUOTIENT, token, numb, start
                            continue
                    if char == '[':
                        # Segment
                        yield SEGMENT_HEADER, token, numb, start
                        continue
                    if char in _sNUMBERS:
                        # Integer or float
                        m = _rFloat.search(token)
                        if m:
                            yield FLOAT, token, numb, start
                        else:
                            yield INTEGER, token, numb, start
                        continue
                    if char == '.' and size > 1:
                        # Float
                        yield FLOAT, token, numb, start
                        continue
                    if token[-1] == '(':
                        # Function opening
                        # Yield function name, then back up to yield PAREN_OPEN
                        # on next pass
                        fnc = token[:-1]
                        if fnc in _TEXT_OP:
                            yield _TEXT_OP[fnc], fnc, numb, start
                        else:
                            yield FUNC_NAME, fnc, numb, start
                        pos -= 1
                        continue
                    elif token[-1] == ':':
                        # Meta property name
                        # Yield name, then back up to yield COLON on next pass
                        yield META_NAME, token[:-1].strip(), numb, start
                        pos -= 1
                        continue
                    elif token in KEYWORD_MAP:
                        # Reserved keyword (and, or etc.)
                        yield KEYWORD_MAP[token], token, numb, start
                        continue
                    else:
                        # Variable name
                        yield NAME, token, numb, start
                        continue
                else:
                    m = _rSPACE.search(line, pos)
                    if not m:
                        token = line[pos:]
                    else:
                        token = line[pos:m.start()]
                    raise ParseError('Unknown or invalid token', numb, pos,
                        'Unrecognized token: ' + token)
            # Join lines with open parentheses
            if bracket_depth > 0:
                append_next_line = True
            else:
                # Yield end-of-line
                if not (append_next_line or in_multi_string):
                    yield EOL, _sEOL, numb, ln
        # End of file, test parentheses mismatch
        if bracket_depth > 0:
            numb, pos = bracket_lines.pop()
            raise ParseError('Parentheses mismatch', numb, pos,
                'Parentheses opened but never closed')
        if in_multi_string:
            raise ParseError('Unclosed multi-line string', text_start[0],
                text_start[1] - 3, 'Comment opened but never closed')
        # De-dent at end of file
        if check_indenting:
            while dents > 0:
                dents -= 1
                yield DEDENT, '', numb, ln
        yield EOF, '', numb, ln
    def _post_process_multiline_string(self, text_buffer):
        """
        Post-processes a multi line meta value, removing unnecessary
        whitespace.

        Any value given using the triple-quote syntax must be pre-processed.
        This function finds the lowest indentation level used in the string's
        declaration and trims the corresponding whitespace from each line. The
        result is then returned, with the line breaks and everything else left
        intact.
        """
        # Find lowest whitespace level, trim from left
        ind = 'first'
        for line in text_buffer:
            if ind == 'first':
                # Skip first line
                ind = None
            elif line.strip() == '':
                # Skip blank lines
                continue
            else:
                # Find whitespace level, update global
                white = _rWHITE.match(line)
                white = line[0:white.end()]
                pos = 0
                column = 0
                for char in white:
                    if char == ' ':
                        column += 1
                    elif char == '\t':
                        column = (1 + column // tabsize) * tabsize
                    elif char == '\f':
                        column = 0
                if ind is None or column < ind:
                    ind = column
        # No indentation? Then return as is
        if ind == 0 or ind is None or ind == 'first':
            return '\n'.join(text_buffer)
        # Strip initial ``ind`` characters from lines
        text = []
        first = True
        for line in text_buffer:
            if first:
                text.append(line.lstrip())
                first = False
            elif line.strip() == '':
                text.append('')
            else:
                pos = 0
                column = 0
                while column < ind:
                    if line[pos] == ' ':
                        column += 1
                    elif line[pos] == '\t':
                        column = (1 + column // tabsize) * tabsize
                    elif line[pos] == '\f':
                        column = 0
                    else:
                        raise Exception('Unexpected character in multi-line'
                            ' string\'s whitespace: "' + line[pos] + '"')
                    pos += 1
                text.append(line[pos:])
        return ('\n'.join(text)).strip()
def parse_proto_expression(stream, info=None, rbp=0):
    """
    Parses an expression, requires a :class:`TokenStream` ``stream``, a
    :class:`ParseInfo` object ``info`` and an integer ``rbp`` representing the
    current right-binding-power.

    The function will return only when all tokens are exhausted or a lower
    binding power is encountered.
    
    Returns a "proto-expression"; a tuple ``(Type, Arguments, Tokens)``. Here
    ``Type`` is the expression type (either a :class:`myokit.Expression` or the
    name of a user function). ``Arguments`` is a sequence of constructor
    arguments and ``Tokens`` is a sequence of tokens.
    """
    if info is None:
        info = ParseInfo()
    # Parse first token using null denomination
    code, text, line, char = stream.peek()
    try:
        parser = nud_parsers[code]
    except KeyError:
        unexpected_token(stream.peek(), 'expression')
    expr = parser.parse(stream, info)
    # Parse follow up tokens
    code, text, line, char = stream.peek()
    while True:
        try:
            parser = led_parsers[code]
        except KeyError:
            break
        if parser._rbp <= rbp:
            break
        expr = parser.parse(expr, stream, info)
        code, text, line, char = stream.peek()
    return expr
def parse_expression_stream(stream, context=None):
    """
    Parses an expression from a stream. A :class:`myokit.Variable` object can
    be given as ``context`` to resolve any references against.
    """
    return convert_proto_expression(parse_proto_expression(stream), context)
def parse_expression_string(string, context=None):
    """
    Parses string data into a :class:`myokit.Expression`.
    
    A :class:`myokit.Variable` object can be given as ``context`` to resolve
    any references against.
    """
    s = Tokenizer(string)
    e = parse_proto_expression(s)
    expect(s.next(), EOL)
    expect(s.next(), EOF)
    try:
        s.next()
        raise ParseError('Unused tokens', 0, 0, 'Expecting a string containing'
            ' only a single expression.')
    except StopIteration:
        return convert_proto_expression(e, context)
def parse_number_string(string):
    """
    Parses string data into a :class:`myokit.Number`.
    """
    s = Tokenizer(string)
    p = NumberParser()
    e = p.parse(s, ParseInfo())
    expect(s.next(), EOL)
    expect(s.next(), EOF)
    try:
        s.next()
        raise ParseError('Unused tokens', 0, 0, 'Expecting a string containing'
            ' only a single number.')
    except StopIteration:
        return convert_proto_expression(e)
def convert_proto_expression(e, context=None, info=None):
    """
    Resolves a proto-expression into a :class:`myokit.Expression`. Requires a
    :class:`myokit.Variable` object ``context`` if any references are to be
    resolved. If a :class:`ParseInfo` object ``info`` is passed in it will be
    used to register tokens.
    """
    def convert(x):
        # Unpack proto expression
        element, ops, tokens = x
        # Create expression
        if element == myokit.Number:
            # Handle Numbers
            e = myokit.Number(*ops)
        elif element == myokit.Name:
            # Handle Names, resolve references
            if context is None:
                # No variable given to determine scope and resolve references
                # with, return a string-based name for function templates (used
                # for user functions) and debugging.
                return myokit.Name(*ops)
            # Resolve reference
            # Note: I tried caching resolved references per variable, but it
            # didn't speed-up or slow down execution time.
            try:
                e = myokit.Name(context._resolve(ops[0]))
            except myokit.UnresolvedReferenceError as e:
                a, b = tokens[0][2:4] if tokens else (0, 0)
                m = e.message
                raise ParseError('Unresolved reference', a, b, m, cause=e)
        elif isinstance(element, myokit.UserFunction):
            # Handle user function
            # Get mapping of function argument names to input values
            args = {}
            for k, p in enumerate(element.arguments()):
                args[p] = convert_proto_expression(ops[k], context, info)
            e = element.convert(args)
        else:
            # Handle other types
            ops = [convert(op) for op in ops]
            try:
                e = element(*ops)
            except myokit.IntegrityError as e:
                line, char = tokens[0][2:4] if tokens else (0, 0)
                raise ParseError('Syntax error', line, char, e.message,
                    cause=e)
        # Register tokens
        if info:
            for token in tokens:
                if token:
                    reg_token(info, token, e)
        return e
    return convert(e)
def format_parse_error(ex, source=None):
    """
    Turns a ParseError ``ex`` into a detailed error message.

    If a filename or input stream is passed in as ``source``, this will
    be used to show the line on which an error occurred. If a stream is passed,
    it should be rewinded to the same point the original parsing started.
    """
    out = [ex.name]
    if ex.desc is not None:
        out.append('  ' + ex.desc)
    out.append('On line ' + str(ex.line) + ' character ' + str(ex.char))
    tab = '\t'
    line = None
    if ex.line > 0 and source is not None:
        kind = type(source)
        if (kind == str or kind == unicode) and os.path.isfile(source):
            # Re-open file, find line
            f = open(source, 'r')
            for i in xrange(0, ex.line):
                line = f.next()
            line = line.rstrip()
        else:
            i = 0
            for line in source:
                i += 1
                if i == ex.line:
                    break
            if i != ex.line:
                line = None
    if line is not None:
        # Skip initial whitespace
        pos = 0
        _sWHITE = ' \t\f'
        for char in line[0:ex.char]:
            if not char in _sWHITE:
                break
            pos += 1
        # Add line
        line = line[pos:].expandtabs(tabsize)
        char = ex.char - pos
        n = len(line)
        if n > 56:
            # Trim if too long
            p2 = min(n, ex.char + 30)
            p1 = p2 - 56
            if p1 < 0:
                p2 -= p1
                p1 = 0
            line = line[p1:p2]
            char = char - p1
            if p1 > 0:
                line = '..' + line
                char += 2
            if p2 < n:
                line += '..'
        out.append('  ' + line)
        # Add error indication
        out.append(' ' * (2 + char) + '^')
    return '\n'.join(out)
class NudParser(object):
    """
    Expression parser for nud operators.

    An expression parser to use when parsing a token as the first element
    of an expression. The term nud stands for 'null denotation' as it has no
    previous expression bound to it.
    """
    def __init__(self, element=None):
        self.element = element
        self._rbp = element._rbp if element else 0
    def parse(self, stream, info):
        raise NotImplementedError
class NumberParser(NudParser):
    """
    Parser for numeric literals.
    """
    def parse(self, stream, info):
        token = stream.next()
        unit = None
        if stream.peek()[0] == BRACKET_OPEN:
            unit = parse_bracketed_unit(stream)
        return (myokit.Number, (token[1], unit), (token,))
class NameParser(NudParser):
    """
    Parser for names.
    """
    def parse(self, stream, info):
        t1 = stream.next()
        t2 = t3 = t4 = None
        name = t1[1]
        if stream.peek()[0] == DOT:
            t2 = stream.next()
            t3 = expect(stream.next(), NAME)
            name += '.' + t3[1]
        return (myokit.Name, (name, ), (t1, t2, t3, t4))
class PrefixParser(NudParser):
    """
    Parser for prefix (single operand) operators.
    """
    def parse(self, stream, info):
        token = stream.next()
        arg = parse_proto_expression(stream, info, self._rbp)
        return (self.element, (arg, ), (token,))
class GroupingParser(NudParser):
    """
    Parser for grouping IE parentheses; as in 5 * (2 + 3).
    """
    def parse(self, stream, info):
        token = stream.next()
        expr = parse_proto_expression(stream, info, self._rbp)
        expect(stream.next(), PAREN_CLOSE)
        return expr
class FunctionParser(NudParser):
    """
    Parser for function calls.
    """
    def __init__(self):
        super(FunctionParser, self).__init__()
        self._rbp = myokit.Function._rbp
    def parse(self, stream, info):
        name = stream.next()
        ops = []
        token = stream.next()
        while token[0] != PAREN_CLOSE:
            ops.append(parse_proto_expression(stream, info))
            token = expect(stream.next(), [COMMA, PAREN_CLOSE])
        if name[1] in functions:
            # Predefined function
            func = functions[name[1]]
            if func._nargs is not None:
                # Allow number-of-arguments check to be bypassed
                if not len(ops) in func._nargs:
                    raise ParseError('Syntax error', name[2], name[3], 'Wrong'
                        ' number of arguments for function '+ str(func._fname)
                        + '()')
            return (func, ops, (name,))
        else:
            # User-defined function
            try:
                func = info.model.get_function(name[1], len(ops))
            except KeyError:
                raise ParseError('Unknown function', name[2], name[3],
                    'A function ' + name[1] + '() with ' + str(len(ops)) +
                     ' argument(s) could not be found.')            
            # Found function, return template, arguments and tokens. "func" is
            # now a (template) Expression object.
            return (func, ops, (name,))
class LedParser(object):
    """
    Expression parser for led operators.

    An expression parser to use when parsing a token with an existing
    expression. The term led stands for 'left denotation' as it is bound to
    an existing expression on its left.
    
    Arguments:
    
    ``element``
        The model element this parser will create.
    ``rbp``
        This parser's right binding power. A ``LedParser`` will absorb tokens
        on its right as long as it has more binding power than the token
        on its left.
    """
    def __init__(self, element):
        self.element = element
        self._rbp = element._rbp if element else 0
    def parse(self, left, stream, info):
        raise NotImplementedError
class InfixParser(LedParser):
    """
    Parser for infix operators.
    """
    def parse(self, left, stream, info):
        token = stream.next()
        right = parse_proto_expression(stream, info, self._rbp)
        return (self.element, (left, right), (token,))
# Null denotation parsers
nud_parsers = {}
nud_parsers[NAME] = NameParser()
nud_parsers[INTEGER] = NumberParser()
nud_parsers[FLOAT] = NumberParser()
nud_parsers[PLUS] = PrefixParser(myokit.PrefixPlus)
nud_parsers[MINUS] = PrefixParser(myokit.PrefixMinus)
nud_parsers[NOT] = PrefixParser(myokit.Not)
nud_parsers[PAREN_OPEN] = GroupingParser()
nud_parsers[FUNC_NAME] = FunctionParser()
# Left denomination parsers
led_parsers = {}
led_parsers[PLUS] = InfixParser(myokit.Plus)
led_parsers[MINUS] = InfixParser(myokit.Minus)
led_parsers[STAR] = InfixParser(myokit.Multiply)
led_parsers[SLASH] = InfixParser(myokit.Divide)
led_parsers[REMAINDER] = InfixParser(myokit.Remainder)
led_parsers[QUOTIENT] = InfixParser(myokit.Quotient)
led_parsers[POWER] = InfixParser(myokit.Power)
led_parsers[EQEQUAL] = InfixParser(myokit.Equal)
led_parsers[NOTEQUAL] = InfixParser(myokit.NotEqual)
led_parsers[MORE] = InfixParser(myokit.More)
led_parsers[LESS] = InfixParser(myokit.Less)
led_parsers[MOREEQUAL] = InfixParser(myokit.MoreEqual)
led_parsers[LESSEQUAL] = InfixParser(myokit.LessEqual)
led_parsers[AND] = InfixParser(myokit.And)
led_parsers[OR] = InfixParser(myokit.Or)
# Functions
functions = {}
functions['sqrt'] = myokit.Sqrt
functions['sin'] = myokit.Sin
functions['cos'] = myokit.Cos
functions['tan'] = myokit.Tan
functions['asin'] = myokit.ASin
functions['acos'] = myokit.ACos
functions['atan'] = myokit.ATan
functions['exp'] = myokit.Exp
functions['log'] = myokit.Log
functions['log10'] = myokit.Log10
functions['floor'] = myokit.Floor
functions['ceil'] = myokit.Ceil
functions['abs'] = myokit.Abs
functions['if'] = myokit.If
# Advanced functions
functions['dot'] = myokit.Derivative
functions['piecewise'] = myokit.Piecewise
functions['opiecewise'] = myokit.OrderedPiecewise
functions['spline'] = myokit.Spline
functions['polynomial'] = myokit.Polynomial
