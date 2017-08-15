#
# Provides helper functions for import and export of XML based formats
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
import re
import xml.dom
import textwrap
import HTMLParser
import myokit
def dom_child(node, selector=None):
    """
    Returns the first child element of the given DOM node.

    If the optional selector is given it searches for an element of a
    particular type.
    
    Returns ``None`` if no such node is found.
    """
    enode = xml.dom.Node.ELEMENT_NODE
    e = node.firstChild
    if selector:
        while e is not None:
            if e.nodeType == enode and e.tagName == selector:
                return e
            e = e.nextSibling
    else:
        while e is not None:
            if e.nodeType == enode:
                return e
            e = e.nextSibling
    return None
def dom_next(node, selector=False):
    """
    Returns the next sibling element after the given DOM node.

    If the optional selector is given it searches for an element of a
    particular type.
    
    Returns ``None`` if no such node is found.
    """
    enode = xml.dom.Node.ELEMENT_NODE
    e = node.nextSibling
    if selector:
        while e is not None:
            if e.nodeType == enode and e.tagName == selector:
                return e
            e = e.nextSibling
    else:
        while e is not None:
            if e.nodeType == enode:
                return e
            e = e.nextSibling
    return None
def html2ascii(html, width=79, indent='  '):
    """
    Flattens HTML and attempts to create readable ASCII code.

    The ascii will be text-wrapped after ``width`` characters. Each new level
    of nesting will be indented with the text given as ``indent``.
    """
    class Asciifier(HTMLParser.HTMLParser):
        INDENT = 1
        DEDENT = -1
        WHITE = [' ', '\t', '\f', '\r', '\n']
        def __init__(self, line_width=79, indent='  '):
            HTMLParser.HTMLParser.__init__(self) # HTMLParser is old school
            self.text = []  # Current document
            self.line = []  # Current (unwrapped) line
            self.limode = None
            self.licount = 0
            self.LW = line_width
            self.TAB = indent
        def endline(self):
            """
            End the current line.
            """
            line = ''.join(self.line)
            self.line = []
            if line:
                self.text.append(line)
        def blankline(self):
            """
            Inserts a blank line
            """
            i = -1
            last = self.text[-1:]
            while last in [[self.INDENT], [self.DEDENT]]:
                i -= 1
                last = self.text[i:1+i]
            if last != ['']:
                self.text.append('')
        def handle_data(self, data):
            """
            Handle text between tags
            """
            data = str(data.strip().encode('ascii', 'ignore'))
            if data:
                self.line.append(data)
        def handle_starttag(self, tag, attrs):
            """
            Opening tags
            """
            if tag == 'p':
                self.blankline()
            elif tag == 'h1':
                self.text.append('='*self.LW)
            elif tag == 'h2':
                self.blankline()
                self.text.append('-'*self.LW)
            elif tag == 'h3':
                self.blankline()
                self.text.append('.'*self.LW)
            elif tag == 'ul' or tag == 'ol':
                self.endline()
                self.text.append(self.INDENT)
                self.limode = tag
                self.licount = 0
            elif tag == 'li':
                if self.limode == 'ul':
                    self.line.append('* ')
                else:
                    self.licount += 1
                    self.line.append(str(self.licount) + ' ')
            elif tag == 'em' or tag =='i':
                self.line.append(' *')
            elif tag == 'strong' or tag =='b':
                self.line.append(' **')
            elif tag == 'u':
                self.line.append(' _')
        def handle_startendtag(self, tag, attrs):
            if tag == 'br':
                self.endline()
            elif tag == 'hr':
                self.text.append('-'*self.LW)
        def handle_endtag(self, tag):
            if tag == 'p':
                self.endline()
                self.blankline()
            elif tag == 'h1':
                self.endline()
                self.text.append('='*self.LW)
                self.blankline()
            elif tag == 'h2':
                self.endline()
                self.text.append('-'*self.LW)
                self.blankline()
            elif tag == 'h3':
                self.endline()
                self.text.append('.'*self.LW)
                self.blankline()
            elif tag == 'ul' or tag == 'ol':
                self.endline()
                self.text.append(self.DEDENT)
                self.blankline()
            elif tag == 'li':
                self.endline()
            elif tag == 'em' or tag == 'i':
                self.line.append('* ')
            elif tag == 'strong' or tag == 'b':
                self.line.append('** ')
            elif tag == 'u':
                self.line.append('_ ')
        def gettext(self):
            self.endline()
            buf = []
            pre = ''
            n = self.LW
            ntab = len(self.TAB)
            dent = 0
            white = '[' + ''.join(self.WHITE) + ']+'
            space = ' '
            for line in self.text:
                if line == self.INDENT:
                    dent += 1
                    pre = dent * self.TAB
                    n -= ntab
                elif line == self.DEDENT:
                    dent -= 1
                    pre = dent * self.TAB
                    n += ntab
                else:
                    line = re.sub(white, space, line).strip()
                    if line == '':
                        buf.append('')
                    else:
                        buf.extend([pre + x for x in textwrap.wrap(line, n)])
            return ('\n'.join(buf)).strip()
    f = Asciifier(width, indent)
    f.feed(html)
    f.close()
    return f.gettext()
class TinyHtmlPage(object):
    """
    Can be used to create tiny html pages, in an object oriented way.

    Every page has a head and a body. The head contains meta information and
    can be manipulated using the methods of the TinyHtmlPage object. The body
    is represented as a TinyHtmlNode and can contain nodes (corresponding to
    html tags) out of a small subset of HTML.

    ** Example **

    The example below shows how you can create a very simple html page.

    >>> import myokit
    ... page = myokit.mxml.TinyHtmlPage()
    ... page.set_title('TinyHtml test page')
    ... p = page.append('p')
    ... p.text('This is some text. ')
    ... e = p.append('em')
    ... e.text('This bit is emphasised! ')
    ... p.text('This bit comes after the emphasised bit.')
    ... p.append('strong').text('You can use chaining too!')

    Now calling "print(page)" should output a complete web page.
    """
    def __init__(self):
        self._title = 'Untitled'
        self._body = TinyHtmlNode('body')
    def append(self, name):
        """
        Appends an element to this page and returns the created node.
        """
        return self._body.append(name)
    def html(self, pretty=False):
        """
        Returns this page's html
        """
        buf = [
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"'
                ' "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">',
            '<html xmlns="http://www.w3.org/1999/xhtml">',
            '<head>',
	        '<meta name="title" content="',
	            self._title.replace('"', ''), '" />',
	        '<title>', self._title, '</title>',
            '</head>']
        self._body._html(buf)
        buf.append('</html>')
        buf = ''.join(buf)
        if not pretty:
            return buf
        else:
            import xml.dom.minidom
            buf = xml.dom.minidom.parseString(buf)
            return buf.toprettyxml()
    def set_title(self, title):
        """
        Changes this page's title
        """
        self._title = title
class TinyHtmlNode(object):
    """
    TinyHtml pages are built up out of TinyHtmlNode objects. TinyHtmlNodes are
    constructed following the scheme laid out in the TinyHtmlScheme object.
    """
    def __init__(self, name, definitions=None):
        """
        Creates a new TinyHtmlNode.
        """
        self._name = name
        if not definitions:
            definitions = TinyHtmlScheme.node_defs[name]
        self._opt_attr = definitions[0] or []
        self._req_attr = definitions[1] or []
        self._pos_kids = definitions[2] or []
        self._kids = []
        self._text = None   # Only set for <text> nodes
        self._math = None   # Only set for <math> nodes
        self._attributes = {}
    def append(self, name):
        """
        Creates a nodeppends it to this nodend returns the new node.
        """
        # Get node definitions
        if name == 'text':
            raise ValueError('Text nodes can only be created using the text()'
                ' method.')
        try:
            defs = TinyHtmlScheme.node_defs[name]
        except KeyError:
            raise ValueError('Unknown node type: "' + str(name) + '".')
        # Test if type is accepted
        if not name in self._pos_kids:
            raise ValueError('Node type "' + str(name) + '" can not be added'
                ' to node of type "' + str(self._name) + '".')
        # Test if type can be appended at this time
        if len(defs) > 3:
            if not defs[3](self, name):
                raise ValueError('Node type "' + str(name) + '" could not be'
                    ' added to node of type "' + str(self._name) + '" at this'
                    ' time.')
        # Create, append and return
        node = TinyHtmlNode(name, defs)
        self._kids.append(node)
        return node
    def attr(self, attribute):
        """
        Returns the value set for this attribute
        """
        return self._attributes[attribute]
    def html(self):
        """
        Returns this node's html representation
        """
        return ''.join(self._html())
    def _html(self, buf=None):
        for at in self._req_attr:
            if not at in self._attributes:
                raise Exception('Missing required attribute: "' + str(at)
                    + '" in <' + str(self._name) + '> element.')
        if buf is None:
            buf = []
        if self._text:
            buf.append(self._text)
            return
        buf.extend(['<', self._name])
        if self._attributes:
            buf.extend([' ' + k + '="' + v.replace('"', '') + '"' for k,v in
                self._attributes.iteritems()])
        if self._math:
            buf.extend(('>', self._math, '</', self._name, '>'))
        elif self._pos_kids:
            buf.append('>')
            for kid in self._kids:
                kid._html(buf)
            buf.extend(['</', self._name, '>'])
        else:
            buf.append('/>')
        return buf
    def set_attr(self, attribute, value):
        """
        Sets the value of attribute ``attribute`` to ``value``.
        Returns this node for chaining.
        """
        if not (attribute in self._opt_attr or attribute in self._req_attr):
            raise ValueError('Node type "' + self._name + '" does not support'
                ' attribute "' + str(attribute) + '".')
        self._attributes[attribute] = unicode(value)
    def text(self, text):
        """
        Creates and appends a text node to this node.
        """
        # Test if text is accepted
        if not 'text' in self._pos_kids:
            raise ValueError('No text can be added to node of type "'
                 + str(self._name) + '".')
        # Create, append and return
        node = TinyHtmlNode('text', TinyHtmlScheme.text_def)
        node._text = unicode(text)
        self._kids.append(node)
        return node
    def math(self, expr):
        """
        Adds a mathml element to this node. The element's contents will be set
        using the write_mathml() method on the given myokit ``Equation`` or
        ``Expression``.
        """
        if not 'math' in self._pos_kids:
            raise ValueError('No math can be added to node of type "'
                + str(self._name) + '".')
        # Create, append and return
        node = TinyHtmlNode('math', TinyHtmlScheme.math_def)
        node._math = write_mathml(expr, presentation=True)
        node.set_attr('xmlns', 'http://www.w3.org/1998/Math/MathML')
        self._kids.append(node)
        return node
class TinyHtmlScheme(object):
    """
    Acts as a namespace for the static lists and defintions used by
    TinyHtmlNodes.

    A node definition follows the structure of a Document Type Definition
    (DTD). This DTD defines which elements exist, which attributes they may
    have and which child elements they can contain. A special element "PCDATA"
    is used to indicate the textual contents of a node (if any).

    **Grouping**

    The elements (which will be defined in detail later) are grouped together
    in the following categories:

    formatting
        em, strong, q
    special
        br, img, code
    heading
        h1, h2, h3
    block
        div, heading, p, table, ol, ul, dl
    inline
        pcdata, formatting, special, a
    inblock
        pcdata, formatting, special, a, block

    **Formatting tags**

    em
        Used for emphasis, this usually means "italic"
        \\Attributes: id, class, style
        \\Children: inline
    strong
        Used for "strong" aka bolded text
        \\Attributes: id, class, style
        \\Children: inline
    q
        Used for (inline) quotes
        \\Attributes: id, class, style
        \\Children: inline

    ** Special tags **

    br
        This is a line break
        \\Attributes: None
        \\Children: None
    img
        An image
        \\Attributes: id, class, alt, src (required)
        \\Children: None
    code
        Represents a block of code
        \\Attributes: id, class, style
        \\Children: PCDATA

    ** Headings **

    h1
        The first heading
        \\Attributes: id, class, style
        \\Children: PCDATA
    h2
        The second heading
        \\Attributes: id, class, style
        \\Children: PCDATA
    h3
        The third heading (three is enough)
        \\Attributes: id, class, style
        \\Children: PCDATA

    ** Divisions **

    div
        A division
        \\Attributes: id, class, style
        \\Children: inblock

    ** Paragraph **

    p
        A paragraph
        \\Attributes: id, class, style
        \\Children: inline

    ** Lists **

    ul
        An unordered (bulleted) list
        \\Attributes: id, class, style
        \\Children: li (one or more)
    ol
        An ordered (numbered) list
        \\Attributes: id, class, style
        \\Children: li (one or more)
    li
        A list item
        \\Attributes: id, class, style
        \\Children: inblock
    dl
        A definition list
        \\Attributes: id, class, style
        \\Children: One or more pairs (dt, dd)
    dt
        A term to be defined
        \\Attributes: id, class, style
        \\Children: inline
    dd
        A term's definition
        \\Attributes: id, class, style
        \\Children: inblock

    ** Tables **

    table
        A table
        \\Attributes: id, class, style
        \\Children: tr
    thead
        A table header
        \\Attributes: id, class, style
        \\Children: tr
    tbody
        A table footer
        \\Attributes: id, class, style
        \\Children: tr
    tbody
        A table body
        \\Attributes: id, class, style
        \\Children: tr
    tr
        A row in a table
        \\Attributes: id, class, style
        \\Children: th, td
    th
        A cell in a table, formatted as a special header
        \\Attributes: id, class, style, rowspan, colspan
        \\Children: inblock
    td
        An ordinary cell in a table
        \\Attributes: id, class, style, rowspan, colspan
        \\Children: inblock

    ** The anchor element **

    a
        An anchor (a hyperlink)
        \\Attributes: id, class, style, href (required)
        \\Children: PCDATA, formatting, special

    ** The document body **

    body
        The document body
        \\Attributes: none
        \\Children: inblock

    ** Math **

    math
        A mathml container
        \\Attributes: id, class, xmlns (required, but set automatically)
        \\Children: Set automatically

    """
    # Groups
    group_text = ('text',)
    group_formatting = ('em', 'strong', 'q')
    group_special = ('br', 'img', 'code')
    group_heading = ('h1', 'h2', 'h3')
    group_block = group_heading + ('div', 'p', 'table', 'ol', 'ul', 'dl')
    group_inline = group_formatting + group_special + ('text', 'a', 'math')
    group_inblock = group_inline + group_block
    # Common attributes
    attr = ('id', 'class', 'style')
    # "No repeats" function
    f_rep = lambda parent, name: parent.kids and parent.kids[-1]._name != name
    # "Unique element" function
    f_one = lambda parent, name: name not in [x._name for x in parent.kids]
    # Node definitions
    # Tuple structure:
    #  [0] Optional attributes
    #  [1] Required attributes
    #  [2] Allowable child node types
    #  (3) A function handle can_append(parent, child) which should return
    #   True if and only if the node 'parent' can accept a node of the type
    #   'child' at this time. This should be set only for elements that can
    #   only be added to their parents under certain conditions.
    #
    body_def = (None, None, group_block)
    text_def = (None, None, None)
    math_def = (attr, ('xmlns',), None)
    node_defs = {
        # Formatting
        'em'     : (attr, None, group_inline),
        'strong' : (attr, None, group_inline),
        'q'      : (attr, None, group_inline),
        # Special
        'br'   : (attr, None, None),
        'img'  : (attr + ('alt',), ('src',), None),
        'code' : (attr, None, group_text),
        # Anchor & paragraph
        'p' : (attr, None, group_inline),
        'a' : (attr, ('href',), group_text + group_formatting + group_special),
        # Headings
        'h1' : (attr, None, group_text),
        'h2' : (attr, None, group_text),
        'h3' : (attr, None, group_text),
        # Lists
        'ul' : (attr, None, ('li',)),
        'ol' : (attr, None, ('li',)),
        'li' : (attr, None, group_inblock),
        'dl' : (attr, None, ('dt', 'dd')),
        'dt' : (attr, None, group_inline, f_rep),
        'dd' : (attr, None, group_inblock, f_rep),
        # Tables
        'table' : (attr, None, ('thead', 'tfoot', 'tbody', 'tr')),
        'thead' : (attr, None, ('tr',), f_one),
        'tfoot' : (attr, None, ('tr',), f_one),
        'tbody' : (attr, None, ('tr',), f_one),
        'tr'    : (attr, None, ('th', 'td')),
        'th'    : (attr + ('rowspan', 'colspan'), None, group_inblock),
        'td'    : (attr + ('rowspan', 'colspan'), None, group_inblock),
        # Text
        'text' : text_def,
        # Body
        'body' : body_def,
        # Divisions
        'div' : (attr, None, group_inblock),
        # MathML
        'math' : math_def,
        }
def write_mathml(expression, presentation):
    """
    Converts a myokit :class:`Expression` to a mathml expression.
    
    The boolean argument ``presentation`` can be used to select between
    Presentation MathML and Content MathML.
    """
    from myokit.formats.mathml import MathMLExpressionWriter
    w = MathMLExpressionWriter()
    w.set_mode(presentation=presentation)
    return w.ex(expression)
