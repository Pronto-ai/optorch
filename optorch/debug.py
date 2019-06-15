import torch
from graphviz import Digraph

def _size_to_str(size):
    return '('+','.join(['%d' % v for v in size])+')'

_ANNOTATIONS = {}

def annotate(var, name):
    _ANNOTATIONS[var.grad_fn] = name

def make_dot(var, params):
    node_attr = dict(style='filled', shape='box')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size='12,12'))
    seen = set()
    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), _size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                dot.node(str(id(var)), 'variable '+str(type(var).__name__), fillcolor='lightblue')
            else:
                text = str(type(var).__name__)
                if var in _ANNOTATIONS:
                    text = _ANNOTATIONS[var] + ' (' + text + ')'
                    dot.node(str(id(var)), text, fillcolor='green')
                else:
                    dot.node(str(id(var)), text)
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)

    dot.format = 'pdf'
    dot.render()
