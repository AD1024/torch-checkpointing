import torch
import torch.jit
from .graph_node import Node
from .utils import to_pyid


def to_python_src(params: Node, start: Node, graph: dict):
    '''
        Compile the computation graph to Python source code

        :params:
            params: parameters (input) of the graph
            start : the entry node of the graph
            graph : a string->Node map represents the nodes in the graph
    '''
    env = dict(((k, v) for k, v in zip(params.outputs, map(to_pyid, params.outputs))))
    result = \
'''
def forward({}):
    import torch
    import torchvison
    {}
'''
    lines = []
    nodes = list(graph.values())
    nodes.sort(key=lambda node: node.outputs[0])
    for n in nodes:
        lines.append(n.to_python(env, src=True))
    return result.format(", ".join((to_pyid(x) for x in params.outputs)), "\n    ".join(lines))
    