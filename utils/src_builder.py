import torch
import torch.jit
from .graph_node import Node
from .utils import traverse_graph


def to_python_src(start: Node, graph: dict):
    env = set()
    result = \
'''
import torch
import torchvison
{}
'''
    lines = []
    nodes = list(graph.values())
    nodes.sort(key=lambda node: node.outputs[0])
    for n in nodes:
        lines.append(n.to_python(env, src=True))
    return result.format("\n".join(lines))
    