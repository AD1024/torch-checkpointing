import torch.utils.checkpoint
import hashlib
import re
from .graph_node import *
from functools import reduce
from collections import namedtuple
from torch import nn


def get_shape(node):
    # Extract the shape from the
    # string of a node
    # Reference: 
    # https://github.com/waleedka/hiddenlayer/blob/master/hiddenlayer/pytorch_builder.py
    outputs = dict()
    for o in node.outputs():
        grouped = re.match(r".*Float\(([\d\s\,]+)\).*", str(o).split(':')[1])
        if grouped:
            outputs[o.unique] = tuple(map(int, grouped.group(1).split(',')))
    return outputs

def get_value(node):
    grouped = re.search(r'value=\d+', str(node)) 
    if grouped:
        return grouped.group().split('=')[1]
    return None

def create_name(node):
    '''
        Create a name for a node (defined in PytorchScript) based
        on the kind of the operator and the output variable names.
    '''
    return node.kind() \
            + '~>'     \
            + hashlib.md5(str(reduce(
                                lambda x,y: x + y,
                                [str(x.unique()) for x in sorted(node.outputs())]
                            )).encode()).hexdigest()

def parse_to_graph(model, args):
    '''
        Use torch.jit._get_trace_graph to convert a 
        given model to a directed-graph.

        :returns:
            start   The entry point of the graph
            graph   A string->Node map that maps node names to instances
    '''
    # Note: 1.4.0 slightly changed the output.
    #       Here, trace is a Graph instance, so we
    #       don't have to call trace.graph
    graph, _ = torch.jit._get_trace_graph(model, args)
    # graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    start = None
    parsed_graph = {}
    for node in graph.nodes():
        op = node.kind();
        params = dict([(i, node[i]) for i in node.attributeNames()]) if hasattr(node, '__getitem__') else None
        outputs = [o.unique() for o in node.outputs()]
        inputs  = [i.unique() for i in node.inputs()]
        if op.split('::')[0] == 'prim':
            value = get_value(node)
            graph_node = PrimNode(create_name(node), op, params, value, inputs, outputs)
        else:
            shape = get_shape(node)
            graph_node = AtenNode(create_name(node), op, params, shape, inputs, outputs)
        parsed_graph[graph_node.id] = graph_node
        if start is None:
            start = graph_node
        # Reference:
        # https://github.com/waleedka/hiddenlayer/blob/master/hiddenlayer/pytorch_builder.py
        for to in graph.nodes():
            to_inputs = [i.unique() for i in to.inputs()]
            edges = set(outputs) & set(to_inputs)
            if edges:
                graph_node.edges.append((create_name(to), edges))
    return start, parsed_graph

def checkpointing_with_budget(start: Node, graph: dict, budget: int, verbose=False):
    '''
        Perform checkpointing on a parsed graph restricting
        the maximum memory usage.

        :params:
            start   The entry node
            graph   A string->Node map that maps node id to instances
            budget  The maximum memory usage restriction in bytes
            verbose Enable debug output
        
        :returns:
            result  A namedtuple that has 3 fields:
                        `start`:        entry point
                        `inter_stage`:  "approximate cost to store inter-stage feature map"
                        `max_usage`:    maximum substage memory cost
                        `graph`:        checkpointed computation graph 
    '''
    if (verbose):
        print(f'Checkpointing with {budget} bytes budget')
    checkpointed_graph = graph.copy()
    temp, x, y = 0, 0, 0
    queue = [start]
    in_queue = set({start.id})
    # Reference:
    # Tianqi Chen, Bing Xu, Chiyuan Zhang and Carlos Guestrin.
    # "Training Deep Nets with Sublinear Memory Cost".
    # Apr. 22, 2016
    # https://arxiv.org/abs/1604.06174
    while queue:
        first, *queue = queue
        temp += first.output_size
        in_queue.remove(first.id)
        if temp > budget:
            if verbose:
                print(f'Usage: {temp} -- Checkpoint on {first.id}')
            x += first.output_size
            y = max(temp, y)
            temp = 0
            checkpointed_graph[first.id].checkpoint = True
        for v, _ in first.adjacent_nodes(graph):
            if v.id not in in_queue:
                in_queue.add(v.id)
                queue.append(v)
    Result = namedtuple('Result', ['start', 'inter_stage', 'max_usage', 'graph'])
    return Result(start=checkpointed_graph[start.id],
                  inter_stage=x,
                  max_usage=y,
                  graph=checkpointed_graph)

def auto_checkpoint(model, inp, budget, verbose=False):
    '''
        Perform checkpointing algorithm on a model

        :params:
            model   Model to tag checkpoints
            inp     Input for the model
            budget  Memory budget, i.e. restriction
            verbose D
    '''
    model.train(False)
    start, graph = parse_to_graph(model, inp)
    return checkpointing_with_budget(start, graph, budget, verbose=verbose)

###################
# Debug-use code #
###################
def test_resnet18():
    # TODO: debug use. remove this function when not needed
    model = torchvision.models.resnet18()
    start, graph = parse_to_graph(model, torch.zeros([64, 3, 7, 7]))
    return start, graph

def test_vgg16():
    # TODO: debug use. remove this function when not needed
    model = torchvision.models.vgg16()
    model.train(False)
    start, graph = parse_to_graph(model, torch.zeros([1, 3, 224, 224]))
    return start, graph
