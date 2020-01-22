import torch.utils.checkpoint
import hashlib
import re
from .graph_node import *
from functools import reduce
from collections import namedtuple
from torch import nn
from .utils import traverse_graph

Shape = namedtuple('Shape', ['type', 'sizes'])
Value = namedtuple('Value', ['type', 'value', 'sizes'])

def get_shape(node) -> dict:
    '''
        Get the shape (Tensor) of a node's outputs
        this is designed for aten:: scope
    '''
    outputs = dict()
    for o in node.outputs():
        typeIs = o.type()
        outputs[o.unique()] = Shape(type=re.match(r'\w+', typeIs.str()).group(), sizes=tuple(typeIs.sizes()))
    return outputs

def get_value(node) -> dict:
    '''
        Get the value (type and shape) of a node's outputs
        this is designed for prim:: scope
    '''
    outputs = dict()
    for o in node.outputs():
        typeIs = o.type().str()
        value  = o.toIValue()
        outputs[o.unique()] = Value(type=typeIs, value=value,\
                                    sizes=len(list(node.outputs())) if typeIs.endswith('[]') else 1)
    return outputs

def create_name(node):
    '''
        Create a name for a node (defined in PytorchScript) based
        on the kind of the operator and the output variable names.
    '''
    return node.kind() \
            + '~>'     \
            + hashlib.md5(str(reduce(
                                lambda x,y: x + y,
                                [str(x) for x in sorted((y.unique() for y in node.outputs()))]
                            )).encode()).hexdigest()

def parse_to_graph(model, args):
    '''
        Use torch.jit._get_trace_graph to convert a 
        given model to a directed-graph.

        :returns:
            params  The parameter that the graph takes in
            start   The entry point of the graph
            graph   A string->Node map that maps node names to instances
    '''
    # Note: 1.4.0 slightly changed the output.
    #       Here, trace is a Graph instance, so we
    #       don't have to call trace.graph
    graph, _ = torch.jit._get_trace_graph(model, args)
    # graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    start = None
    parsed_graph  = {}
    param_node    = graph.param_node()
    param_shape   = get_shape(param_node)
    # print(param_shape)
    param_outputs = [o.unique() for o in param_node.outputs()]
    params        = dict(((k, v) for k, v in map(lambda v: (v, param_shape[v]),\
                                                param_outputs)))
    parsed_param_node = Node(create_name(param_node), param_node.kind(),\
                             None, param_shape, None, param_outputs)
    for node in graph.nodes():
        op = node.kind();
        params = dict([(i, node[i]) for i in node.attributeNames()]) if hasattr(node, '__getitem__') else None
        outputs = [o.unique() for o in node.outputs()]
        inputs  = [i.unique() for i in node.inputs()]
        if op.split('::')[0] == 'prim':
            graph_node = PrimNode(create_name(node), op, params, get_value(node), inputs, outputs)
        else:
            graph_node = AtenNode(create_name(node), op, params, get_shape(node), inputs, outputs)
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
    return parsed_param_node, start, parsed_graph

def checkpointing_with_budget(start: Node, graph: dict, budget: int, params: Node, verbose=False):
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
                        `params`:       the parameters that graph takes
                        `start`:        entry point
                        `inter_stage`:  "approximate cost to store inter-stage feature map"
                        `max_usage`:    maximum substage memory cost
                        `graph`:        checkpointed computation graph 
                        `checkpoints`:  a list of node id that are marked as checkpoints
    '''
    if (verbose):
        print(f'Checkpointing with {budget} bytes budget')
    checkpointed_graph = graph.copy()
    checkpoints = []
    temp, x, y = 0, 0, 0
    # Reference:
    # Tianqi Chen, Bing Xu, Chiyuan Zhang and Carlos Guestrin.
    # "Training Deep Nets with Sublinear Memory Cost".
    # Apr. 22, 2016
    # https://arxiv.org/abs/1604.06174
    def decide_checkpoint(first):
        nonlocal temp, x, y
        output_size = first.get_output_size()
        temp += output_size
        if temp > budget:
            if verbose:
                print(f'Usage: {temp} -- Checkpoint on {first.id}')
            x += output_size
            y = max(temp, y)
            temp = 0
            checkpointed_graph[first.id].checkpoint = True
            # Graph traversal ensures topological order
            checkpoints.append(first.id)
    traverse_graph(start, graph, decide_checkpoint)
    Result = namedtuple('Result', ['params', 'start', 'inter_stage', 'max_usage', 'graph', 'checkpoints'])
    return Result(params=params,
                  start=checkpointed_graph[start.id],
                  inter_stage=x,
                  max_usage=y,
                  graph=checkpointed_graph,
                  checkpoints=checkpoints)

def auto_checkpoint(model, inp, budget, verbose=False):
    '''
        Perform checkpointing algorithm on a model

        :params:
            model   Model to tag checkpoints
            inp     Input for the model
            budget  Memory budget, i.e. restriction
            verbose Debug info switch
    '''
    model.train(False)
    params, start, graph = parse_to_graph(model, inp)
    return checkpointing_with_budget(start, graph, budget, params, verbose=verbose)
