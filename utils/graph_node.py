import torch
import torchvision
from collections import namedtuple
from .utils import to_pyid, make_func_call, to_camel_cases
from functools import reduce, partial


SIZE_FLOAT = 4
SIZE_INT = 4
SIZE_LONG  = 8
SIZE_NONE  = 1
SIZE_BOOL  = 1

calc_dict = { # dict for tensors
             'Long' : lambda size: SIZE_LONG  * size,
             'Float': lambda size: SIZE_FLOAT * size, 
             'None' : lambda size: SIZE_NONE * size, 
             # dict for prim nodes
             'int'  : lambda size: SIZE_INT * size,
             'float': lambda size: SIZE_FLOAT * size,
             'bool' : lambda size: SIZE_BOOL  * size,
             'int[]'    : lambda size: SIZE_INT * size,
             'Tensor[]' : lambda sizes: sum(map(partial(reduce, lambda x, y: x * y), sizes))}

# Specify parameters used to construct a layer
module_params = {
    'torch.nn.AdaptiveAvgPool2d' : lambda params: [ params[1] ],
    'torch.nn.AvgPool2d'         : lambda params: params[1:]
}

# Specify what the constructed layer should accept in forward pass
module_accepts = {
    'torch.nn.AdaptiveAvgPool2d' : lambda params: [ params[0] ],
    'torch.nn.AvgPool2d'         : lambda params: [ params[0] ],
}

def list_params_to_code(params: list):
    '''
        Convert a list to source code
    '''
    result = '['
    for param in params:
        if isinstance(param, list):
            result += list_params_to_code(param)
        else:
            result += str(param)
        result += ', '
    return result + ']'

# Store the function / operator and its argument to
# convert the procedure to a checkpointed procedure
# ParsedCode = namedtuple('ParsedCode', ' '.join(['code', 'func', 'args', 'node_id', 'output_var']))
class ParsedCode:
    def __init__(self, code=None, func=None, args=None, node_id=None, output_var=None):
        self.code = code
        self.func = func
        self.args = args
        self.node_id = node_id
        self.output_var = output_var
    
    def __repr__(self):
        return f'{self.func} accepts {self.args}'
    
    def __str__(self):
        return self.__repr__()

class Node:
    '''
        A Computation Graph Node Representation
    '''
    def __init__(self, name, op, params, shape, inputs, outputs, output_size=None):
        self.id = name
        self.op = op
        self.params = params
        self.shape = shape
        self.outputs = outputs
        self.inputs  = inputs
        self.checkpoint = False
        self.output_size = output_size
        self.edges = []
    
    def adjacent_nodes(self, graph):
        '''
            Get Nodes adjacent to current node
        '''
        if (self.edges):
            for name, src_name in self.edges:
                yield graph[name], src_name
    
    def get_output_size(self):
        '''
            Get the size of outputs in bytes
        '''
        raise NotImplementedError()
    
    def to_python(self, ctx: dict, src=False, inline=True):
        '''
            Parse to Python Object / Source Code
            Note: Parsing to Python src will modify (add new variables) the context (env)

            :params:
                ctx     Context
                src     Pass True to parse to python source code
                inline  whether to inline lists and constants
            
            :returns:
                A Python Object of current Node (in-progress) or 
                A ParsedCode object:
                    `code`:         the actual source code
                    `func`:         function called in current node
                    `args`:         parameters to `func` (maybe inlined)
                    `node_id`:      id of the node
                    `output_var`:   variable name of the output of the node
        '''
        raise NotImplementedError()


class PrimNode(Node):
    '''
        prim Scope Node
    '''
    def __init__(self, name, op, params, value, inputs, outputs):
        super().__init__(name, op, params, None, inputs, outputs)
        self.value = value
    
    def get_output_size(self):
        if self.output_size is None:
            self.output_size = sum([calc_dict[value.type](value.sizes) for value in self.value.values()])
        return self.output_size
    
    def to_python(self, ctx: dict, src=False, inline=True):
        _, op_name = self.op.split('::')
        out_var = to_pyid(self.outputs[0])  # SSA
        # Update variable name
        ctx.update({ self.outputs[0] : out_var })
        if op_name == 'Constant':
            output = self.value[self.outputs[0]].value

            if not src:
                return output
            if inline:
                # Update constant value
                ctx[out_var] = output
                return None
            return ParsedCode(code=f'{out_var} = {output}', func=None,\
                              args=output, node_id=self.id, output_var=out_var)

        elif op_name == 'ListConstruct':
            input_vars = self.inputs
            if not src:
                return list

            # Ensures the identifiers referred in the current call
            # is defined in the context
            if all((ctx.get(i, None) is not None for i in input_vars)):

                if inline:
                    ctx[out_var] = list((ctx.get(ctx[i], ctx[i]) for i in input_vars))
                    return None
                
                inputs = [ctx[i] for i in input_vars]
                return ParsedCode(code=f'{out_var} = [{", ".join(inputs)}]',\
                                  func='list',\
                                  args=inputs, node_id=self.id, output_var=out_var)
            else:
                raise Exception(f'{", ".join((x for x in filter(lambda i: ctx.get(i, None) is None, input_vars)))}' +\
                                f'are referred in prim::ListConstruct ({self.outputs[0]}) but not found in ctx')
        else:
            raise NotImplementedError(f'prim::{op_name} is not supported')


class AtenNode(Node):
    '''
        aten Operator Node
    '''
    def __init__(self, name, op, params, shape, inputs, outputs):
        super().__init__(name, op, params, shape, inputs, outputs)
        self.func_call_rules = {
            'torch.nn.AvgPool2d'        : lambda func_name, *params: f'self.{func_name}({", ".join(params[1:])})({params[0]})',
            'torch.nn.AdaptiveAvgPool2d': lambda func_name, *params: f'self.{func_name}({params[1]})({params[0]})',
            'torch.addmm'               : lambda func_name, *params: f'{func_name}({", ".join(params[:-2])}, beta={params[-2]}, alpha={params[-1]})',
            'torch.add'                 : lambda func_name, *params: f'{func_name}({", ".join(params[:-1])}, alpha={params[-1]})'
        }

    def get_output_size(self):
        if self.output_size is None:
            self.output_size = sum([calc_dict[shape.type](reduce(lambda x, y: x * y, shape.sizes))\
                                                            for shape in self.shape.values()])
        return self.output_size
    
    def to_python(self, ctx: dict, src=False, inline=True):
        _, op_name = self.op.split('::')
        func = ''
        camel_case_name = to_camel_cases(op_name)

        # check whether it is a torch operator
        # or a torch module
        if hasattr(torch, op_name):
            func = f'torch.{op_name}' if src else getattr(torch, op_name)
        elif hasattr(torch.nn, camel_case_name):
            func = f'torch.nn.{camel_case_name}' if src else getattr(torch.nn, camel_case_name)
        elif hasattr(torch, op_name.replace('_', '')):
            func = f'torch.{op_name.replace("_", "")}'
        else:
            raise Exception(f'Unknown operator: {op_name}')
        
        if src:
            input_vars = self.inputs
            out_var = to_pyid(self.outputs[0])  # SSA
            ctx.update({ self.outputs[0] : out_var })
            if all((ctx.get(i, None) is not None for i in input_vars)):
                if inline:
                    func_args = [ctx.get(ctx.get(x), ctx.get(x)) for x in input_vars]
                else:
                    func_args = [ctx.get(ctx.get(x)) for x in input_vars]

                # Parameters to the function might be changed since
                # there may be variables in list, which will cause the code gen
                # to generate a new name for the list in a lifted function
                def func_call(func_name=func, func_args=func_args):
                    processed_arg = map(lambda x: list_params_to_code(x) if isinstance(x, list) else str(x), func_args)
                    rhs = self.func_call_rules.get(func, make_func_call)(func, *processed_arg)
                    return f'{out_var} = {rhs}'

                return ParsedCode(code=func_call, func=func, args=func_args,\
                                  node_id=self.id, output_var=out_var)
            else:
                raise Exception(f'{", ".join((str(x) for x in filter(lambda x: ctx.get(x, None) is None, input_vars)))}' \
                                + f' are referred in {self.id} ({self.outputs[0]}) but not found in ctx')
        return func

