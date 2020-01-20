import torch
import torchvision
from .utils import to_pyid, make_func_call, to_camel_cases
from functools import reduce


SIZE_FLOAT = 4

class Node:
    '''
        A Computation Graph Node Representation
    '''
    def __init__(self, name, op, params, shape, inputs, outputs, output_size=0):
        self.id = name
        self.op = op
        self.params = params
        self.shape = shape
        self.outputs = outputs
        self.inputs  = inputs
        self.checkpoint = False
        self.edges = []
        if self.shape:
            self.output_size = reduce(lambda x,y: x + y, 
                                        map(lambda v: SIZE_FLOAT * reduce(lambda x,y: x * y, v), 
                                            shape.values()))
        else:
            self.output_size = output_size
    
    def adjacent_nodes(self, graph):
        '''
            Get Nodes adjacent to current node
        '''
        if (self.edges):
            for name, src_name in self.edges:
                yield graph[name], src_name
    
    def to_python(self, env: set, src=False):
        '''
            Parse to Python Object / Source Code
            Note: Parsing to Python src will modify (add new variables) the context (env)

            :params:
                graph   The computation graph contains current node
                env     Context
                src     Pass True to parse to python source code
            
            :returns:
                A Python Object of current Node or a corresponding
                Python Source Code
        '''
        raise NotImplementedError()


class PrimNode(Node):
    '''
        prim Scope Node
    '''
    def __init__(self, name, op, params, value, inputs, outputs):
        super().__init__(name, op, params, None, inputs, outputs)
        self.value = value
    
    def to_python(self, env: set, src=False):
        _, op_name = self.op.split('::')
        if op_name == 'Constant':
            if not src:
                return self.value
            out_var = to_pyid(self.outputs[0])  # SSA
            env.add(out_var)
            return f'{out_var} = {self.value}'
        elif op_name == 'ListConstruct':
            input_vars = [to_pyid(x) for x in self.inputs]
            if not src:
                return list
            out_var = to_pyid(self.outputs[0]) # SSA
            env.add(out_var)
            return f'{out_var} = [{", ".join(input_vars)}]'
        else:
            raise NotImplementedError(f'{op_name} is not supported')


class AtenNode(Node):
    '''
        aten Operator Node
    '''
    def __init__(self, name, op, params, shape, inputs, outputs):
        super().__init__(name, op, params, shape, inputs, outputs)
    
    def to_python(self, env: set, src=False):
        _, op_name = self.op.split('::')
        func = ''
        if hasattr(torch, op_name):
            func = f'torch.{op_name}' if src else getattr(torch, op_name)
        elif hasattr(torch.nn, camel_case_name):
            camel_case_name = to_camel_cases(op_name)
            func = f'torch.nn.{op_name}' if src else getattr(torch.nn, camel_case_name)
        else:
            raise Exception(f'Unknown operator: {op_name}')
        
        if src:
            input_vars = [to_pyid(x) for x in self.inputs]
            if not input_vars or all((i in env for i in input_vars)):
                func_call = make_func_call(func, *input_vars)
                out_var = to_pyid(self.outputs[0])
                env.add(out_var)    # SSA
                return f'{out_var} = {func_call}'
            else:
                raise Exception(f'{", ".join([str(x) for x in filter(lambda x: x not in env, input_vars)])}' \
                                + f' are referred in {self.id}({self.outputs[0]}) but not found in ctx')
        return func

