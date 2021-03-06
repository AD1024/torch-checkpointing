import torch
import torch.jit
import random
import hashlib
from .graph_node import Node, ParsedCode, module_params, module_accepts
from .utils import to_pyid, validate_indice, make_func_call
from itertools import count

# ad hoc parameter renaming
process_id = lambda name: name.replace('[', '_').replace(']', '') \
                           if isinstance(name, str) and name.startswith('input') else name

clone_variable = lambda name: f'{name}.clone()' if isinstance(name, str) else name

def process_parameters(params: list, process_id) -> list:
    '''
        Functionally processing parameters using a given function

        :params:
            params          the list of parameters
            process_id      the parameter processing function
        
        :returns:
            A new list containing processed parameters
    '''
    if params == []:
        return []
    else:
        x, *xs = params
        if isinstance(x, list):
            return [ [ process_parameters(c, process_id) if isinstance(c, list) else process_id(c) for c in x ] ] + process_parameters(xs, process_id)
        else:
            return [ process_id(x) ] + process_parameters(xs, process_id)

def make_function(func_name, params, body):
    result = \
'''def {}(self, {}):
        {}
'''
    return result.format(func_name, ", ".join(map(process_id, params)),\
                        ("\n" + 8 * " ").join(body))

def make_torch_checkpoint_call(func_name, params):
    return f'torch.utils.checkpoint.checkpoint({func_name}, {", ".join(params)})'

def variable_in_list(xs: list):
    if isinstance(xs, list):
        return list(filter(lambda x: isinstance(x, str), xs))
    return []

@validate_indice
def local_variables(start: int, end: int, parsed_code: list) -> set:
    return set((code.output_var for code in (parsed_code[i] for i in range(start, end + 1))))

@validate_indice
def free_variables(start: int, end: int, parsed_code: list, local_vars=None) -> set:
    '''
        Collect free varibles if we lift codes in [start, end] are lifted to
        a lambda.

        :params:
            start         the start index of lifting
            end           the end index of lifting
            parsed_code   ParsedCode objects that are sorted in
                          topological order with respect to the original node
        
        :returns:
            a set of free varible names
    '''
    if local_vars:
        local_ref = local_vars
    else:
        local_ref = local_variables(start, end, parsed_code)
    free_vars  = set()
    for i in range(start, end + 1):
        for arg in parsed_code[i].args:
            if isinstance(arg, str) and arg not in local_ref:
                free_vars.add(arg)
            elif isinstance(arg, list):
                free_vars = free_vars.union(set(filter(lambda x: x not in local_ref, variable_in_list(arg))))
    return free_vars

@validate_indice
def referred_variables(start: int, end: int, parsed_code: list):
    '''
        Get variables that are used as inputs

        :params:
            start         the start index of lifting
            end           the end index of lifting
            parsed_code   ParsedCode objects that are sorted in
                          topological order with respect to the original node
        
        :returns:
            a set of referred variable names
    '''
    result = set()
    for i in range(start, end + 1):
        for arg in parsed_code[i].args:
            if isinstance(arg, str):
                result.add(arg)
            elif isinstance(arg, list):
                result = result.union(set(variable_in_list(arg)))
    return result

def checkpointing(parsed_code: list, checkpoints: list, output_var: str) -> str:
    '''
        Compile a checkpointed model forward pass code

        :params:
            parsed_code     Function calls and parameters in each line in SSA form
            checkpoints     Node ids that are marked as checkpoints
            output_var       The variable that stores the output result of the graph
        
        :returns:
            An executable python code that represents the checkpointed model
    '''
    def func_name_generator():
        cnt = count(0)
        while True:
            yield f'jojo_{next(cnt)}'
    
    def module_name_generator():
        cnt = count(0)
        while True:
            yield f'layer_{next(cnt)}'
    
    def hash_module(func_name, params):
        return hashlib.md5((func_name + str(params)).encode()).hexdigest()

    name_iter = func_name_generator()
    module_name_iter = module_name_generator()

    local_code = []
    declared_code = []
    modules_code = []
    module_name_map = dict()

    def lift_module(code: ParsedCode, args=None, clone_args=False):
        if not args:
            args = code.args
        if clone_args:
            args = process_parameters(args, clone_variable)
        if code.func.startswith('torch.nn'):
            constructor_params_getter = module_params.get(code.func, None)
            accept_params_getter      = module_accepts.get(code.func, None)
            if constructor_params_getter and accept_params_getter:
                constr_params = constructor_params_getter(args)
                accept_param  = accept_params_getter(args)
                module_id     = hash_module(code.func, constr_params)

                # Ensures no two identical layers are created
                if module_id not in module_name_map.keys():
                    module_name = next(module_name_iter)
                    module_name_map[module_id] = module_name
                    modules_code.append(f'self.{module_name} = {make_func_call(code.func, *constr_params)}')
                else:
                    module_name = module_name_map[module_id]

                rhs = make_func_call(f'self.{module_name}', *accept_param)
                return f'{code.output_var} = {rhs}'
            else:
                raise Exception(f'{code.func} not supported.')
        else:
            return code.code(func_args=args)

    cons = lambda elem, xs: [ elem ] + xs
    tail = len(parsed_code) - 1
    # Process the trailing segment (the last checkpoint to the end of the graph)
    head = tail
    while head >= 0 and parsed_code[head].node_id not in checkpoints:
        local_code = cons(lift_module(parsed_code[head]), local_code)
        head -= 1
    # Get the local refs of trailing segment
    if head >= 0:
        local_refs = referred_variables(head, tail, parsed_code)
    else:
        # Not a valid plan
        # local_code = [ lift_module(x) for x in parsed_code ] + [ f'return {output_var}' ]
        return {
            'modules'       : modules_code,
            'class_declared': [],
            'forward_local' : local_code + [ f'return {output_var}' ]
        }
    tail = head
    while tail >= 0:
        # Ensures: tail points either 0, -1 or a checkpoint
        local_code = cons(lift_module(parsed_code[tail]), local_code)
        local_refs  = local_refs.union(referred_variables(tail, tail, parsed_code))
        head = tail - 1
        while head >= 0 and parsed_code[head].node_id not in checkpoints:
            head -= 1
        # The adjacent precedence of the last checkpoint is not a checkpoint
        # then lift the segment to a closure
        if head != tail - 1:
            func_name = next(name_iter)
            body = []  # the lifted lambda body
            body_code = []
            referred = set()             # variables got referred later in the context (should not be lifted)

            for i in range(head + 1, tail):
                body.append(parsed_code[i])
                # if the output later is referred
                if parsed_code[i].output_var in local_refs:
                    referred.add(parsed_code[i].output_var)

            body.append(f'return {", ".join(referred)}')  # return the variables that are referred later

            for line in body:
                if isinstance(line, ParsedCode):
                    body_code.append(lift_module(line, args=process_parameters(line.args, process_id), clone_args=len(body_code) == 0 and line.func.endswith('_')))
                else:
                    body_code.append(str(line))

            # Free variables should be included in the parameters
            args_after_lift = free_variables(head + 1, tail - 1, parsed_code)
            declared_code.append(make_function(func_name, args_after_lift, body_code))
            local_code = cons(f'{", ".join(referred)} = {make_torch_checkpoint_call(f"self.{func_name}", args_after_lift)}', local_code)
            local_refs = local_refs.union(args_after_lift)
        tail = head
    local_code.append(f'return {output_var}')
    return {
        'modules'      :  modules_code,
        'forward_local':  local_code,
        'class_declared': declared_code
    }


def weight_gen(v_to_shape):
    '''
        Generate random weight according to the shape of inputs

        :params:
            v_to_shape      (Node * Shape)
        
        :returns:
            python code that assigns random weights
    '''
    shape = v_to_shape[1]
    weight_type = shape.type
    sizes = shape.sizes
    return {
        'Float': lambda sizes: f'torch.randn({sizes}) * 1e-7',
        'Long':  lambda     _: f'random.randint(32, 64)',
    }.get(weight_type, lambda x: '0')(sizes)

def build_forward():
    '''
        String constant. Wrapper for self.forward_
    '''
    result = \
'''def forward(self, inputs):
        return self.forward_([ inputs.requires_grad_(True) ] + self.weights)
'''
    return result

def build_init(param_node, modules):
    '''
        Weight initialization code

        :params:
            param_node      the parameter node of the computation graph
    '''
    result = \
'''def __init__(self):
        super().__init__()
        self.weights = [{}]
        {}
'''.format(', '.join(map(weight_gen, sorted(param_node.shape.items())[1:])), '\n        '.join(modules))
    return result

def build_src(name: str, param_node, modules: list, class_defined: list, forward_pass: list):
    '''
        Build the source code for a Module

        :params:
            name            the name of the module
            param_node      the parameter node of the computation graph
            class_defined   defined closures of lifted codes in forward
            forward_pass    actual codes in self.forward_
        
        :returns:
            Python source code of a checkpointed module
    '''
    foward_template = '''def forward_(self, {}):
        {}
    '''
    return '''import torch\nimport torch.utils.checkpoint\nimport random\n
class {}(torch.nn.Module):
    {}
    {}
    {}
    {}'''.format(name, build_init(param_node, modules),\
                        "\n    ".join(map(lambda x: f'{x}', class_defined)),\
                        foward_template.format("input_vars", ("\n" + 8 * " ").join(forward_pass)),\
                        build_forward())

def to_python_src(module_name: str, params: Node, start: Node, graph: dict, checkpoints: list):
    '''
        Compile the computation graph to Python source code

        :params:
            params:         parameters (input) of the graph
            start :         the entry node of the graph
            graph :         a string->Node map represents the nodes in the graph
            checkpoints:    node id that are marked as checkpoints
    '''
    cid = count(0)
    # Make sure the input variables are in the right places
    outputs = sorted(params.outputs)
    # Assign positions in the list
    env = dict(((k, v) for k, v in zip(outputs, map(lambda name: f'input_vars[{next(cid)}]', outputs))))
    lines = []
    nodes = list(graph.values())
    nodes.sort(key=lambda node: node.outputs[0])
    for n in nodes:
        new_line = n.to_python(env, src=True, inline=True)
        if new_line:
            lines.append(new_line)
    result_checkpoint = checkpointing(lines, checkpoints, lines[-1].output_var)
    return build_src(module_name, params,\
                     result_checkpoint['modules'],\
                     result_checkpoint['class_declared'],\
                     result_checkpoint['forward_local'])
