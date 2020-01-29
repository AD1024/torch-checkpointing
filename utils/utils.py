from functools import reduce


def validate_indice(func):
    '''
        Validate indices for functions that
        accepts start, end and a list to operate
    '''
    def wrapper(start, end, xs, *args):
        assert start <= end
        assert start >= 0 and start < len(xs)
        assert end >= 0 and end < len(xs)
        return func(start, end, xs, *args)
    return wrapper

def to_camel_cases(x):
    '''
        Convert a PEP8 style variable name to camel case style
    '''
    if not x:
        return ''
    if '_' not in x:
        return x[0].upper() + x[1:]
    xs = x.split('_')
    if len(xs) > 1:
        hd, *tail = xs
        return to_camel_cases(hd) \
             + reduce(lambda x,y: x + y, map(to_camel_cases, tail))
    else:
        result = xs[0]
        if result:
            return result[0].upper() + result[1:]

def to_pyid(name):
    return f'var_{name}'

def make_func_call(func, *params):
    return f'{func}({", ".join(map(str, params))})'

def traverse_graph(start, graph: dict, func=lambda _: None):
    '''
        Traverse a graph and execute provided
        operation on current nodes.
    '''
    queue = [start]
    in_queue = set({start.id})
    while queue:
        first, *queue = queue
        in_queue.remove(first.id)
        func(first)
        for v, _ in first.adjacent_nodes(graph):
            if v.id not in in_queue:
                in_queue.add(v.id)
                queue.append(v)

def collect_operators(scope: str, start, graph: dict) -> set:
    result = set()
    def process_first(first):
        nonlocal result
        op_scope, op = first.op.split('::')
        if op_scope.lower() == scope.lower():
            result.add(op)
    traverse_graph(start, graph, process_first)
    return result
