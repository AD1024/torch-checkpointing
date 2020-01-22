from functools import reduce


def to_camel_cases(x):
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