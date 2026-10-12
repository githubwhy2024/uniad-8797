"""Remove proven internal identity views; keep rank/layout-changing SDK routes."""
import argparse
import copy
import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as H
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'validation'))
import shape_dataflow as engine
from prove_graph_extents import sha, save


def rewrite(model):
    types, constants, errors = engine.prove_static_dataflow(model)
    if errors:
        raise ValueError('identity-view root inference failed')
    public = {v.name for v in model.graph.output}
    aliases = {}
    removed = []
    nodes = []

    def resolve(name):
        while name in aliases:
            name = aliases[name]
        return name

    consumers={}
    for node in model.graph.node:
        for value in node.input:consumers.setdefault(value,[]).append(node)
    scalar_rank_guards={node.name for node in model.graph.node if node.op_type=='Reshape' and engine.fixed_extents(types[node.output[0]])==(1,) and any(child.op_type=='Reshape' and engine.fixed_extents(types[child.output[0]])==() for child in consumers.get(node.output[0],[]))}
    for node in model.graph.node:
        identity = False
        if len(node.output) == 1 and node.output[0] not in public:
            shape = engine.fixed_extents(types[node.input[0]]) if node.input else None
            if node.op_type == 'Reshape':
                identity = shape is not None and shape == engine.fixed_extents(types[node.output[0]]) and node.input[1] in constants
            elif node.op_type == 'Cast':
                identity = types[node.input[0]].tensor_type.elem_type == types[node.output[0]].tensor_type.elem_type
            elif node.op_type == 'Identity':
                identity = True
            elif node.op_type == 'Transpose' and shape is not None:
                perm = next((list(a.ints) for a in node.attribute if a.name == 'perm'), list(reversed(range(len(shape)))))
                identity = perm == list(range(len(shape)))
        if node.name in scalar_rank_guards:identity=False
        if identity:
            aliases[node.output[0]] = resolve(node.input[0])
            removed.append(dict(node=node.name, op_type=node.op_type, source=node.input[0], output=node.output[0], aliased_to=aliases[node.output[0]], shape=list(engine.fixed_extents(types[node.output[0]])), bytes=int(np.prod(engine.fixed_extents(types[node.output[0]])))*np.dtype(H.tensor_dtype_to_np_dtype(types[node.output[0]].tensor_type.elem_type)).itemsize))
        else:
            new = copy.deepcopy(node)
            for i, name in enumerate(new.input):
                new.input[i] = resolve(name)
            nodes.append(new)
    # Preserve a public identity-Cast name on its actual internal producer.
    # This avoids an input/output passthrough cut which the SDK cannot squash.
    protected=public|{v.name for v in model.graph.input}|{v.name for v in model.graph.initializer}
    produced={v for n in nodes for v in n.output}
    promotions={}
    kept=[]
    for node in nodes:
        name=node.input[0] if node.input else None
        while name in promotions:name=promotions[name]
        if node.op_type=='Cast' and node.output[0] in public and name in produced-protected and types[node.input[0]].tensor_type.elem_type==types[node.output[0]].tensor_type.elem_type:
            promotions[name]=node.output[0]
            removed.append(dict(node=node.name,op_type='Cast',source=node.input[0],output=node.output[0],aliased_to=node.output[0],shape=list(engine.fixed_extents(types[node.output[0]])),bytes=0,public_name_promoted=True))
        else:
            kept.append(node)
    for node in kept:
        for field in ('input','output'):
            values=getattr(node,field)
            for i,name in enumerate(values):
                while name in promotions:name=promotions[name]
                values[i]=name
    nodes=kept
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return removed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--model-sha256', required=True)
    a = p.parse_args()
    if sha(a.model) != a.model_sha256:
        raise ValueError('identity-view source differs')
    model = onnx.load(str(a.model))
    boundary = {k: [v.SerializeToString() for v in getattr(model.graph, k)] for k in ('input', 'output', 'initializer')}
    removed = rewrite(model)
    if not removed:
        raise ValueError('no identity view matched')
    if any(boundary[k] != [v.SerializeToString() for v in getattr(model.graph, k)] for k in boundary):
        raise ValueError('identity elimination changed ABI or original weights')
    output = Path.cwd()/'model.identity-eliminated.onnx'
    onnx.save(model, str(output))
    save(Path.cwd()/'identity_views.json', dict(status='pass', source_model=str(a.model), source_model_sha256=a.model_sha256, model=str(output), model_sha256=sha(output), script_sha256=sha(__file__), removed=removed, removed_logical_output_bytes=sum(row['bytes'] for row in removed), ordered_abi_and_weights_unchanged=True, scope='Root-proven internal identity views only. Shape/rank-changing packing and SDK guards retained. Summed extents are not measured copy traffic; actual backend, memory, recurrence and metrics separate.'))


if __name__ == '__main__':
    main()
