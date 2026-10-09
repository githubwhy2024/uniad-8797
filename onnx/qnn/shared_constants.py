"""Fold repeated immutable operands once, preserving the public graph ABI."""
import hashlib
import math
from pathlib import Path
import numpy as np
from onnx import helper as H, numpy_helper as N

VIEW_OPS = {'Identity', 'Reshape', 'Transpose', 'Expand', 'Squeeze', 'Unsqueeze'}


def pool_values(model, types, constants, minimum_bytes):
    initial = {v.name: v for v in model.graph.initializer}
    producer = {v: n for n in model.graph.node for v in n.output}
    immutable = set(initial)
    for n in model.graph.node:
        if n.op_type == 'Constant' or (n.input and all(v in immutable for v in n.input if v) and n.op_type not in ('RandomNormal', 'RandomUniform', 'RandomNormalLike', 'RandomUniformLike', 'Multinomial', 'Dropout')):
            immutable.update(n.output)
    operands = {v for n in model.graph.node if any(o not in immutable for o in n.output) for v in n.input if v in immutable}
    selected = []
    for name in sorted(operands):
        t = types[name].tensor_type
        shape = tuple(d.dim_value for d in t.shape.dim)
        if t.elem_type == H.TensorProto.FLOAT and math.prod(shape)*4 >= minimum_bytes:
            selected.append(name)
    if not selected:
        return {}
    # Evaluate only the deterministic initializer/Constant closure. No public
    # frame/state input is admitted, and no neural graph is used as a fallback.
    import copy
    import onnx
    import onnxruntime as ort
    if ort.__version__ != '1.19.2':
        raise ValueError('immutable folding ORT runtime differs')
    pending = list(selected)
    needed = set(selected)
    kept = set()
    while pending:
        name = pending.pop()
        if name in initial:
            continue
        node = producer[name]
        if node.name in kept:
            continue
        if any(v not in immutable for v in node.input if v):
            raise ValueError('immutable circuit depends on a frame input')
        kept.add(node.name)
        for value in node.input:
            if value and value not in needed:
                needed.add(value)
                pending.append(value)
    graph = H.make_graph([copy.deepcopy(n) for n in model.graph.node if n.name in kept], 'immutable-operand-circuit', [], [H.make_value_info(n, types[n]) for n in selected], [copy.deepcopy(t) for n,t in initial.items() if n in needed])
    circuit = H.make_model(graph, opset_imports=list(model.opset_import))
    circuit.ir_version = model.ir_version
    onnx.checker.check_model(circuit, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(circuit.SerializeToString(), options, providers=['CPUExecutionProvider'])
    values = dict(zip(selected, session.run(None, {})))
    for name, value in values.items():
        expected = tuple(d.dim_value for d in types[name].tensor_type.shape.dim)
        if value.shape != expected or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('immutable operand contract differs: '+name)
    # Share only repeated byte-identical constant data. Single-use large
    # tensors keep their original native representation and lifetime.
    groups={}
    for name,value in values.items():
        key=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        groups.setdefault(key,[]).append(name)
    repeated=set()
    for names in groups.values():
        uses=sum(sum(name in n.input for n in model.graph.node if any(v not in immutable for v in n.output)) for name in names)
        if uses>1:repeated.update(names)
    return {name:value for name,value in values.items() if name in repeated}


def pack_pool(values, path, minimum_bytes):
    arrays, rows = {}, []
    for name, value in values.items():
        flat = np.ascontiguousarray(value).reshape(-1)
        digest = hashlib.sha256(flat.tobytes()).hexdigest()
        key = 'sha_' + digest
        if key not in arrays:
            arrays[key] = flat
        rows.append(dict(source_name=name, key=key, shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes, data_sha256=digest))
    if not rows:
        raise ValueError('no large immutable operand selected')
    np.savez_compressed(path, **arrays)
    return dict(schema='immutable-operand-pool-v1', path=str(Path(path).absolute()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), minimum_bytes=minimum_bytes, entries=rows, unique_data_bytes=sum(v.nbytes for v in arrays.values()), logical_view_bytes=sum(r['bytes'] for r in rows))


def verify_pool(model, types, constants, report):
    if not isinstance(report['minimum_bytes'],int) or report['minimum_bytes']<1:raise ValueError('immutable threshold differs')
    path = Path(report['path'])
    if report['schema'] != 'immutable-operand-pool-v1' or hashlib.sha256(path.read_bytes()).hexdigest() != report['sha256']:
        raise ValueError('immutable pool resource differs')
    expected = pool_values(model, types, constants, report['minimum_bytes'])
    rows = {r['source_name']: r for r in report['entries']}
    if len(rows) != len(report['entries']) or set(rows) != set(expected):
        raise ValueError('immutable root selection differs')
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {r['key'] for r in rows.values()}:
            raise ValueError('immutable pool key closure differs')
        for name, value in expected.items():
            row = rows[name]
            data = archive[row['key']]
            if row['shape'] != list(value.shape) or row['dtype'] != str(value.dtype) or row['bytes'] != value.nbytes or data.dtype != value.dtype or data.shape != (value.size,):
                raise ValueError('immutable pool view ABI differs')
            digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            if digest != row['data_sha256'] or row['key'] != 'sha_'+digest or not np.array_equal(data, value.reshape(-1)):
                raise ValueError('immutable pool differs from original deterministic roots')
        if report['unique_data_bytes'] != sum(archive[k].nbytes for k in archive.files) or report['logical_view_bytes'] != sum(r['bytes'] for r in rows.values()):
            raise ValueError('immutable pool size accounting differs')
    return rows
