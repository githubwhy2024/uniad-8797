"""Recognize sampling/weight/reduction by data flow, independent of CPU names."""
import hashlib
import math
import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N


def static_types(model):
    values = {v.name: v.type for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)}
    for value in model.graph.initializer:
        values[value.name] = H.make_tensor_type_proto(value.data_type, list(value.dims))
    return values


def extents(types, name):
    tensor = types.get(name)
    if tensor is None or not tensor.tensor_type.HasField('shape'):
        return None
    dims = tensor.tensor_type.shape.dim
    if any(not d.HasField('dim_value') for d in dims):
        return None
    return tuple(d.dim_value for d in dims)


def small_constants(model, limit=4096):
    result = {}
    for value in model.graph.initializer:
        if math.prod(value.dims) <= limit:
            result[value.name] = N.to_array(value)
    for node in model.graph.node:
        if node.op_type == 'Constant':
            value = next((a.t for a in node.attribute if a.name == 'value'), None)
            if value is not None and math.prod(value.dims) <= limit:
                result[node.output[0]] = N.to_array(value)
    return result


def attention_regions(model, types, constants):
    """Match direct standard ONNX or an exact rank-three broadcast adapter.

    Names locate results only. Four GridSamples concatenated in point order,
    FP32 broadcasting and the last-axis sum are proven from edges and extents.
    A removable intermediate must be private to this region. Public result
    consumers stay intact. Unknown policies or shapes are not guessed.
    """
    nodes = list(model.graph.node)
    producer = {v: i for i, n in enumerate(nodes) for v in n.output}
    consumers = {}
    for index, node in enumerate(nodes):
        for value in node.input:
            if value:
                consumers.setdefault(value, set()).add(index)
    public = {v.name for v in model.graph.output}

    def node_for(value):
        index = producer.get(value)
        return (index, nodes[index]) if index is not None else (None, None)

    rows = []
    for reduce_index, reduce in enumerate(nodes):
        if reduce.op_type != 'ReduceSum' or len(reduce.input) != 2 or len(reduce.output) != 1:
            continue
        shape = extents(types, reduce.input[0])
        if shape is None or len(shape) != 4 or any(d <= 0 for d in shape):
            continue
        batch, channels, queries, points = shape
        axes = constants.get(reduce.input[1])
        attrs = {a.name: H.get_attribute_value(a) for a in reduce.attribute}
        if axes is None or np.asarray(axes).reshape(-1).tolist() not in ([-1], [3]) or attrs.get('keepdims', 1) != 0 or set(attrs) - {'keepdims', 'noop_with_empty_axes'}:
            continue
        if extents(types, reduce.output[0]) != shape[:3]:
            continue
        chain = [reduce_index]
        index, multiply = node_for(reduce.input[0])
        packed = multiply is not None and multiply.op_type == 'Reshape'
        if packed:
            if extents(types, multiply.input[0]) != (batch, channels, queries * points):
                continue
            chain.append(index)
            index, multiply = node_for(multiply.input[0])
        if multiply is None or multiply.op_type != 'Mul' or len(multiply.input) != 2 or multiply.attribute:
            continue
        chain.append(index)
        operands = []
        operand_views = []
        for value in multiply.input:
            if packed:
                view_index, view = node_for(value)
                if view is None or view.op_type != 'Reshape' or len(view.input) != 2:
                    break
                before = extents(types, view.input[0])
                after = extents(types, value)
                if before not in (shape, (batch, 1, queries, points)) or after != (before[0], before[1], queries * points):
                    break
                operands.append(view.input[0]); operand_views.append(view_index)
            else:
                operands.append(value)
        if len(operands) != 2:
            continue
        sampled = [v for v in operands if extents(types, v) == shape]
        weights = [v for v in operands if extents(types, v) == (batch, 1, queries, points)]
        # With C=1 both shapes coincide. Resolve by the actual Concat producer.
        matches = [(v, producer.get(v)) for v in sampled if producer.get(v) is not None and nodes[producer[v]].op_type == 'Concat']
        if len(matches) != 1:
            continue
        sampled_value, concat_index = matches[0]
        weight_candidates = [v for v in weights if v != sampled_value]
        if len(weight_candidates) != 1:
            continue
        weight = weight_candidates[0]
        concat = nodes[concat_index]
        concat_attrs = {a.name: H.get_attribute_value(a) for a in concat.attribute}
        if concat_attrs != {'axis': -1} and concat_attrs != {'axis': 3}:
            continue
        if len(concat.input) != 4 or points != 32:
            continue
        samples = []
        for value in concat.input:
            sample_index, sample = node_for(value)
            if sample is None or sample.op_type != 'GridSample' or len(sample.input) != 2 or len(sample.output) != 1:
                break
            feature_shape = extents(types, sample.input[0])
            if feature_shape is None or len(feature_shape) != 4 or feature_shape[:2] != (batch, channels) or any(d <= 0 for d in feature_shape):
                break
            if extents(types, sample.input[1]) != (batch, queries, 8, 2) or extents(types, value) != (batch, channels, queries, 8):
                break
            samples.append((sample_index, sample))
        if len(samples) != 4:
            continue
        attributes = {a.name: H.get_attribute_value(a) for a in samples[0][1].attribute}
        if set(attributes) - {'mode', 'padding_mode', 'align_corners'}:
            raise ValueError('unknown sampling attribute in recognized region')
        if attributes.get('mode', b'bilinear') != b'bilinear' or attributes.get('padding_mode', b'zeros') not in (b'zeros', b'border', b'reflection') or attributes.get('align_corners', 0) not in (0, 1):
            raise ValueError('sampling policy outside verified region')
        if any({a.name: H.get_attribute_value(a) for a in sample.attribute} != attributes for _, sample in samples):
            raise ValueError('sampling policy differs between levels')
        chain += operand_views + [concat_index] + [i for i, _ in samples]
        selected = set(chain)
        if len(selected) != len(chain):
            raise ValueError('sampling region overlaps itself')
        for index in selected - {reduce_index}:
            for value in nodes[index].output:
                if value in public or consumers.get(value, set()) - selected:
                    raise ValueError('sampling intermediate has another consumer: ' + value)
        data_values = [weight] + [v for _, n in samples for v in list(n.input) + list(n.output)] + list(multiply.output) + [reduce.input[0], reduce.output[0]]
        if any(types[v].tensor_type.elem_type != onnx.TensorProto.FLOAT for v in data_values):
            raise ValueError('recognized sampling region must retain FP32 data')
        rows.append(dict(anchor=reduce.name, output=reduce.output[0], shape=shape, weights=weight,
                         features=[n.input[0] for _, n in samples], grids=[n.input[1] for _, n in samples],
                         attributes=attributes, node_indices=sorted(selected), adapter='rank3_broadcast_views' if packed else 'standard_onnx',
                         node_sha256=[hashlib.sha256(nodes[i].SerializeToString()).hexdigest() for i in sorted(selected)],
                         float_point_order_unchanged=True))
    claimed = [i for row in rows for i in row['node_indices']]
    if len(set(claimed)) != len(claimed):
        raise ValueError('recognized sampling regions overlap')
    return rows
