"""Stream deformable-attention queries, preserving point order and FP32 reduction."""
import argparse
import copy
import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation'))
import shape_dataflow as engine
from prove_graph_extents import sha, save
from attention_regions import attention_regions


def streamed_nodes(grids, features, weights, output, shape, chunk_queries, prefix, attributes, packed_multiply=True):
    batch, channels, queries, points = shape
    nodes = []
    ordinal = 0

    def op(kind, inputs, **attrs):
        nonlocal ordinal
        name = prefix + '/' + str(ordinal)
        ordinal += 1
        nodes.append(H.make_node(kind, inputs, [name], name=name, **attrs))
        return name

    def constant(values):
        return op('Constant', [], value=N.from_array(np.array(values, dtype=np.int64)))

    def view(value, dims):
        return op('Reshape', [value, constant(dims)])

    reduced = []
    for start in range(0, queries, chunk_queries):
        end = min(start + chunk_queries, queries)
        width = end - start
        first, last = constant([start]), constant([end])
        grid_axis, weight_axis = constant([1]), constant([2])
        sampled = []
        for grid, feature in zip(grids, features):
            sliced = op('Slice', [grid, first, last, grid_axis])
            sampled.append(op('GridSample', [feature, sliced], **attributes))
        attention = op('Slice', [weights, first, last, weight_axis])
        values = op('Concat', sampled, axis=-1)
        if packed_multiply:
            values = view(values, [batch, channels, width * points])
            attention = view(attention, [batch, 1, width * points])
        product = op('Mul', [values, attention])
        if packed_multiply:
            product = view(product, [batch, channels, width, points])
        reduced.append(op('ReduceSum', [product, constant([-1])], keepdims=0))
    nodes.append(H.make_node('Concat', reduced, [output], name=prefix + '/reduced', axis=2))
    return nodes


def rewrite(model, chunk_queries, expected_regions=6, packed_multiply=True):
    if chunk_queries < 1:
        raise ValueError('positive query block required')
    types, constants, errors = engine.prove_static_dataflow(model)
    if errors:
        raise ValueError('source root inference failed')
    regions = attention_regions(model, types, constants)
    if len(regions) != expected_regions:
        raise ValueError('expected attention regions ' + str(expected_regions) + ', found ' + str(len(regions)))
    replacements, removed, changes = {}, set(), []
    nodes = list(model.graph.node)
    for region in regions:
        reduce_index = max(region['node_indices'])
        reduce = nodes[reduce_index]
        if reduce.op_type != 'ReduceSum' or reduce.output[0] != region['output']:
            raise ValueError('sampling region anchor is not topologically last')
        shape = region['shape']
        prefix = (reduce.name or 'attention_' + str(reduce_index)) + '/query_stream'
        replacements[reduce_index] = streamed_nodes(region['grids'], region['features'], region['weights'],
                                                   region['output'], shape, chunk_queries, prefix,
                                                   region['attributes'], packed_multiply=packed_multiply)
        removed.update(region['node_indices'])
        changes.append(dict(node=reduce.name, source_shape=list(shape), query_chunk=chunk_queries,
                            chunks=(shape[2] + chunk_queries - 1) // chunk_queries,
                            original_largest_intermediate_bytes=int(np.prod(shape))*4,
                            candidate_largest_intermediate_bytes=shape[0]*shape[1]*min(shape[2], chunk_queries)*shape[3]*4,
                            reduction_order_unchanged=True, source_node_indices=region['node_indices'],
                            source_node_sha256=region['node_sha256'], source_adapter=region['adapter'],
                            emission_profile='cpu_rank3_broadcast' if packed_multiply else 'standard_onnx'))
    names = {v for n in model.graph.node for v in n.output} | {v.name for v in model.graph.initializer}
    new_values = [v for chain in replacements.values() for n in chain[:-1] for v in n.output]
    if len(set(new_values)) != len(new_values) or names.intersection(new_values):
        raise ValueError('streamed value names collide')
    candidate = []
    for index, node in enumerate(nodes):
        if index in replacements:
            candidate.extend(replacements[index])
        elif index not in removed:
            candidate.append(copy.deepcopy(node))
    original = copy.deepcopy(model)
    del model.graph.node[:]
    model.graph.node.extend(candidate)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    for boundary in ('input', 'output', 'initializer'):
        if [v.SerializeToString() for v in getattr(model.graph, boundary)] != [v.SerializeToString() for v in getattr(original.graph, boundary)]:
            raise ValueError('source ABI or weights changed')
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--chunk-queries', type=int, default=2048)
    parser.add_argument('--standard-multiply', action='store_true', help='Emit standard rank-four multiply without the CPU broadcast adapter')
    args = parser.parse_args()
    if not 1 <= args.chunk_queries <= 10201 or sha(args.model) != args.model_sha256:
        raise ValueError('source identity or chunk limit differs')
    model = onnx.load(str(args.model))
    changes = rewrite(model, args.chunk_queries, packed_multiply=not args.standard_multiply)
    output = Path.cwd() / 'model.query-streamed.onnx'
    onnx.save(model, str(output))
    save(Path.cwd() / 'query_streaming.json', dict(status='pass', source_model=str(args.model), source_model_sha256=args.model_sha256, model=str(output), model_sha256=sha(output), script_sha256=sha(__file__), engine_sha256=sha(engine.__file__), regions_sha256=sha(Path(__file__).with_name('attention_regions.py')), changes=changes, ordered_abi_unchanged=True, original_weights_unchanged=True, scope='FP32 query-axis streaming only; actual backend execution, recurrence, metrics and measured peak memory require independent controls.'))


if __name__ == '__main__':
    main()
