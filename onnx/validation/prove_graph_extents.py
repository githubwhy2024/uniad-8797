#!/usr/bin/env python3
"""Certify a fixed graph from ABI roots; output metadata is never an axiom."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnx
from onnx import helper as H
import onnxruntime as ort

import shape_dataflow as engine


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    compact = Path(path).name in ('extent_certificate.json', 'shape_control_values.json', 'control_arithmetic_comparison.json', 'shape_bindings.json')
    Path(path).write_text(json.dumps(data, indent=None if compact else 2, separators=(',', ':') if compact else None, allow_nan=False) + '\n')


def checkpoint(stage, **fields):
    print(json.dumps(dict(stage=stage, **fields)), flush=True)


def controls():
    checks = engine.verify_rejection_and_schema_rules()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    for name, dims in {'matrix': [2, 3], 'scalar': [], 'zero_extent': [0, 3], 'rank3': [7, 11, 3]}.items():
        model = engine.checked_model([H.make_node('Size', ['x'], ['size'])],
                                     [H.make_tensor_value_info('x', onnx.TensorProto.FLOAT, dims)],
                                     [H.make_tensor_value_info('size', onnx.TensorProto.INT64, [])])
        types, constants, errors = engine.prove_static_dataflow(model)
        actual = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider']).run(None, {'x': np.zeros(dims, np.float32)})[0]
        checks['size_' + name + '_actual_ort'] = not errors and np.array_equal(constants['size'], actual) and actual.dtype == np.int64 and engine.fixed_extents(types['size']) == ()
    model = engine.checked_model([H.make_node('Identity', ['x'], ['fake_static']), H.make_node('Size', ['fake_static'], ['size'])],
                                 [H.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [None, 3])],
                                 [H.make_tensor_value_info('size', onnx.TensorProto.INT64, [])])
    model.graph.value_info.append(H.make_tensor_value_info('fake_static', onnx.TensorProto.FLOAT, [2, 3]))
    types, constants, errors = engine.prove_static_dataflow(model)
    checks['size_runtime_value_rejected_despite_spoofed_metadata'] = not errors and 'size' not in constants and engine.fixed_extents(types['size']) == ()
    model = engine.checked_model([H.make_node('Size', ['x'], ['size'])],
                                 [H.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [2 ** 32, 2 ** 32])],
                                 [H.make_tensor_value_info('size', onnx.TensorProto.INT64, [])])
    _, _, errors = engine.prove_static_dataflow(model)
    checks['size_int64_product_overflow_rejected'] = any('int64 extent range' in v['error'] for v in errors)
    if not all(checks.values()):
        raise ValueError('root/Size control rejected: ' + str(checks))
    return checks


def prove(args):
    if onnx.__version__ != '1.13.1':
        raise ValueError('proof schema runtime must remain ONNX1.13.1')
    run = Path.cwd()
    if sha(args.model) != args.model_sha256:
        raise ValueError('source model identity differs')
    source = onnx.load(str(args.model))
    onnx.checker.check_model(source)
    save(run / 'counterexample_checks.json', dict(status='pass', checks=controls()))
    checkpoint('counterexamples_accepted')
    stripped = copy.deepcopy(source)
    del stripped.graph.value_info[:]
    for value in stripped.graph.output:
        value.type.tensor_type.ClearField('shape')
    types, constants, errors = engine.prove_static_dataflow(stripped, checkpoint)
    produced = {name: node for node in source.graph.node for name in node.output if name}
    unresolved = [dict(name=name, producer=node.name, op_type=node.op_type)
                  for name, node in produced.items() if engine.fixed_extents(types.get(name, onnx.TypeProto())) is None]
    control_rows = [dict(node=node.name, op_type=node.op_type, operand=i, tensor=node.input[i],
                         dtype=str(constants[node.input[i]].dtype) if node.input[i] in constants else None,
                         value=constants[node.input[i]].tolist() if node.input[i] in constants else None)
                    for node in source.graph.node for i in engine.CONTROLLERS.get(node.op_type, [])
                    if i < len(node.input) and node.input[i]]
    unknown_controls = [row for row in control_rows if row['tensor'] not in constants]
    contradictions = []
    for value in list(source.graph.value_info) + list(source.graph.output):
        inferred = types.get(value.name)
        if inferred is None:
            continue
        actual = engine.fixed_extents(inferred)
        declared = value.type.tensor_type
        if declared.elem_type and declared.elem_type != inferred.tensor_type.elem_type:
            contradictions.append(dict(name=value.name, kind='dtype'))
        if declared.HasField('shape') and actual is not None:
            dims = declared.shape.dim
            if len(dims) != len(actual) or any(d.HasField('dim_value') and d.dim_value != actual[i] for i, d in enumerate(dims)):
                contradictions.append(dict(name=value.name, kind='extent'))
    save(run / 'proof_residual.json', dict(unresolved=unresolved, unresolved_controls=unknown_controls,
                                         errors=errors, contradictions=contradictions))
    save(run / 'shape_control_values.json', dict(controls=control_rows, unresolved_controls=unknown_controls))
    if unresolved or unknown_controls or errors or contradictions:
        raise ValueError('root proof incomplete: ' + str(dict(unresolved=len(unresolved), controls=len(unknown_controls), errors=len(errors), contradictions=len(contradictions))))
    checkpoint('executing_pure_control_arithmetic', operand_slots=len(control_rows))
    arithmetic = engine.verify_control_arithmetic(source, control_rows, types, constants, run, sys.modules[__name__])
    certificate = {name: dict(shape=list(engine.fixed_extents(types[name])),
                              dtype=onnx.TensorProto.DataType.Name(types[name].tensor_type.elem_type),
                              producer=node.name, op_type=node.op_type) for name, node in produced.items()}
    save(run / 'extent_certificate.json', dict(roots='Graph input ABI, initializer and Constant only; no original internal/output metadata.', extents=certificate))
    derived = copy.deepcopy(source)
    del derived.graph.value_info[:]
    boundary = {v.name for v in list(derived.graph.input) + list(derived.graph.output)}
    for name in produced:
        if name not in boundary:
            value = onnx.ValueInfoProto(name=name)
            value.type.CopyFrom(types[name])
            derived.graph.value_info.append(value)
    a, b = copy.deepcopy(source), copy.deepcopy(derived)
    del a.graph.value_info[:]
    del b.graph.value_info[:]
    if a.SerializeToString() != b.SerializeToString():
        raise ValueError('annotation changed computation or boundary ABI')
    onnx.checker.check_model(derived, full_check=True)
    path = run / 'model.shape-certified.onnx'
    onnx.save(derived, str(path))
    proof = dict(status='pass', execution_status='complete', source_model=str(args.model),
                 source_model_sha256=args.model_sha256, derived_model=str(path), derived_model_sha256=sha(path),
                 inputs=len(source.graph.input), outputs=len(source.graph.output), nodes=len(source.graph.node),
                 produced_values=len(produced), all_node_output_extents_proven=True,
                 shape_control_slots=len(control_rows), all_control_values_proven_constant=True,
                 all_actual_control_arithmetic_exact=True, arithmetic=arithmetic,
                 certificate_sha256=sha(run / 'extent_certificate.json'), controls_sha256=sha(run / 'shape_control_values.json'),
                 counterexamples_sha256=sha(run / 'counterexample_checks.json'), residual_sha256=sha(run / 'proof_residual.json'),
                 engine_sha256=sha(engine.__file__), script_sha256=sha(__file__),
                 upstream_source=engine.UPSTREAM_SOURCE, upstream_source_sha256=engine.UPSTREAM_SHA256,
                 onnx_version=onnx.__version__, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 metadata_only=True, checker_full_check=True,
                 scope='Root-only static certificate and pure control execution; no neural QNN/recurrence/task acceptance.')
    save(run / 'shape_proof.json', proof)
    checkpoint('root_proof_complete', outputs=proof['outputs'], produced_values=proof['produced_values'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    args = parser.parse_args()
    args.model = args.model.absolute()
    prove(args)


if __name__ == '__main__':
    main()
