#!/usr/bin/env python3
"""Replay certificates without metadata and independently execute their controls."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

import shape_dataflow as engine
from prove_graph_extents import sha, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.proof_dir.absolute()
    proof = json.loads((run / 'shape_proof.json').read_text())
    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise ValueError('extent audit rejected: ' + name)
    check('proof_complete', proof['status'] == 'pass' and proof['execution_status'] == 'complete')
    check('source_hash', sha(proof['source_model']) == proof['source_model_sha256'])
    check('derived_hash', sha(proof['derived_model']) == proof['derived_model_sha256'])
    check('engine_hash', sha(engine.__file__) == proof['engine_sha256'])
    for name, field in {'extent_certificate.json': 'certificate_sha256', 'shape_control_values.json': 'controls_sha256',
                        'counterexample_checks.json': 'counterexamples_sha256', 'proof_residual.json': 'residual_sha256'}.items():
        check(name + '_hash', sha(run / name) == proof[field])
    negative = json.loads((run / 'counterexample_checks.json').read_text())
    check('counterexamples_and_Size_actual_checks', len(negative['checks']) == 10 and all(negative['checks'].values()))
    residual = json.loads((run / 'proof_residual.json').read_text())
    check('no_residual', not any(residual.values()))
    source = onnx.load(proof['source_model'])
    derived = onnx.load(proof['derived_model'])
    onnx.checker.check_model(derived, full_check=True)
    check('full_checker', True)
    a, b = copy.deepcopy(source), copy.deepcopy(derived)
    del a.graph.value_info[:]
    del b.graph.value_info[:]
    check('entire_model_unchanged_except_internal_metadata', a.SerializeToString() == b.SerializeToString())
    for value in a.graph.output:
        value.type.tensor_type.ClearField('shape')
    types, constants, errors = engine.prove_static_dataflow(a)
    check('root_replay_no_errors', not errors)
    certificate = json.loads((run / 'extent_certificate.json').read_text())['extents']
    produced = {name for node in source.graph.node for name in node.output if name}
    check('every_produced_value_certified', set(certificate) == produced and len(produced) == proof['produced_values'])
    check('root_replay_all_extents_dtypes', all(engine.fixed_extents(types.get(name, onnx.TypeProto())) == tuple(row['shape'])
          and onnx.TensorProto.DataType.Name(types[name].tensor_type.elem_type) == row['dtype'] for name, row in certificate.items()))
    annotation = {v.name: v for v in derived.graph.value_info}
    boundary = {v.name for v in list(source.graph.input) + list(source.graph.output)}
    check('all_internal_values_annotated', set(annotation) == produced - boundary)
    check('annotation_matches_certificate', all(engine.fixed_extents(v.type) == tuple(certificate[name]['shape'])
          and onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type) == certificate[name]['dtype'] for name, v in annotation.items()))
    rows = json.loads((run / 'shape_control_values.json').read_text())['controls']
    expected = {(n.name, i, n.input[i]) for n in source.graph.node for i in engine.CONTROLLERS.get(n.op_type, []) if i < len(n.input) and n.input[i]}
    check('all_control_operand_slots', expected == {(r['node'], r['operand'], r['tensor']) for r in rows} and len(rows) == proof['shape_control_slots'])
    check('all_control_values_from_roots', all(r['tensor'] in constants and str(constants[r['tensor']].dtype) == r['dtype'] and constants[r['tensor']].tolist() == r['value'] for r in rows))
    arithmetic = json.loads((run / 'control_arithmetic_comparison.json').read_text())
    check('arithmetic_artifacts_bound', sha(run / 'control_arithmetic_comparison.json') == proof['arithmetic']['comparison_sha256']
          and sha(run / 'shape_control_arithmetic.onnx') == arithmetic['model_sha256'] == proof['arithmetic']['model_sha256'])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(run / 'shape_control_arithmetic.onnx'), options, providers=['CPUExecutionProvider'])
    check('control_graph_has_no_runtime_inputs', not session.get_inputs())
    check('all_controls_actually_executed', {v.name for v in session.get_outputs()} == {r['tensor'] for r in rows})
    values = session.run(None, {})
    check('independent_ort_integer_control_values', all(value.dtype == constants[info.name].dtype and value.shape == constants[info.name].shape
          and np.array_equal(value, constants[info.name]) for info, value in zip(session.get_outputs(), values)))
    save(Path.cwd() / 'audit.json', dict(status='pass', execution_status='complete', checks=checks,
         shape_proof_sha256=sha(run / 'shape_proof.json'), derived_model_sha256=proof['derived_model_sha256'],
         auditor_sha256=sha(__file__), scope='Root replay, metadata equality and actual pure controls only; no neural QNN/task acceptance.'))
    print(json.dumps(dict(status='pass', checks=len(checks), produced_values=len(produced))))


if __name__ == '__main__':
    main()
