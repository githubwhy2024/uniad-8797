#!/usr/bin/env python3
"""Prepare root-proven static control candidates for the installed QNN converter."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation'))
import shape_dataflow as engine
from prove_graph_extents import sha, save


def bind_shape_controls(model, types, constants):
    original = copy.deepcopy(model)
    aliases, bindings, additional = {}, [], []
    existing = {v.name for v in model.graph.initializer} | {name for node in model.graph.node for name in list(node.input) + list(node.output)}
    rows = []
    for node in model.graph.node:
        for i in engine.CONTROLLERS.get(node.op_type, []):
            if i >= len(node.input) or not node.input[i]:
                continue
            name = node.input[i]
            if name not in constants:
                raise ValueError('shape control is not root-constant: ' + name)
            if name not in aliases:
                alias = '_qnn_shape_control_' + str(len(aliases))
                if alias in existing:
                    raise ValueError('generated shape-control name collides')
                aliases[name] = alias
                additional.append(N.from_array(constants[name], name=alias))
            bindings.append(dict(node=node.name, op_type=node.op_type, operand=i, original_tensor=name,
                                 initializer=aliases[name], dtype=str(constants[name].dtype),
                                 value=constants[name].tolist()))
            rows.append(dict(node=node.name, op_type=node.op_type, operand=i, tensor=name))
            node.input[i] = aliases[name]
    # Execute the original, root-derived control circuit before using its values.
    arithmetic = engine.verify_control_arithmetic(original, rows, types, constants, Path.cwd(), sys.modules[__name__])
    model.graph.initializer.extend(additional)
    needed = {v.name for v in model.graph.output}
    selected = []
    for node in reversed(model.graph.node):
        if needed.intersection(node.output):
            selected.append(copy.deepcopy(node))
            needed.update(v for v in node.input if v)
    selected.reverse()
    removed = [dict(name=node.name, op_type=node.op_type, outputs=list(node.output)) for node in model.graph.node
               if not needed.intersection(node.output)]
    del model.graph.node[:]
    model.graph.node.extend(selected)
    produced = {name for node in selected for name in node.output}
    retained_info = [copy.deepcopy(v) for v in model.graph.value_info if v.name in produced]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_info)
    originals = {v.name: v for v in original.graph.initializer}
    if any(v.SerializeToString() != originals[v.name].SerializeToString() for v in model.graph.initializer if v.name in originals):
        raise ValueError('binding altered original initializer bytes')
    before = {node.name: node for node in original.graph.node}
    binding_table = {(row['node'], row['operand']): row for row in bindings}
    for node in selected:
        expected = copy.deepcopy(before[node.name])
        for i in range(len(expected.input)):
            row = binding_table.get((node.name, i))
            if row:
                expected.input[i] = row['initializer']
        if node.SerializeToString() != expected.SerializeToString():
            raise ValueError('non-control computation changed: ' + node.name)
    save(Path.cwd() / 'shape_bindings.json', dict(bindings=bindings, removed_nodes=removed, arithmetic=arithmetic))
    return dict(control_operand_bindings=len(bindings), control_initializers=len(additional),
                removed_dead_nodes=len(removed), source_nodes=len(original.graph.node), arithmetic=arithmetic,
                shape_bindings_sha256=sha(Path.cwd() / 'shape_bindings.json'), retained_noncontrol_nodes_unchanged=True,
                original_initializer_bytes_unchanged=True)


def fold_reduceprod(model, constants):
    nodes, changes = [], []
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    for node in model.graph.node:
        if node.op_type != 'ReduceProd' or len(node.output) != 1 or node.output[0] not in constants or constants[node.output[0]].dtype != np.int64:
            nodes.append(copy.deepcopy(node))
            continue
        if any(name and name not in constants for name in node.input):
            raise ValueError('ReduceProd output lacks root-derived input values: ' + node.name)
        initializers = [N.from_array(constants[name], name=name) for name in node.input if name]
        output = H.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, list(constants[node.output[0]].shape))
        control = engine.checked_model([copy.deepcopy(node)], [], [output], initializers)
        actual = ort.InferenceSession(control.SerializeToString(), options, providers=['CPUExecutionProvider']).run(None, {})[0]
        if actual.dtype != np.int64 or not np.array_equal(actual, constants[node.output[0]]):
            raise ValueError('integer reduction control differs from actual ORT: ' + node.name)
        replacement = H.make_node('Constant', [], list(node.output), name=node.name + '/root_folded', value=N.from_array(actual))
        nodes.append(replacement)
        changes.append(dict(node=node.name, output=node.output[0], dtype='int64', shape=list(actual.shape),
                            value=actual.tolist(), original_node_sha256=hashlib.sha256(node.SerializeToString()).hexdigest(),
                            actual_integer_control_exact=True))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return changes


def prune_empty_concat(model, types):
    changes = []
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    for node in model.graph.node:
        if node.op_type != 'Concat':
            continue
        shapes = [engine.fixed_extents(types[name]) for name in node.input]
        if any(shape is None for shape in shapes):
            raise ValueError('Concat extent lacks root proof')
        axis = next(a.i for a in node.attribute if a.name == 'axis') % len(shapes[0])
        empty = [i for i, shape in enumerate(shapes) if shape[axis] == 0]
        if not empty:
            continue
        kept = [i for i in range(len(shapes)) if i not in empty]
        if not kept:
            raise ValueError('all-empty Concat is unsupported')
        if any(len(shape) != len(shapes[0]) or any(v != shapes[0][j] for j, v in enumerate(shape) if j != axis) for shape in shapes):
            raise ValueError('empty Concat operand differs outside concatenation axis')
        output_infos = [H.make_value_info(name, types[name]) for name in node.output]
        input_infos = [H.make_value_info(name, types[name]) for name in node.input]
        before = engine.checked_model([copy.deepcopy(node)], input_infos, output_infos, [])
        reduced = copy.deepcopy(node)
        del reduced.input[:]
        reduced.input.extend(node.input[i] for i in kept)
        after = engine.checked_model([reduced], [input_infos[i] for i in kept], output_infos, [])
        sessions = [ort.InferenceSession(m.SerializeToString(), options, providers=['CPUExecutionProvider']) for m in (before, after)]
        for case in ('zeros', 'values', 'nonfinite'):
            feed = {}
            for name, shape in zip(node.input, shapes):
                dtype = H.tensor_dtype_to_np_dtype(types[name].tensor_type.elem_type)
                value = np.arange(np.prod(shape), dtype=dtype).reshape(shape) if case != 'zeros' else np.zeros(shape, dtype=dtype)
                if case == 'nonfinite' and value.size and np.issubdtype(dtype, np.floating):
                    value.flat[0] = np.nan
                    if value.size > 1:
                        value.flat[1] = np.inf
                feed[name] = value
            a = sessions[0].run(None, feed)
            b = sessions[1].run(None, {name: feed[name] for name in reduced.input})
            if any(x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b)):
                raise ValueError('actual empty concat identity failed')
        changes.append(dict(node=node.name, original_inputs=list(node.input), retained_inputs=list(reduced.input),
                            root_shapes=shapes, axis=axis, actual_ort_cases=['zeros', 'values', 'nonfinite']))
        del node.input[:]
        node.input.extend(reduced.input)
    if not changes:
        raise ValueError('no empty Concat operand matched observed converter fault')
    needed = {v.name for v in model.graph.output}
    selected = []
    for node in reversed(model.graph.node):
        if needed.intersection(node.output):
            selected.append(copy.deepcopy(node))
            needed.update(v for v in node.input if v)
    selected.reverse()
    removed = [node.name for node in model.graph.node if not needed.intersection(node.output)]
    del model.graph.node[:]
    model.graph.node.extend(selected)
    produced = {name for node in selected for name in node.output}
    infos = [copy.deepcopy(v) for v in model.graph.value_info if v.name in produced]
    del model.graph.value_info[:]
    model.graph.value_info.extend(infos)
    return dict(empty_concat_changes=changes, empty_branch_removed_nodes=removed)


def tile_expands(model, types):
    changes, controls, nodes, weights = [], {}, [], []
    existing = {v for n in model.graph.node for v in list(n.input)+list(n.output)}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.log_severity_level = 3
    for node in model.graph.node:
        if node.op_type != 'Expand':
            nodes.append(copy.deepcopy(node))
            continue
        source = engine.fixed_extents(types[node.input[0]])
        target = engine.fixed_extents(types[node.output[0]])
        if source is None or target is None or len(source)>len(target):
            raise ValueError('Expand extent is not a proven broadcast')
        padded = (1,)*(len(target)-len(source))+source
        if any(a not in (1,b) for a,b in zip(padded,target)):
            raise ValueError('Expand dimension is not broadcastable')
        repeats = [b if a==1 else 1 for a,b in zip(padded,target)]
        prefix = node.name+'/qnn_tile'
        view = prefix+'_view'
        shape = prefix+'_shape'
        repeat_name = prefix+'_repeats'
        if {view,shape,repeat_name}&existing:
            raise ValueError('Tile replacement names collide')
        existing.update((view,shape,repeat_name))
        new = []
        if not target:
            new.append(H.make_node('Identity',[node.input[0]],list(node.output),name=prefix))
        else:
            weights.extend([N.from_array(np.array(padded,np.int64),name=shape),N.from_array(np.array(repeats,np.int64),name=repeat_name)])
            new.extend([H.make_node('Reshape',[node.input[0],shape],[view],name=prefix+'/view'),H.make_node('Tile',[view,repeat_name],list(node.output),name=prefix+'/tile')])
        key = (source,target)
        if key not in controls:
            count = __import__('math').prod(source)
            if count > 2**31-1:
                raise ValueError('index control too large')
            feed = {'x': np.arange(count,dtype=np.int32).reshape(source)}
            initial = [N.from_array(np.array(target,np.int64),name='target'),N.from_array(np.array(padded,np.int64),name='padded'),N.from_array(np.array(repeats,np.int64),name='repeats')]
            before = engine.checked_model([H.make_node('Expand',['x','target'],['y'])],[H.make_tensor_value_info('x',onnx.TensorProto.INT32,source)],[H.make_tensor_value_info('y',onnx.TensorProto.INT32,target)],initial)
            chain = [H.make_node('Reshape',['x','padded'],['v']),H.make_node('Tile',['v','repeats'],['y'])] if target else [H.make_node('Identity',['x'],['y'])]
            after = engine.checked_model(chain,[H.make_tensor_value_info('x',onnx.TensorProto.INT32,source)],[H.make_tensor_value_info('y',onnx.TensorProto.INT32,target)],initial)
            a = ort.InferenceSession(before.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,feed)[0]
            b = ort.InferenceSession(after.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,feed)[0]
            if a.shape!=b.shape or not np.array_equal(a,b):
                raise ValueError('Tile broadcast element addressing differs')
            controls[key] = dict(id=len(controls),source_shape=list(source),output_shape=list(target),repeats=repeats,output_elements=int(a.size),actual_ort_every_index_exact=True)
        changes.append(dict(node=node.name,source_shape=list(source),output_shape=list(target),repeats=repeats,control_id=controls[key]['id']))
        nodes.extend(new)
    if not changes:
        raise ValueError('no Expand matched installed backend coefficient materialization')
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(weights)
    return dict(expand_changes=changes,controls=list(controls.values()),scope='Tile is element replication only; no neural float arithmetic or task acceptance.')


def fold_root_extents(model, types, constants):
    """Materialize Shape/Size from freshly proven input extents, keeping neural nodes exact."""
    nodes, changes, controls = [], [], {}
    options=ort.SessionOptions();options.intra_op_num_threads=1
    for node in model.graph.node:
        if node.op_type not in ('Shape','Size'):
            nodes.append(copy.deepcopy(node));continue
        if len(node.input)!=1 or len(node.output)!=1 or node.output[0] not in constants:
            raise ValueError('extent root lacks proven value: '+node.name)
        dims=engine.fixed_extents(types[node.input[0]]);value=constants[node.output[0]]
        attributes={at.name:H.get_attribute_value(at) for at in node.attribute}
        expected=np.array(dims,np.int64)[attributes.get('start',0):attributes.get('end',len(dims))] if node.op_type=='Shape' else np.array(np.prod(dims,dtype=object),np.int64)
        if value.dtype!=np.int64 or not np.array_equal(value,expected):raise ValueError('root extent value differs from independently evaluated operator')
        key=(node.op_type,len(dims),repr(attributes))
        if key not in controls:
            cases=[]
            for case in ('nonzero','empty'):
                small=tuple(2+(i%2) for i in range(len(dims)))
                if case=='empty' and small:small=(0,)+small[1:]
                control_node=copy.deepcopy(node);del control_node.input[:];control_node.input.append('x');del control_node.output[:];control_node.output.append('y')
                input_info=H.make_tensor_value_info('x',H.TensorProto.FLOAT,list(small))
                target=np.array(small,np.int64)[attributes.get('start',0):attributes.get('end',len(small))] if node.op_type=='Shape' else np.array(np.prod(small,dtype=object),np.int64)
                graph=H.make_graph([control_node],'extent-semantics',[input_info],[H.make_tensor_value_info('y',H.TensorProto.INT64,list(target.shape))]);control=H.make_model(graph,opset_imports=list(model.opset_import));control.ir_version=model.ir_version
                actual=ort.InferenceSession(control.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,{'x':np.zeros(small,np.float32)})[0]
                if actual.dtype!=target.dtype or actual.shape!=target.shape or not np.array_equal(actual,target):raise ValueError('actual extent semantics differ')
                cases.append(dict(shape=list(small),value=actual.tolist(),exact=True))
            controls[key]=dict(op_type=node.op_type,rank=len(dims),attributes=attributes,actual_cases=cases)
        replacement=H.make_node('Constant',[],list(node.output),name=node.name+'/root_extent',value=N.from_array(value));nodes.append(replacement)
        changes.append(dict(node=node.name,source_input=node.input[0],source_input_shape=list(dims),output=node.output[0],value=value.tolist(),original_node_sha256=hashlib.sha256(node.SerializeToString()).hexdigest(),replacement_node_sha256=hashlib.sha256(replacement.SerializeToString()).hexdigest()))
    if not changes:raise ValueError('no root extent nodes matched')
    for before,after in zip(model.graph.node,nodes):
        if before.op_type not in ('Shape','Size') and before.SerializeToString()!=after.SerializeToString():raise ValueError('non-extent operation changed')
    del model.graph.node[:];model.graph.node.extend(nodes)
    return dict(changes=changes,actual_extent_controls=list(controls.values()),non_extent_operations_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--fold-root-extents', action='store_true')
    parser.add_argument('--bind-shape-controls', action='store_true')
    parser.add_argument('--tile-expands', action='store_true')
    parser.add_argument('--prune-empty-concat', action='store_true')
    parser.add_argument('--fold-reduceprod', action='store_true', help='Also fold proven int64 reductions in the bound candidate')
    args = parser.parse_args()
    if sha(args.model) != args.model_sha256:
        raise ValueError('source graph SHA differs')
    model = onnx.load(str(args.model))
    types, constants, errors = engine.prove_static_dataflow(model)
    if errors:
        raise ValueError('constant-root inference has errors')
    original = copy.deepcopy(model)
    if args.fold_root_extents:
        if args.tile_expands or args.bind_shape_controls or args.fold_reduceprod or args.prune_empty_concat:parser.error('--fold-root-extents is a separate candidate mode')
        evidence=fold_root_extents(model,types,constants)
        for boundary in ('input','output','initializer'):
            if [v.SerializeToString() for v in getattr(original.graph,boundary)]!=[v.SerializeToString() for v in getattr(model.graph,boundary)]:raise ValueError('extent fold changed ABI or source weights')
        onnx.checker.check_model(model,full_check=True)
        path=Path.cwd()/'model.root-extents.onnx';onnx.save(model,str(path))
        save(Path.cwd()/'preparation.json',dict(status='pass',source_model=str(args.model.absolute()),source_model_sha256=args.model_sha256,model=str(path),model_sha256=sha(path),script_sha256=sha(__file__),engine_sha256=sha(engine.__file__),ordered_abi_and_source_weights_unchanged=True,scope='Shape/Size from fresh root extents with actual small operator semantics controls; no neural or task acceptance.',**evidence));return
    if args.tile_expands:
        if args.bind_shape_controls or args.fold_reduceprod or args.prune_empty_concat:
            parser.error('--tile-expands is a separate accepted-candidate preparation mode')
        evidence = tile_expands(model, types)
        onnx.checker.check_model(model,full_check=True)
        for boundary in ('input','output'):
            if [v.SerializeToString() for v in getattr(original.graph,boundary)] != [v.SerializeToString() for v in getattr(model.graph,boundary)]:
                raise ValueError('Tile preparation altered ordered boundary')
        if any(v.SerializeToString()!=model.graph.initializer[i].SerializeToString() for i,v in enumerate(original.graph.initializer)):
            raise ValueError('Tile preparation altered source weights')
        path=Path.cwd()/'model.tiled-expands.onnx'
        onnx.save(model,str(path))
        save(Path.cwd()/'preparation.json',dict(status='pass',source_model_sha256=args.model_sha256,model=str(path),model_sha256=sha(path),script_sha256=sha(__file__),engine_sha256=sha(engine.__file__),**evidence))
        return
    if (args.fold_reduceprod or args.prune_empty_concat) and not args.bind_shape_controls:
        parser.error('--fold-reduceprod requires --bind-shape-controls')
    if args.bind_shape_controls:
        evidence = bind_shape_controls(model, types, constants)
        if args.fold_reduceprod:
            _, bound_constants, bound_errors = engine.prove_static_dataflow(model)
            if bound_errors:
                raise ValueError('bound candidate root inference has errors')
            evidence['reduceprod_changes'] = fold_reduceprod(model, bound_constants)
            if not evidence['reduceprod_changes']:
                raise ValueError('combined candidate has no proven int64 reductions')
        if args.prune_empty_concat:
            bound_types, _, bound_errors = engine.prove_static_dataflow(model)
            if bound_errors:
                raise ValueError('empty concat root inference has errors')
            evidence.update(prune_empty_concat(model, bound_types))
        onnx.checker.check_model(model, full_check=True)
        if [v.SerializeToString() for v in original.graph.input] != [v.SerializeToString() for v in model.graph.input] or [v.SerializeToString() for v in original.graph.output] != [v.SerializeToString() for v in model.graph.output]:
            raise ValueError('shape binding changed ordered boundary ABI')
        path = Path.cwd() / 'model.static-controls.onnx'
        onnx.save(model, str(path))
        save(Path.cwd() / 'preparation.json', dict(status='pass', source_model=str(args.model), source_model_sha256=args.model_sha256,
             model=str(path), model_sha256=sha(path), inputs=len(model.graph.input), outputs=len(model.graph.output),
             nodes=len(model.graph.node), ordered_abi_unchanged=True, checker_full_check=True, **evidence,
             engine_sha256=sha(engine.__file__), script_sha256=sha(__file__),
             scope='Bind proven static shape controls and remove now-dead control paths; neural tensor/task acceptance separate.'))
        print(json.dumps(dict(status='pass', model_sha256=sha(path), nodes=len(model.graph.node), control_bindings=evidence['control_operand_bindings'])))
        return
    changes = fold_reduceprod(model, constants)
    if not changes:
        raise ValueError('no proven int64 ReduceProd matched the observed fault')
    onnx.checker.check_model(model, full_check=True)
    if [v.SerializeToString() for v in original.graph.input] != [v.SerializeToString() for v in model.graph.input] or [v.SerializeToString() for v in original.graph.output] != [v.SerializeToString() for v in model.graph.output]:
        raise ValueError('ordered boundary ABI changed')
    if [v.SerializeToString() for v in original.graph.initializer] != [v.SerializeToString() for v in model.graph.initializer]:
        raise ValueError('source weights changed')
    path = Path.cwd() / 'model.reduceprod-folded.onnx'
    onnx.save(model, str(path))
    save(Path.cwd() / 'preparation.json', dict(status='pass', source_model=str(args.model),
         source_model_sha256=args.model_sha256, model=str(path), model_sha256=sha(path), changes=changes,
         inputs=len(model.graph.input), outputs=len(model.graph.output), nodes=len(model.graph.node),
         ordered_abi_unchanged=True, source_initializers_unchanged=True, checker_full_check=True,
         engine_sha256=sha(engine.__file__), script_sha256=sha(__file__),
         scope='Int64 ReduceProd constants only, checked using actual ORT control graphs; no neural QNN acceptance.'))
    print(json.dumps(dict(status='pass', changed_nodes=len(changes), model_sha256=sha(path))))


if __name__ == '__main__':
    main()
