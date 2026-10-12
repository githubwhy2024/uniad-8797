"""Backend-independent static work/dependency ledger; never a traffic estimate."""
import collections
import hashlib
import math
import numpy as np
import onnx
from onnx import helper as H
from attention_regions import extents, static_types


def graph_dependencies(model):
    nodes = list(model.graph.node)
    producer = {}
    consumers = collections.defaultdict(set)
    for index,node in enumerate(nodes):
        for value in node.output:
            if value in producer:
                raise ValueError('duplicate graph producer: ' + value)
            producer[value] = index
        for value in node.input:
            if value:consumers[value].add(index)
    live=set(); pending=[v.name for v in model.graph.output]
    while pending:
        value=pending.pop();index=producer.get(value)
        if index is None or index in live:continue
        live.add(index);pending.extend(v for v in nodes[index].input if v)
    immutable={v.name for v in model.graph.initializer};roots=set()
    nondeterministic={'RandomNormal','RandomUniform','RandomNormalLike','RandomUniformLike','Multinomial','Dropout'}
    for index,node in enumerate(nodes):
        if node.op_type=='Constant' or (node.input and all(v in immutable for v in node.input if v) and node.op_type not in nondeterministic and not node.domain):
            immutable.update(node.output);roots.add(index)
    return producer,consumers,live,roots,immutable


def cost_ledger(model, model_sha256, plan=None):
    nodes=list(model.graph.node);types=static_types(model)
    producer,consumers,live,roots,immutable=graph_dependencies(model)
    executed=live-roots
    if plan is not None:
        planned=[i for part in plan['parts'] for i in part['source_node_indices']]
        if len(planned)!=len(set(planned)) or set(planned)!=executed or plan['source_model_sha256']!=model_sha256:
            raise ValueError('independent executable coverage differs from plan')
    public_inputs={v.name for v in model.graph.input};public_outputs={v.name for v in model.graph.output}
    scope_notes=[];dense=[];samples=[];reductions=[];tensor_rows=[];output_bytes=collections.Counter();macs=collections.Counter();op_counts=collections.Counter();max_internal=0
    missing=set()
    def shape(value):
        dims=extents(types,value)
        if dims is None:missing.add(value)
        return dims
    def byte_count(value):
        dims=shape(value)
        return None if dims is None else math.prod(dims)*np.dtype(H.tensor_dtype_to_np_dtype(types[value].tensor_type.elem_type)).itemsize
    owners={} if plan is None else {i:p['index'] for p in plan['parts'] for i in p['source_node_indices']}
    for index in sorted(executed):
        node=nodes[index];op_counts[node.op_type]+=1
        for value in node.output:
            amount=byte_count(value)
            if amount is not None:output_bytes[node.op_type]+=amount;max_internal=max(max_internal,amount)
        attrs={a.name:H.get_attribute_value(a) for a in node.attribute}
        if node.op_type in ('Conv','MatMul','Gemm'):
            left,right,result=[shape(v) for v in [node.input[0],node.input[1],node.output[0]]]
            if None in (left,right,result):
                scope_notes.append(dict(node=index,op=node.op_type,reason='shape unavailable'));continue
            if node.op_type=='Conv':
                if len(left)!=len(right) or left[1]!=right[1]*attrs.get('group',1):raise ValueError('invalid grouped Conv contract')
                contracted=right[1]*math.prod(right[2:])
            elif node.op_type=='MatMul':
                contracted=left[-1]
                if contracted!=(right[-2] if len(right)>1 else right[-1]):raise ValueError('MatMul contracted axis differs')
            else:
                if len(left)!=2 or len(right)!=2:raise ValueError('Gemm must be rank two')
                contracted=left[0] if attrs.get('transA',0) else left[1]
                if contracted!=(right[1] if attrs.get('transB',0) else right[0]):raise ValueError('Gemm contracted axis differs')
            amount=math.prod(result)*contracted;macs[node.op_type]+=amount
            dense.append(dict(index=index,name=node.name,op=node.op_type,inputs=[list(left),list(right)],output=list(result),contracted_terms=contracted,dense_macs=amount,node_sha256=hashlib.sha256(node.SerializeToString()).hexdigest()))
        if node.op_type=='GridSample':
            dims=[shape(v) for v in list(node.input)+list(node.output)]
            sample_shape=dims[-1]
            samples.append(dict(index=index,name=node.name,input_shapes=dims[:-1],output_shape=sample_shape,output_bytes=byte_count(node.output[0]),
                                mode=str(attrs.get('mode',b'bilinear')),padding=str(attrs.get('padding_mode',b'zeros')),align_corners=attrs.get('align_corners',0),
                                output_elements=None if sample_shape is None else math.prod(sample_shape),bilinear_neighbor_bound=4))
        if node.op_type.startswith('Reduce'):
            reductions.append(dict(index=index,name=node.name,op=node.op_type,input_shape=shape(node.input[0]),output_shape=shape(node.output[0]),
                                   input_elements=None if shape(node.input[0]) is None else math.prod(shape(node.input[0]))))
    for value,index in producer.items():
        if index not in executed:continue
        user_indices=sorted(consumers.get(value,set()) & executed)
        outside=[] if plan is None else sorted({owners[i] for i in user_indices if owners[i]!=owners[index]})
        tensor_rows.append(dict(name=value,producer=index,consumers=user_indices,shape=shape(value),dtype=H.TensorProto.DataType.Name(types[value].tensor_type.elem_type) if value in types else None,
                                extent_bytes=byte_count(value),public_output=value in public_outputs,producer_part=owners.get(index),consumer_parts=outside))
    # SSA frontier deliberately excludes roots/inputs and all backend scratch.
    last={v:max(consumers.get(v,set()) & executed,default=-1) for v in producer}
    for v in public_outputs:last[v]=len(nodes)
    frontier={};peak=0
    for index in sorted(executed):
        for v in nodes[index].output:
            amount=byte_count(v)
            if amount is not None:frontier[v]=amount
        peak=max(peak,sum(frontier.values()))
        for v in list(frontier):
            if last[v]<=index and v not in public_outputs:frontier.pop(v)
    logical_cross=sum((r['extent_bytes'] or 0)*len(r['consumer_parts']) for r in tensor_rows)
    summary=dict(nodes=len(nodes),live_nodes=len(live),immutable_root_nodes=len(roots & live),executable_nodes=len(executed),omitted_unreachable_nodes=len(nodes)-len(live),
                 dense_nodes=len(dense),dense_macs_per_frame=sum(macs.values()),dense_macs_by_op=dict(macs),
                 op_counts=dict(op_counts),produced_logical_extent_bytes=sum(output_bytes.values()),logical_extent_by_op=dict(output_bytes),
                 largest_output_extent_bytes=max_internal,ssa_dynamic_frontier_bytes=peak,cut_consumer_logical_bytes=logical_cross if plan else None,
                 sampling_nodes=len(samples),sampling_output_extent_bytes=sum(r['output_bytes'] or 0 for r in samples),
                 reduction_nodes=len(reductions),missing_shape_names=sorted(missing),dense_coverage_gaps=scope_notes)
    return dict(model_sha256=model_sha256,summary=summary,dense_nodes=dense,sampling=samples,reductions=reductions,tensors=tensor_rows,
                executable_node_indices=sorted(executed),immutable_root_node_indices=sorted(roots & live),
                scope='Static dense arithmetic of live non-root Conv/MatMul/Gemm and explicit tensor dependency/extents. MatMul vector/broadcast dimensions use output shape. Not original-algorithm minimum work, backend instructions, FLOPs/TOPS or FPS. Sampling/reduction geometry separate. SSA frontier excludes root/input storage, native allocation/scratch, aliases, fragmentation and retained prior outputs; not peak RAM. Extent totals are not physical copy/DDR/disk measurements.')
