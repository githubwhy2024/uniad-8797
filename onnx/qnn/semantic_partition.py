"""Pure dependency/extent planning; backend layouts and graphs stay replaceable."""
import bisect
import hashlib
import math
import numpy as np
import onnx
from onnx import helper as H
from attention_regions import attention_regions,extents,small_constants,static_types
from graph_cost import graph_dependencies


def semantic_regions(model):
    types=static_types(model);nodes=list(model.graph.node)
    producer,consumers,live,roots,immutable=graph_dependencies(model)
    executed=sorted(live-roots);position={i:p for p,i in enumerate(executed)}
    regions=[]
    for row in attention_regions(model,types,small_constants(model)):
        indices=row['node_indices']
        regions.append(dict(kind='sample_weight_reduce',id=hashlib.sha256(('attention:'+row['output']).encode()).hexdigest(),source_node_indices=indices,source_node_sha256=row['node_sha256'],output=row['output'],shape=list(row['shape']),adapter=row['adapter'],features=row['features'],grids=row['grids'],weights=row['weights']))
    public={v.name for v in model.graph.output};reserved={i for r in regions for i in r['source_node_indices']}
    # Only exact single-consumer standard spatial chains. Add may consume an
    # external residual, which remains an explicit incoming dependency.
    for index in executed:
        node=nodes[index]
        if node.op_type!='Conv' or node.domain or index in reserved:continue
        chain=[index];value=node.output[0]
        for _ in range(8):
            users=consumers.get(value,set())&live
            if value in public or len(users)!=1:break
            follow=next(iter(users));next_node=nodes[follow]
            if follow not in position or follow in reserved or next_node.domain or next_node.op_type not in ('Relu','LeakyRelu','Add','Identity'):break
            shape=extents(types,value)
            if shape is None or len(shape)!=4 or any(extents(types,v)!=shape for v in next_node.output):break
            chain.append(follow);value=next_node.output[0]
        if len(chain)>1:
            reserved.update(chain);regions.append(dict(kind='conv_spatial_consumers',id=hashlib.sha256(('conv:'+node.output[0]).encode()).hexdigest(),source_node_indices=chain,source_node_sha256=[hashlib.sha256(nodes[i].SerializeToString()).hexdigest() for i in chain],output=value))
    for row in regions:
        positions=[position[i] for i in row['source_node_indices']]
        row['start']=min(positions);row['end']=max(positions)+1
        selected=set(row['source_node_indices']);produced={v for i in selected for v in nodes[i].output}
        row['external_consumers']={v:sorted(consumers.get(v,set())-selected) for v in produced if consumers.get(v,set())-selected or v in public}
        row['public_outputs']=sorted(produced&public)
    return regions,executed


def plan_boundaries(model,reference_plan,target_bytes,hard_bytes,policy):
    nodes=list(model.graph.node);types=static_types(model)
    producer,consumers,live,roots,immutable=graph_dependencies(model)
    regions,executed=semantic_regions(model);position={i:p for p,i in enumerate(executed)}
    old_flat=[i for p in reference_plan['parts'] for i in p['source_node_indices']]
    if old_flat!=executed:raise ValueError('reference independent ordered executable coverage differs')
    def size(value):
        dims=extents(types,value)
        if dims is None:raise ValueError('unproven planning extent: '+value)
        return math.prod(dims)*np.dtype(H.tensor_dtype_to_np_dtype(types[value].tensor_type.elem_type)).itemsize
    prefix=[0]
    for index in executed:prefix.append(prefix[-1]+sum(size(v) for v in nodes[index].output))
    protected=set()
    for row in regions:protected.update(range(row['start']+1,row['end']))
    legal=[i for i in range(1,len(executed)+1) if i not in protected]
    impossible=[r['id'] for r in regions if prefix[r['end']]-prefix[r['start']]>hard_bytes]
    if impossible:raise ValueError('region exceeds selected host extent guard: '+str(impossible))
    public={v.name for v in model.graph.output}
    frontier_delta=[0]*(len(executed)+2)
    for value,index in producer.items():
        if index not in position:continue
        start=position[index]+1;users=[position[i] for i in consumers.get(value,set()) if i in position]
        end=len(executed)+1 if value in public else max(users,default=position[index])+1
        frontier_delta[start]+=size(value);frontier_delta[end]-=size(value)
    frontier=[];amount=0
    for change in frontier_delta:amount+=change;frontier.append(amount)
    reference_ends=[];total=0
    for part in reference_plan['parts']:total+=len(part['source_node_indices']);reference_ends.append(total)
    if policy=='protect_reference':
        # Remove only boundaries cutting a protected region. This is a single
        # conservative candidate, not a sweep of every graph-size setting.
        ends=[end for end in reference_ends if end not in protected]
        starts=[0]+ends[:-1]
        if any(prefix[e]-prefix[s]>hard_bytes for s,e in zip(starts,ends)):
            raise ValueError('protected merge exceeds host guard')
    elif policy=='frontier_aware':
        ends=[];start=0
        while start<len(executed):
            upper=bisect.bisect_right(prefix,prefix[start]+hard_bytes)-1
            target=bisect.bisect_left(prefix,prefix[start]+target_bytes)
            lo=bisect.bisect_right(legal,start);hi=bisect.bisect_right(legal,upper)
            choices=[v for v in legal[lo:hi] if abs(v-target)<=128]
            if not choices:choices=legal[max(lo,hi-1):hi]
            if not choices:raise ValueError('no legal bounded dependency cut')
            if len(executed)<=upper:end=len(executed)
            else:end=min(choices,key=lambda v:(frontier[v]+abs(prefix[v]-prefix[start]-target_bytes)//4,v))
            ends.append(end);start=end
    else:raise ValueError('unknown boundary policy')
    if not ends or ends[-1]!=len(executed):raise ValueError('incomplete boundary coverage')
    intervals=[];start=0
    for end in ends:intervals.append((start,end));start=end
    owners={index:p for p,(s,e) in enumerate(intervals) for index in executed[s:e]}
    cross=0;out_extent=0
    for value,index in producer.items():
        if index not in owners:continue
        outside={owners[i] for i in consumers.get(value,set()) if i in owners and owners[i]!=owners[index]}
        cross+=size(value)*len(outside)
        if outside or value in public:out_extent+=size(value)
    unchanged={tuple(p['source_node_indices']) for p in reference_plan['parts']}
    parts=[dict(index=p,source_node_indices=executed[s:e],estimated_produced_bytes=prefix[e]-prefix[s],boundary_frontier_bytes=frontier[e],reference_nodes_unchanged=tuple(executed[s:e]) in unchanged) for p,(s,e) in enumerate(intervals)]
    return dict(policy=policy,target_produced_bytes=target_bytes,hard_produced_bytes=hard_bytes,regions=regions,parts=parts,summary=dict(parts=len(parts),preserved_reference_node_groups=sum(p['reference_nodes_unchanged'] for p in parts),cut_consumer_logical_bytes=cross,cut_output_extent_bytes=out_extent,max_produced_extent_bytes=max(p['estimated_produced_bytes'] for p in parts),max_boundary_frontier_bytes=max(p['boundary_frontier_bytes'] for p in parts),protected_attention_regions=sum(r['kind']=='sample_weight_reduce' for r in regions),protected_conv_chains=sum(r['kind']=='conv_spatial_consumers' for r in regions)),scope='Full exact source operations and every consumer retained. Protected standard semantic intervals, memory proxy and cut extent only; no physical copy elimination, backend fusion or graph compile/execution acceptance. All logical public/state ports unchanged. Backend layout remains a separate profile.')
