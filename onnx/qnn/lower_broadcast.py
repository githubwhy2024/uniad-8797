#!/usr/bin/env python3
"""Group proven adjacent axes around the observed mixed-layout attention multiply."""
import argparse, copy, json, sys
from pathlib import Path
import onnx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
import shape_dataflow as engine
from prove_graph_extents import sha,save
from lower_rank import Lowering,witness

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--model-sha256',required=True)
    args=parser.parse_args()
    if sha(args.model)!=args.model_sha256:raise ValueError('source identity differs')
    model=onnx.load(str(args.model));original=copy.deepcopy(model)
    types,constants,errors=engine.prove_static_dataflow(model)
    if errors:raise ValueError('source root inference failed')
    lowering=Lowering(types,constants,set());nodes=[];changes=[];controls={}
    existing={v for n in model.graph.node for v in list(n.input)+list(n.output)}|{v.name for v in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type!='Mul' or '/deformable_attention/' not in node.name or not node.name.endswith('/Mul_8'):
            nodes.append(copy.deepcopy(node));continue
        shapes=[lowering.dims(v) for v in node.input];out=lowering.dims(node.output[0])
        if len(shapes)!=2 or len(out)!=4 or shapes[0]!=out or shapes[1]!=(out[0],1,out[2],out[3]):
            raise ValueError('observed attention broadcast pattern changed')
        chain,weights=lowering.lower(node)
        new={v for n in chain for v in n.output}|{v.name for v in weights}
        if (new-set(node.output))&existing:raise ValueError('broadcast node collision')
        existing.update(new);nodes.extend(chain);model.graph.initializer.extend(weights)
        key=tuple(shapes)
        if key not in controls:
            controls[key]=witness(node,chain,weights,lowering,Path.cwd(),len(controls))
        changes.append(dict(node=node.name,source_shapes=shapes,grouped_shapes=[[out[0],out[1],out[2]*out[3]],[out[0],1,out[2]*out[3]]],control_id=controls[key]['id']))
    if not changes:raise ValueError('observed broadcast pattern absent')
    del model.graph.node[:];model.graph.node.extend(nodes);del model.graph.value_info[:]
    onnx.checker.check_model(model,full_check=True)
    for boundary in ('input','output'):
        if [v.SerializeToString() for v in getattr(original.graph,boundary)]!=[v.SerializeToString() for v in getattr(model.graph,boundary)]:raise ValueError('ordered ABI changed')
    if any(v.SerializeToString()!=model.graph.initializer[i].SerializeToString() for i,v in enumerate(original.graph.initializer)):raise ValueError('source weights changed')
    output=Path.cwd()/'model.grouped-broadcast.onnx';onnx.save(model,str(output))
    save(Path.cwd()/'broadcast.json',dict(status='pass',source_model=str(args.model),source_model_sha256=args.model_sha256,model=str(output),model_sha256=sha(output),script_sha256=sha(__file__),lowering_sha256=sha(Path(__file__).with_name('lower_rank.py')),engine_sha256=sha(engine.__file__),changes=changes,controls=list(controls.values()),ordered_abi_unchanged=True,original_weights_unchanged=True,scope='Adjacent axes with identical broadcasting behavior are grouped into rank three via pure views; original multiply remains. Float statistics recorded; neural task acceptance separate.'))
if __name__=='__main__':main()
