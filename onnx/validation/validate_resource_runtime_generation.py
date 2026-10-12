"""Portable resource implementation controls without model/library copies."""
import argparse,ast,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'onnx/qnn'))
from planning_identity import definition,generated_resource_runtime,portable_resource_policy
from resource_partition_session import PROVENANCE
from tool_run import save,sha
def main():
    partition=definition((ROOT/'onnx/qnn/partition_session.py').read_text(),'PartitionSession',ast.ClassDef)
    ranges=definition((ROOT/'onnx/qnn/resources.py').read_text(),'RANGES',ast.Assign)
    journal=definition((ROOT/'onnx/qnn/tool_run.py').read_text(),'save',ast.FunctionDef)
    code=generated_resource_runtime(partition,ranges,journal,(ROOT/'onnx/qnn/resource_partition_session.py').read_text());path=Path.cwd()/'generated_resource_runtime.py';path.write_text(code)
    for name in ('borrowed_session.py','session.py','graph_resources.py'):shutil.copyfile(ROOT/'onnx/qnn'/name,Path.cwd()/name)
    sys.path.insert(0,str(Path.cwd()))
    for name in ('borrowed_session','session','graph_resources'):sys.modules.pop(name,None)
    spec=importlib.util.spec_from_file_location('portable_resource_control',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    if module.PROVENANCE!=PROVENANCE:raise ValueError('portable execution scope or source provenance differs')
    resource=Path.cwd()/'dummy_resource.txt';resource.write_text('dummy ABI fixture');digest=sha(resource)
    abi=dict(schema='qnn-native-abi-v1',inputs=[dict(name='x',native_name='x',shape=[2],native_shape=[2],dtype='float32')],outputs=[dict(name='y',native_name='y',shape=[2],native_shape=[2],dtype='float32')])
    part=dict(index=0,inputs=[dict(source_name='x',name='x',shape=[2],dtype='float32',bytes=8)],outputs=[dict(source_name='y',name='y',shape=[2],dtype='float32',bytes=8)],drop_after=['x'])
    row=dict(index=0,native_model_sha256=digest,library=str(resource),library_sha256=digest,abi=abi)
    manifest=dict(assets={k:dict(path=str(resource),sha256=digest,bytes=resource.stat().st_size) for k in ('bridge','backend_lib')},parts=[row],plan=dict(parts=[part]),abi=dict(inputs=[dict(name='x',shape=[2],parent_shape=[2],dtype='float32')],outputs=[dict(name='y',shape=[2],parent_shape=[2],dtype='float32')]))
    policy=dict(context_estimates_bytes={'0':10},retain_indices=[0],max_retained_contexts=1,max_active_contexts=2,context_budget_bytes=20,source='constructed portable control')
    policy['source']=dict(kind='constructed_portable_control',load_control=str(ROOT/'not-a-runtime-dependency.json'),load_control_sha256=digest,physical_rss_limit_bytes=18*1024**3)
    before=json.dumps(policy,sort_keys=True);portable=portable_resource_policy(policy)
    if json.dumps(policy,sort_keys=True)!=before or 'load_control' in portable['source'] or portable['source']!={k:v for k,v in policy['source'].items() if k!='load_control'} or any(portable[k]!=policy[k] for k in policy if k!='source'):raise ValueError('portable policy changed a measured constraint or source binding')
    owner=module.ResourcePartitionSession(manifest,portable);created=[]
    class Native:
        def __init__(self):self.native_abi=abi;self.closed=False;created.append(self)
        def run(self,names,feed):return [feed['x']+np.float32(1)]
        def close(self):self.closed=True
    owner._native=lambda row:Native();held=[]
    for cycle in range(3):
        feed=np.array([cycle,2],np.float32);out=owner.run(None,{'x':feed})[0]
        if not np.array_equal(out,feed+1):raise ValueError('portable arithmetic differs')
        held.append((out,out.copy()))
    owner.reset_state();counts=owner.graph_resources.snapshot();owner.close()
    if counts['prepare_count']!=1 or counts['cache_hit']!=2 or not all(v.closed for v in created) or not all(np.array_equal(a,b) for a,b in held):raise ValueError('portable lifetime/ownership differs')
    bad=Path.cwd()/'forged_resource_runtime.py';tree=ast.parse(code);next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PartitionSession').body.append(ast.parse('forged=True').body[0]);bad.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    bad_spec=importlib.util.spec_from_file_location('forged_resource_control',bad);bad_module=importlib.util.module_from_spec(bad_spec);sys.modules[bad_spec.name]=bad_module
    try:bad_spec.loader.exec_module(bad_module)
    except ValueError as error:
        if 'portable resource base class differs' not in str(error):raise
    else:raise ValueError('portable base class mutation was accepted')
    save(Path.cwd()/'resource_runtime_generation_control.json',dict(status='pass',provenance=module.PROVENANCE,counts=counts,old_outputs_owned_after_close=True,base_class_mutation_rejected=True,portable_policy_preserves_constraints_and_evidence_hashes=True,generated_sha256=sha(path),generator_sha256=sha(ROOT/'onnx/qnn/planning_identity.py'),scope='Exact resource scope/provenance and generated base-class guard, three-cycle dummy lifetime/owned-output control. No NN/library/Host/portable neural/task acceptance.'))
if __name__=='__main__':main()
