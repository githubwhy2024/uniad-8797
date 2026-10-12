"""Frozen/current full-frame PT regression with isolated processes and own recurrence.

This is the cleanup acceptance gate, not ONNX/ORT or target acceptance.
Large source snapshots, tensors and logs belong in an external --run-dir.
"""
import argparse
import ast
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = '35c8cacdbf023a7a52da82738d3fc4c6a81f87ea'
MODULES = {
    'qualcomm_attention':'attention', 'msda_compat':'attention',
    'exportable_dcnv2':'dcnv2', 'onnx_export_patches':'model_patches',
    'qualcomm_spatial':'frame_core', 'stateful_bev':'frame_core',
    'stateful_spatial':'frame_core', 'stateful_map':'frame_core',
    'stateful_rotation':'frame_core', 'stateful_tracking':'temporal',
    'stateful_wrapper':'temporal', 'stateful_runtime':'host',
    'stateful_postprocess':'host', 'export_onnx_stateful':'export',
    'export_support':'export', 'preflight_validation_assets':'assets',
    'prepare_nuscenes_scene':'prepare_scene', 'validate_mini_equivalence':'validate',
}
CLASSES = {'UniADStage2ExportWrapper':'FrameCore','UniADStatefulExportWrapper':'StatefulStep'}
EXPECTED = {'frame_core','attention','dcnv2','model_patches','temporal','state_contract',
            'host','prepare_scene','export','assets','validate','check_host','verify_refactor'}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def source_hashes():
    files=subprocess.check_output(['git','ls-files','*.py'],cwd=ROOT,text=True).splitlines()
    files=set(files)|{str(p.relative_to(ROOT)) for p in (Path(__file__).resolve().parent).glob('*.py')}
    return {p:sha(ROOT/p) for p in sorted(files) if (ROOT/p).is_file()}


class Normalize(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        node.module=MODULES.get(node.module,node.module)
        for alias in node.names:alias.name=CLASSES.get(alias.name,alias.name)
        return node
    def visit_Name(self,node):
        node.id=CLASSES.get(node.id,node.id);return node
    def visit_ClassDef(self,node):
        node.name=CLASSES.get(node.name,node.name);self.generic_visit(node);return node
    def visit_Expr(self,node):
        # Documentation is not executable math. Ignore moved/reworded docstrings.
        if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):return None
        return self.generic_visit(node)


def nodes(text):
    return {n.name:n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


def static_gate():
    files={p.stem:p for p in (Path(__file__).resolve().parent).glob('*.py')}
    assert set(files)==EXPECTED, sorted(set(files)^EXPECTED)
    for p in files.values():ast.parse(p.read_bytes())
    matched=[]
    excluded={'export_onnx_stateful':{'build_model','write_initial_state_bundle'},
              'stateful_runtime':{'validate_state'},
              'stateful_spatial':{'dynamic_spatial_forward'},
              'stateful_rotation':{'qualcomm_custom_rotate','QualcommGridSampleRotation'},
              'exportable_dcnv2':{'modulated_deform_conv2d_grid_sample'}}
    def compare(old,new,label):
        assert ast.dump(Normalize().visit(old))==ast.dump(Normalize().visit(new)), label
        matched.append(label)
    for origin,target in MODULES.items():
        if origin=='export_support':continue  # Introduced after the frozen baseline.
        old=nodes(subprocess.check_output(['git','show',REFERENCE+':onnx/'+origin+'.py'],cwd=ROOT))
        for name,node in old.items():
            if name in excluded.get(origin,set()):continue
            current=nodes(files[target].read_bytes())[CLASSES.get(name,name)]
            compare(node,current,origin+':'+name)
    stage2=nodes(subprocess.check_output(['git','show',REFERENCE+':onnx/export_onnx_stage2.py'],cwd=ROOT))
    for name in ['_mock_unused_imports','build_cfg','register_onnx_symbolics']:
        compare(stage2[name],nodes(files['export'].read_bytes())[name],'bootstrap:'+name)
    compare(stage2['TrackBBoxesTensor'],nodes(files['frame_core'].read_bytes())['TrackBBoxesTensor'],'TrackBBoxesTensor')
    old=stage2['UniADStage2ExportWrapper']
    old.body=[n for n in old.body if not (isinstance(n,ast.FunctionDef) and n.name=='forward')]
    compare(old,nodes(files['frame_core'].read_bytes())['FrameCore'],'FrameCore_without_retired_legacy_forward')
    old=nodes(subprocess.check_output(['git','show',REFERENCE+':onnx/export_onnx_stateful.py'],cwd=ROOT))
    compare(old['build_model'],nodes(files['export'].read_bytes())['build_model'],'build_model')
    writer=nodes(files['export'].read_bytes())['write_initial_state_bundle']
    assert isinstance(writer.body[1],ast.ImportFrom) and writer.body[1].module=='temporal'
    writer.body.pop(1)
    compare(old['write_initial_state_bundle'],writer,'bundle_writer_except_local_import')
    assert not subprocess.check_output(['git','diff',REFERENCE,'--','projects','tools'],cwd=ROOT)
    # Host must remain usable without loading Torch or any neural layer.
    code="import sys;sys.path.insert(0,'onnx');import host;assert 'torch' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)
    return dict(status='pass',files=len(files),unchanged_math=matched,projects_tools_unchanged=True)


def worker(args):
    import numpy as np
    import torch
    run=Path(args.run_dir).resolve();out=run/args.worker;out.mkdir(exist_ok=False)
    source=run/'reference_source' if args.worker=='reference' else ROOT
    sys.path.insert(0,str(source));sys.path.insert(0,str(source/'onnx'))
    os.chdir(run/'assets')
    torch.set_num_threads(4);torch.manual_seed(0)
    ex=importlib.import_module('export_onnx_stateful' if args.worker=='reference' else 'export')
    tm=importlib.import_module('stateful_wrapper' if args.worker=='reference' else 'temporal')
    cls=getattr(tm,'UniADStatefulExportWrapper' if args.worker=='reference' else 'StatefulStep')
    record=dict(status='running',stage='building',pid=os.getpid(),frames=[],source=str(source),
                environment=dict(torch=torch.__version__,numpy=np.__version__,threads=4,seed=0,device='cpu'))
    save(out/'status.json',record)
    def tensor_record(t):
        a=t.detach().cpu().contiguous().numpy()
        return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
    try:
        config=str(source/'projects/configs/stage2_e2e/base_e2e.py')
        cfg,model=ex.build_model(config,args.checkpoint)
        net=cls(model,cfg.occflow_grid_conf).eval()
        aliases={}
        for name,p in net.named_parameters(remove_duplicate=False):aliases.setdefault(id(p),[]).append(name)
        snapshot=dict(state_dict={k:tensor_record(v) for k,v in net.state_dict().items()},
                      aliases=sorted(sorted(v) for v in aliases.values() if len(v)>1),
                      inputs=tm.INPUT_NAMES,outputs=tm.OUTPUT_NAMES,dynamic_axes=tm.dynamic_axes(),
                      modules={n:CLASSES.get(type(m).__name__,type(m).__name__) for n,m in net.named_modules()})
        with torch.no_grad():
            inputs=ex.make_inputs(net,seed=0)
            snapshot['initial_inputs']={k:tensor_record(v) for k,v in zip(tm.INPUT_NAMES,inputs)}
            fixture=out/'identity_fixture.onnx';fixture.write_bytes(b'identity-only-not-a-valid-onnx')
            ex.write_initial_state_bundle(net,fixture,out/'initial.npz',config,args.checkpoint)
            try:ex.write_initial_state_bundle(net,fixture,out/'initial.npz',config,args.checkpoint)
            except FileExistsError:pass
            else:raise AssertionError('bundle overwrite allowed')
            save(out/'construction.json',snapshot)
            if args.worker=='candidate':
                assert json.loads(json.dumps(snapshot))==json.loads((run/'reference/construction.json').read_text()),'construction/ABI/input mismatch'
                with np.load(run/'reference/initial.npz') as a,np.load(out/'initial.npz') as b:
                    assert set(a.files)==set(b.files)
                    for name in a.files:assert np.array_equal(a[name],b[name]),'bundle '+name
            trace=[]
            handles=[]
            for name,module in net.named_modules():
                handles.append(module.register_forward_pre_hook(lambda m,ins,name=name:trace.append('module:'+name)))
            # Direct method calls bypass nn.Module hooks. Record the actual head/BEV dispatch too.
            for owner,method,label in [(model,'get_bevs','get_bevs'),(model.pts_bbox_head,'get_detections','get_detections'),
                                       (model.motion_head,'forward_test','motion.forward_test'),
                                       (model.occ_head,'forward_test','occ.forward_test')]:
                original=getattr(owner,method)
                def traced(*a,_original=original,_label=label,**kw):
                    trace.append('method:'+_label);return _original(*a,**kw)
                setattr(owner,method,traced)
            for frame in range(args.frames):
                trace.clear();record.update(stage='forward',current_frame=frame);save(out/'status.json',record)
                entry=dict(frame=frame,inputs={k:tensor_record(v) for k,v in zip(tm.INPUT_NAMES,inputs)})
                if args.worker=='candidate':
                    ref=json.loads((run/'reference'/f'frame_{frame}.json').read_text())
                    assert entry['inputs']==ref['inputs'],'own recurrence inputs differ at frame '+str(frame)
                start=time.monotonic();outputs=net(*inputs);entry['seconds']=time.monotonic()-start
                assert len(outputs)==len(tm.OUTPUT_NAMES)==34
                values={k:v.detach().cpu().numpy() for k,v in zip(tm.OUTPUT_NAMES,outputs)}
                assert all(np.isfinite(v).all() for v in values.values()),'nonfinite output'
                entry['outputs']={k:tensor_record(v) for k,v in zip(tm.OUTPUT_NAMES,outputs)}
                entry['trace']=list(trace)
                entry['next_rows']=len(values['next_query'])
                assert (values['next_obj_idxes'][:901]==-1).all()
                assert (values['next_obj_idxes'][901:]>=0).all()
                for name in tm.INPUT_NAMES[13:]:assert len(values['next_'+name])==entry['next_rows']
                if args.worker=='reference':np.savez(out/f'frame_{frame}.npz',**values)
                else:
                    differences=[]
                    with np.load(run/'reference'/f'frame_{frame}.npz') as expected:
                        for name,v in values.items():
                            e=expected[name]
                            if v.shape!=e.shape or v.dtype!=e.dtype or not np.array_equal(v,e):
                                detail=dict(name=name,shape=list(v.shape),reference_shape=list(e.shape))
                                if v.shape==e.shape and v.size:detail['max_abs']=float(np.max(np.abs(v.astype(np.float64)-e.astype(np.float64))))
                                differences.append(detail)
                    entry['differences']=differences
                    entry['trace_equal']=entry['trace']==ref['trace']
                    save(out/f'frame_{frame}.json',entry)
                    assert not differences and entry['trace_equal'],'output or executed path mismatch at frame '+str(frame)
                save(out/f'frame_{frame}.json',entry)
                record['frames'].append(dict(frame=frame,seconds=entry['seconds'],next_rows=entry['next_rows']))
                inputs=ex.advance(inputs,outputs)  # Always this model's own outputs.
            for handle in handles:handle.remove()
            assert any(f['next_rows']>901 for f in record['frames'][:-1]),'sequence did not consume survivors'
        record.update(status='pass',stage='complete')
    except BaseException:
        record.update(status='fail',stage='failed',error=traceback.format_exc());raise
    finally:
        save(out/'status.json',record);save(out/'result.json',record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--motion-anchor',required=True)
    parser.add_argument('--frames',type=int,default=3)
    parser.add_argument('--worker',choices=['reference','candidate'])
    parser.add_argument('--static-only',action='store_true')
    args=parser.parse_args()
    args.run_dir=str(Path(args.run_dir).resolve());args.checkpoint=str(Path(args.checkpoint).resolve());args.motion_anchor=str(Path(args.motion_anchor).resolve())
    if args.frames<3:parser.error('at least three frames required')
    if args.worker:return worker(args)
    if args.static_only:
        result=static_gate();save(Path(args.run_dir)/'static.json',result);print('static PASS',len(result['unchanged_math']));return
    run=Path(args.run_dir);run.mkdir(parents=True,exist_ok=True)
    if (run/'status.json').exists():parser.error('fresh run directory required; do not overwrite evidence')
    record=dict(status='running',stage='preflight',pid=os.getpid(),reference=REFERENCE,
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                candidate_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                source_hashes=source_hashes(),arguments=vars(args),checks=[])
    save(run/'status.json',record)
    try:
        record['static']=static_gate()
        record['assets']={name:dict(path=path,sha256=sha(path)) for name,path in [('checkpoint',args.checkpoint),('anchor',args.motion_anchor)]}
        dest=run/'reference_source';dest.mkdir(exist_ok=False)
        archive=subprocess.check_output(['git','archive',REFERENCE],cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(dest)
        asset=run/'assets/data/others';asset.mkdir(parents=True)
        (asset/'motion_anchor_infos_mode6.pkl').symlink_to(args.motion_anchor)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
        steps=[('host',[sys.executable,str(Path(__file__).resolve().parent/'check_host.py'),'--run-dir',str(run/'host')])]
        for side in ['reference','candidate']:
            steps.append((side,[sys.executable,str(Path(__file__).resolve()),'--run-dir',str(run),'--checkpoint',args.checkpoint,'--motion-anchor',args.motion_anchor,'--frames',str(args.frames),'--worker',side]))
        for name,cmd in steps:
            record['stage']=name;save(run/'status.json',record)
            with open(run/(name+'.log'),'w') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                record['child_pid']=child.pid;save(run/'status.json',record)
                try:code=child.wait(timeout=14400)
                except subprocess.TimeoutExpired:child.kill();child.wait();raise
            item=dict(name=name,returncode=code);record['checks'].append(item)
            result=run/name/'result.json'
            if result.exists():item.update(result=str(result),sha256=sha(result))
            assert code==0 and json.loads(result.read_text())['status']=='pass',name+' failed; see '+str(run/(name+'.log'))
        assert record['source_hashes']==source_hashes(),'candidate source changed during validation'
        record.update(status='pass',stage='complete',scope='full synthetic three-frame CPU PT exact equivalence; not real-data/task/ONNX/target acceptance')
    except BaseException:
        record.update(status='fail',stage='failed',error=traceback.format_exc());raise
    finally:
        save(run/'status.json',record);save(run/'result.json',record)


if __name__=='__main__':main()
