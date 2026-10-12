#!/usr/bin/env python3
"""Compile one accepted converter artifact, verifying cpp/bin identities first."""
import argparse,json,subprocess
from pathlib import Path
from tool_run import SDK,sha,save,process_identity,stamp

def main():
    p=argparse.ArgumentParser();p.add_argument('--convert-run',type=Path,required=True);p.add_argument('--name',default='q4_model');args=p.parse_args()
    if not args.name.replace('_','').isalnum():raise ValueError('model library name is invalid')
    origin=args.convert_run.absolute();result=json.loads((origin/'result.json').read_text());status=json.loads((origin/'status.json').read_text())
    if result['status']!='pass' or status['status']!='pass' or status['result_sha256']!=sha(origin/'result.json') or result['acceptance_status']!='convert_tool_pass_task_not_evaluated':raise ValueError('converter attempt is not terminal tool-pass')
    identity=json.loads((origin/'source_identity.json').read_text())
    if sha(origin/'source_identity.json')!=result['source_identity_sha256'] or sha(identity['model'])!=identity['model_sha256']:raise ValueError('converter/model provenance differs')
    resources={}
    for name in ('model.cpp',) + (('model.bin',) if 'model.bin' in result['artifacts'] else ()):
        row=result['artifacts'][name];path=origin/name
        if row['path']!=str(path) or path.resolve()!=path or sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('converter artifact identity differs: '+name)
        resources[name]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
    if 'model.bin' not in resources:
        audit_path=origin/'converter_audit.json';net_path=origin/'model_net.json'
        for path in (audit_path,net_path):
            if result['artifacts'][path.name]['sha256']!=sha(path):raise ValueError('weightless converter metadata identity differs')
            resources[path.name]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
        audit=json.loads(audit_path.read_text());net=json.loads(net_path.read_text())
        if result.get('weight_artifact')!='none_no_static_tensors' or audit['static_tensor_count']!=0 or any(t['type']==4 for t in net['graph']['tensors'].values()) or (origin/'model.bin').exists():raise ValueError('weightless converter contract differs')
    run=Path.cwd();command=[str(SDK/'bin/x86_64-linux-clang/qnn-model-lib-generator'),'-c',str(origin/'model.cpp')]
    if 'model.bin' in resources:command+=['-b',str(origin/'model.bin')]
    command+=['-t','x86_64-linux-clang','-o',str(run/'lib'),'-l',args.name]
    child=subprocess.Popen(command)
    save(run/'build_launch.json',dict(command=command,cwd=str(run),compiler_target='x86_64-linux-clang',child=process_identity(child.pid),started_utc=stamp(),converter_result_sha256=sha(origin/'result.json'),converter_source_identity_sha256=sha(origin/'source_identity.json'),source_model_sha256=identity['model_sha256'],resources=resources,generator_sha256=sha(command[0])))
    if child.wait()!=0:raise RuntimeError('model lib generation failed')
    out=run/'lib/x86_64-linux-clang'/('lib'+args.name+'.so')
    if out.resolve()!=out or not out.is_file() or not out.stat().st_size:raise ValueError('compiled model library absent')
    save(run/'model_build.json',dict(status='pass',compiler_target='x86_64-linux-clang',library=str(out),library_sha256=sha(out),library_bytes=out.stat().st_size,converter_result_sha256=sha(origin/'result.json'),resources=resources,source_model_sha256=identity['model_sha256'],cpu_backend_sha256=sha(SDK/'lib/x86_64-linux-clang/libQnnCpu.so'),script_sha256=sha(__file__),scope='Actual host model-lib compilation; backend load, frames and task acceptance are separate.'))
if __name__=='__main__':main()
