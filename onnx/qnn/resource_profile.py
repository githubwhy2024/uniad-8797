"""Bind one measured host resource policy to exact source and backend assets."""
import json,os,threading,time
from pathlib import Path
import borrowed_session
from partition_session import bind_partitions as original_bind
from resource_partition_session import PROVENANCE
from resources import terminal
from tool_run import ROOT,save,sha
def bind_resource_partitions(*args,policy_path,policy_sha256,**kwargs):
    if sha(policy_path)!=policy_sha256:raise ValueError('resource policy identity differs')
    terminal(Path(policy_path).parent);policy=json.loads(Path(policy_path).read_text())
    load=Path(policy['source']['load_control']);terminal(load.parent)
    if sha(load)!=policy['source']['load_control_sha256']:raise ValueError('resource load evidence changed')
    evidence=json.loads(load.read_text())
    if evidence['status']!='pass' or policy['source']['build_sha256']!=evidence['build_sha256']:raise ValueError('resource policy is not bound to a passed native load control')
    if policy['source']['physical_rss_limit_bytes']!=18*1024**3:raise ValueError('resource physical ceiling changed')
    profile=original_bind(*args,**kwargs)
    if evidence['build_sha256']!=profile['assets']['partition_build']['sha256']:raise ValueError('resource load estimate belongs to a different build')
    if [r['index'] for r in evidence['parts']]!=[r['index'] for r in profile['parts']]:raise ValueError('load measurement coverage differs')
    if any(str(r['index']) not in policy['context_estimates_bytes'] or policy['context_estimates_bytes'][str(r['index'])]!=r['context_estimate_bytes'] for r in evidence['parts']):raise ValueError('resource estimate differs from load control')
    roles=[('resource_adapter',ROOT/'onnx/qnn/resource_partition_session.py'),('graph_resource_helper',ROOT/'onnx/qnn/graph_resources.py'),('borrowed_input_adapter',ROOT/'onnx/qnn/borrowed_session.py'),('resource_frame_driver',ROOT/'onnx/validation/run_resource_real_frames.py'),('resource_profile_helper',Path(__file__)),('resource_policy',Path(policy_path))]
    for role,path in roles:profile['assets'][role]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
    profile['transport_policy']=dict(type='synchronous_borrowed_numpy_inputs',source=borrowed_session.PROVENANCE)
    profile['graph_resource_policy']=policy;profile['resource_provenance']=PROVENANCE
    return profile
def enforce_host_rss():
    ceiling=18*1024**2
    while True:
        fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if line.split(':',1)[0] in ('VmRSS','VmHWM','VmSize','VmPeak')}
        if int(fields['VmHWM'].split()[0])>ceiling:
            save(Path.cwd()/'resource_memory_guard.json',dict(status='rejected',reason='actual_process_peak_rss_budget_exceeded',ceiling_kib=ceiling,memory=fields,scope='No task/state promotion; physical peak RSS budget maintained independently of virtual address allowance.'))
            os._exit(86)
        time.sleep(2)
def start_host_rss_guard():threading.Thread(target=enforce_host_rss,daemon=True,name='resource-rss-guard').start()
