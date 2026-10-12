"""Measure serial native preparations before choosing a bounded host policy."""
import argparse,faulthandler,gc,json,math,os,resource,subprocess,sys,threading,time
from pathlib import Path
faulthandler.enable()
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'onnx/qnn'))
import partition_session
from borrowed_session import BorrowedNativeSession
from resources import terminal
from tool_run import save,sha,process_identity
GIB=1024**3
def memory():
    return {line.split(':',1)[0]:int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines() if line.split(':',1)[0] in ('VmRSS','VmHWM','VmSize','VmPeak')}
def guard():
    while True:
        fields=memory()
        if fields['VmHWM']>18*1024**2:
            save(Path.cwd()/'resource_memory_guard.json',dict(status='rejected',reason='actual_process_peak_rss_budget_exceeded',ceiling_kib=18*1024**2,memory_kib=fields));os._exit(86)
        time.sleep(2)
def select_retention(rows,budget):
    # Keep room for the largest transient, even while all selected resources
    # are resident. A measured preparation-time objective replaces scan LRU.
    unit=64*1024**2;maximum=max(r['context_estimate_bytes'] for r in rows)
    capacity=max(0,(budget-maximum)//unit);states={0:(0.,())}
    for row in rows:
        weight=math.ceil(row['context_estimate_bytes']/unit)
        for used,(benefit,indices) in list(states.items()):
            total=used+weight
            if total<=capacity and benefit+row['prepare_seconds']>states.get(total,(-1.,()))[0]:states[total]=(benefit+row['prepare_seconds'],indices+(row['index'],))
    benefit,indices=max(states.values(),key=lambda v:v[0]);ordered=sorted(indices,key=lambda i:rows[i]['prepare_seconds'],reverse=True)
    return ordered,dict(estimated_prepare_seconds_saved_per_warm_cycle=benefit,reserved_largest_transient_bytes=maximum,selection_unit_bytes=unit,rounded_retention_capacity_bytes=capacity*unit)
def main():
    p=argparse.ArgumentParser();p.add_argument('--build-run',type=Path);p.add_argument('--worker-profile',type=Path);p.add_argument('--worker-profile-sha256');p.add_argument('--worker-index',type=int);p.add_argument('--retention-mode',choices=('measured','none'),default='measured');a=p.parse_args()
    if a.worker_profile:
        if sha(a.worker_profile)!=a.worker_profile_sha256:raise ValueError('worker profile changed')
        threading.Thread(target=guard,daemon=True,name='resource-rss-guard').start()
        profile=json.loads(a.worker_profile.read_text());partition_session.NativeSession=BorrowedNativeSession
        owner=partition_session.PartitionSession(profile);native=None;row=profile['parts'][a.worker_index]
        try:
            gc.collect();before=memory();save(Path.cwd()/('part%03d_load_progress.json'%row['index']),dict(stage='before_prepare',index=row['index'],memory_kib=before))
            started=time.perf_counter();native=owner._native(row);seconds=time.perf_counter()-started;after=memory();abi=native.native_abi
            if after['VmHWM']>18*1024**2:
                save(Path.cwd()/'resource_memory_guard.json',dict(status='rejected',reason='actual_prepare_peak_rss_budget_exceeded',ceiling_kib=18*1024**2,memory_kib=after,index=row['index']))
                raise ValueError('actual prepare peak RSS exceeds physical host budget')
            increment=max(0,after['VmRSS']-before['VmRSS'])*1024
            estimate=max(256*1024**2,math.ceil(increment*1.2)+128*1024**2) if a.retention_mode=='measured' else max(256*1024**2,increment)
            native.close();native=None;gc.collect()
            result=dict(index=row['index'],prepare_seconds=seconds,memory_before_kib=before,memory_after_kib=after,memory_after_close_kib=memory(),rss_increment_bytes=increment,context_estimate_bytes=estimate,context_estimate_basis='safety_margin_20_percent_plus_128mib' if a.retention_mode=='measured' else 'observed_prepare_increment_lower_bound_only',native_abi=abi)
            save(Path.cwd()/('part%03d_load.json'%row['index']),result)
        finally:
            if native is not None:native.close()
            owner.close()
        return
    if not a.build_run:raise ValueError('build run required')
    terminal(a.build_run)
    profile=partition_session.bind_partitions(a.build_run,ROOT/'onnx/runs/q4-native-bridge-build-9a762031',ROOT/'onnx/resources/model.planning.onnx','0202b4e2ce5bff2ce94ef04447efcbbe581c73b54e263d4922aefc44f0c01812')
    rows=[]
    save(Path.cwd()/'profile.json',profile)
    for row in profile['parts']:
        # A fresh process avoids mistaking reused allocator pages for a small
        # graph footprint. Only one backend preparation is active at a time.
        command=[sys.executable,str(Path(__file__)), '--worker-profile',str(Path.cwd()/'profile.json'),'--worker-profile-sha256',sha(Path.cwd()/'profile.json'),'--worker-index',str(row['index']),'--retention-mode',a.retention_mode]
        child=subprocess.Popen(command,cwd=Path.cwd())
        save(Path.cwd()/'partition_resource_progress.json',dict(stage='prepare_worker',index=row['index'],child=process_identity(child.pid),command=command,completed=rows))
        if child.wait():raise RuntimeError('native load worker failed: '+str(row['index']))
        result=json.loads((Path.cwd()/('part%03d_load.json'%row['index'])).read_text());rows.append(result)
        save(Path.cwd()/'partition_resource_progress.json',dict(stage='between_parts',completed=rows))
    budget=(13 if a.retention_mode=='measured' else 18)*GIB
    if any(r['context_estimate_bytes']>budget for r in rows):raise ValueError('one measured graph exceeds reserved context budget; restore a semantic boundary')
    report=dict(status='pass',build_run=str(a.build_run),build_sha256=sha(a.build_run/'partition_build.json'),profile_sha256=sha(Path.cwd()/'profile.json'),parts=rows,peak_rss_kib=max(r['memory_after_kib']['VmHWM'] for r in rows),controller_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,address_space_limit=resource.getrlimit(resource.RLIMIT_AS),physical_rss_limit_bytes=18*GIB,scope='Actual serial compose/finalize/ABI/close for every graph, each in a fresh worker to prevent allocator reuse hiding footprint. For measured retention, prepare RSS plus safety allowance is only a host context estimate; for no retention, the raw prepare increment is an advisory lower bound; execution scratch, whole-frame RSS and task acceptance remain to be measured. No simultaneous all-context residency.')
    save(Path.cwd()/'partition_resource_control.json',report)
    indices,selection=select_retention(rows,budget) if a.retention_mode=='measured' else ([],dict(estimated_prepare_seconds_saved_per_warm_cycle=0.,reserved_largest_transient_bytes=max(r['context_estimate_bytes'] for r in rows)))
    scope='Host-only measured capacities; portable lifecycle/ownership contract. Reserve at least 5GiB for frame/state/frontier/shared buffers and retained outputs. Actual execute RSS guard overrides estimates; no device budget guarantee.' if a.retention_mode=='measured' else 'One transient graph only, no retained context overlap. The context estimate is an advisory filter and does not reserve noncontext memory; the unchanged18GiB whole-process RSS guard is authoritative during real execution. Load success does not prove frame budget or device acceptance.'
    policy=dict(context_estimates_bytes={str(r['index']):r['context_estimate_bytes'] for r in rows},retain_indices=indices,max_retained_contexts=len(indices),max_active_contexts=len(indices)+1,context_budget_bytes=budget,source=dict(kind='measured_bounded_host_profile',retention_mode=a.retention_mode,load_control=str(Path.cwd()/'partition_resource_control.json'),load_control_sha256=sha(Path.cwd()/'partition_resource_control.json'),build_sha256=report['build_sha256'],selection=selection,reserved_noncontext_bytes=(5 if a.retention_mode=='measured' else 0)*GIB,physical_rss_limit_bytes=18*GIB,scope=scope))
    save(Path.cwd()/'graph_resource_policy.json',policy);print(json.dumps(dict(status='pass',parts=len(rows),retain_indices=indices,estimated_prepare_saved=selection['estimated_prepare_seconds_saved_per_warm_cycle'],peak_rss_kib=report['peak_rss_kib'])))
if __name__=='__main__':main()
