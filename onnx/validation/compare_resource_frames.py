"""Compare bound initial real frames and record performance without float gates."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'onnx/qnn'))
from resources import terminal
from tool_run import save,sha
def difference(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('saved output ABI differs')
    if not np.issubdtype(a.dtype,np.floating):
        return dict(integer_or_boolean_exact=bool(np.array_equal(a,b)),changed=int(np.count_nonzero(a!=b)))
    maximum=0.;absolute=0.;count=0
    for i in range(0,a.size,1048576):
        x=a.reshape(-1)[i:i+1048576];y=b.reshape(-1)[i:i+1048576]
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('saved floating output nonfinite')
        values=np.abs(x.astype(np.float64)-y);maximum=max(maximum,float(values.max(initial=0)));absolute+=float(values.sum());count+=values.size
    return dict(max_abs=maximum,mean_abs=absolute/max(count,1),scope='Float statistics only; task metrics govern final acceptance.')
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-run',type=Path,required=True);p.add_argument('--candidate-run',type=Path,action='append',required=True);a=p.parse_args()
    terminal(a.reference_run);reference=json.loads((a.reference_run/'neural_result.json').read_text());reference_profile=json.loads((a.reference_run/'profile.json').read_text());first=reference['completed_frames'][0];reports=[]
    for run in a.candidate_run:
        terminal(run);result=json.loads((run/'neural_result.json').read_text());profile=json.loads((run/'profile.json').read_text());frame=result['completed_frames'][0]
        if result['status']!='pass' or result['profile_sha256']!=sha(run/'profile.json'):raise ValueError('candidate execution identity differs')
        if first['token']!=frame['token'] or first['observation']['inputs']!=frame['observation']['inputs'] or profile['abi']!=reference_profile['abi']:raise ValueError('not a same-input initial-frame comparison')
        for role in ('logical_model','native_model','initial_state','host','state_contract','collision_optimizer'):
            if profile['assets'][role]['sha256']!=reference_profile['assets'][role]['sha256']:raise ValueError('candidate logical/Host/initial source changed')
        for row in (first,frame):
            for role in ('outputs','final_plan'):
                if sha(row[role+'_path'])!=row[role+'_sha256']:raise ValueError('saved output changed')
        with np.load(first['outputs_path'],allow_pickle=False) as before,np.load(frame['outputs_path'],allow_pickle=False) as after:
            if set(before.files)!=set(after.files):raise ValueError('saved output names differ')
            values={n:difference(before[n],after[n]) for n in before.files}
        if not all(v.get('integer_or_boolean_exact',True) for v in values.values()):raise ValueError('initial frame discrete outputs differ')
        with np.load(first['final_plan_path'],allow_pickle=False) as before,np.load(frame['final_plan_path'],allow_pickle=False) as after:final=difference(before['planning_final'],after['planning_final'])
        parts=frame['observation']['resource_parts'];timing={k:sum(p[k] for p in parts) for k in ('prepare_seconds','execute_seconds','native_api_execute_seconds','close_seconds','shared_root_guard_seconds','shared_root_guard_bytes')}
        timing.update(frame_seconds=frame['elapsed_seconds'],inference_with_boundaries_seconds=frame['observation']['inference_seconds'],host_journal_serialization_seconds=frame['elapsed_seconds']-frame['observation']['inference_seconds'],python_native_boundary_seconds=timing['execute_seconds']-timing['native_api_execute_seconds'],partition_public_boundary_seconds=frame['observation']['inference_seconds']-timing['prepare_seconds']-timing['execute_seconds']-timing['close_seconds'])
        transport={k:sum(p['input_transport_delta'][k] for p in parts) for k in parts[0]['input_transport_delta']}
        reports.append(dict(run=str(run),result_sha256=sha(run/'result.json'),profile_sha256=result['profile_sha256'],parts=len(profile['parts']),timing=timing,input_transport=transport,peak_rss_kib=result['peak_rss_kib'],resource_counts=frame['observation']['resource_counts'],public_outputs=values,final_plan=final,reference_first_frame_seconds=first['elapsed_seconds'],speedup=first['elapsed_seconds']/frame['elapsed_seconds']))
    save(Path.cwd()/'resource_frame_comparison.json',dict(status='pass',reference_run=str(a.reference_run),reference_result_sha256=sha(a.reference_run/'result.json'),candidates=reports,scope='Exact same initial input/state/logical source/Host frame. Strict ABI/finite/discrete output control and float statistics; cold measured duration/resources only. One-frame evidence does not establish own-state short chain, 81 task metrics or target performance.'))
if __name__=='__main__':main()
