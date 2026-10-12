"""Compare source-protected semantic boundaries without emitting model copies."""
import argparse,gc,json,sys
from pathlib import Path
import onnx
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'onnx/qnn'))
from semantic_partition import plan_boundaries,semantic_regions
from tool_run import sha,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--logical-model',type=Path,required=True);p.add_argument('--logical-model-sha256',required=True);p.add_argument('--expanded',action='store_true');p.add_argument('--extent-budget-gib',action='append',help='Explicit target:hard produced-extent proxy; actual physical memory must be measured separately.');a=p.parse_args()
    if sha(a.plan)!=a.plan_sha256 or sha(a.logical_model)!=a.logical_model_sha256:raise ValueError('planning input identity changed')
    plan=json.loads(a.plan.read_text());source=Path(plan['source_model'])
    if sha(source)!=plan['source_model_sha256']:raise ValueError('planning CPU source changed')
    logical=onnx.load(str(a.logical_model),load_external_data=False);regions,indices=semantic_regions(logical)
    raw_attention=[r for r in regions if r['kind']=='sample_weight_reduce']
    if len(raw_attention)!=6:raise ValueError('logical attention source mapping incomplete')
    del logical;gc.collect()
    model=onnx.load(str(source),load_external_data=False);candidates=[]
    specifications=[('protect_reference',2,4),('frontier_aware',2,4)] if not a.expanded else [('protect_reference',2,4)]+[('frontier_aware',t,h) for t,h in [(1,2),(2,4),(4,8),(8,12),(12,16)]]
    if a.extent_budget_gib:
        specifications=[]
        for value in a.extent_budget_gib:
            target,hard=map(int,value.split(':'))
            if not 0<target<=hard:raise ValueError('positive ordered extent proxy budgets required')
            specifications.append(('frontier_aware',target,hard))
    for policy,target_gib,hard_gib in specifications:
        label=policy+'_'+str(target_gib)+'g' if a.expanded or a.extent_budget_gib else policy
        try:
            row=plan_boundaries(model,plan,target_gib*1024**3,hard_gib*1024**3,policy)
            row.update(status='pass',source_model=str(source),source_model_sha256=plan['source_model_sha256'],reference_plan=str(a.plan),reference_plan_sha256=a.plan_sha256)
        except ValueError as e:
            row=dict(status='rejected',policy=policy,reason=str(e),scope='No model emitted, no compile/execute/task.')
        path=Path.cwd()/(label+'_plan.json');save(path,row)
        candidates.append(dict(label=label,policy=policy,target_gib=target_gib,hard_gib=hard_gib,path=str(path),sha256=sha(path),status=row['status'],summary=row.get('summary'),reason=row.get('reason')))
        save(Path.cwd()/'semantic_partition_progress.json',dict(completed=policy))
    success=[c for c in candidates if c['status']=='pass']
    # Prefer the conservative group-preserving policy if its crossing extent
    # improves; one candidate is selected, not two production graph branches.
    selected=None if a.expanded or a.extent_budget_gib else next((c for c in success if c['policy']=='protect_reference'),success[0] if success else None)
    report=dict(status='pass' if success else 'failed',checks=dict(logical_source_regions_bound=True,complete_cpu_coverage=bool(success)),source=dict(logical=str(a.logical_model),logical_sha256=a.logical_model_sha256,cpu=str(source),cpu_sha256=plan['source_model_sha256']),raw_attention_regions=raw_attention,candidates=candidates,selected=selected,helper_sha256=sha(ROOT/'onnx/qnn/semantic_partition.py'),scope='A1 full source DAG planning only. Expanded comparison authorized by latest user; selection pending actual resource/performance probes, not smallest graph count. Generic groups use all source consumers; actual materialization/layout/SDK/own-state acceptance pending. No model copies or graph compilation.')
    save(Path.cwd()/'semantic_partition_comparison.json',report);print(json.dumps(dict(status=report['status'],selected=selected,candidates=candidates)))
    if not success:raise ValueError('no legal semantic candidate')
if __name__=='__main__':main()
