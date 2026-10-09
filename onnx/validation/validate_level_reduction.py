"""Finite controls of optional level-before-concatenation FP32 graph grouping."""
import copy,json,sys,time
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as H
from validate_query_streaming import fixture,feed,stats
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from attention_fusion_reference import level_reduced_nodes
from tool_run import save,sha

def main():
 o=ort.SessionOptions();o.intra_op_num_threads=1;o.log_severity_level=3;controls=[];checks={}
 for queries,batch,channels in [(1,2,3),(17,2,3),(10201,2,8)]:
  for chunk in (512,2048,4096):
   reference,current=fixture(queries,chunk,batch,channels);candidate=copy.deepcopy(reference);candidate.graph.ClearField('node');candidate.graph.node.extend(level_reduced_nodes(['grid'+str(i) for i in range(4)],['feature'+str(i) for i in range(4)],'attention','result',(batch,channels,queries,32),chunk,'level_reduce',dict(mode='bilinear',padding_mode='zeros',align_corners=0)));candidate.graph.ClearField('initializer');onnx.checker.check_model(candidate,full_check=True)
   sessions=[ort.InferenceSession(m.SerializeToString(),o,providers=['CPUExecutionProvider']) for m in (reference,current,candidate)]
   for seed in (9,21):
    inputs=feed(reference,seed);before={k:v.copy() for k,v in inputs.items()};outputs=[s.run(None,inputs)[0] for s in sessions];old=outputs[-1].copy();changed=feed(reference,seed+1);sessions[-1].run(None,changed)
    checks[str((queries,chunk,seed))+'_ownership']=all(np.array_equal(v,before[k]) for k,v in inputs.items()) and np.array_equal(outputs[-1],old)
    timing=[]
    for session in sessions:
     samples=[]
     for _ in range(3):
      start=time.perf_counter();session.run(None,inputs);samples.append(time.perf_counter()-start)
     timing.append(dict(mean_seconds=float(np.mean(samples)),samples=samples))
    controls.append(dict(queries=queries,batch=batch,channels=channels,chunk_queries=chunk,seed=seed,current_difference=stats(outputs[0],outputs[1]),level_reduced_difference=stats(outputs[0],outputs[2]),cpu_ort_timing=dict(baseline=timing[0],query_streamed=timing[1],level_reduced=timing[2])))
 # Candidate graph retained as a tiny portable prototype, not a deployed model.
 onnx.save(candidate,str(Path.cwd()/'level-reduced-control.onnx'))
 report=dict(status='pass' if all(checks.values()) else 'failed',checks=checks,controls=controls,prototype_model_sha256=sha(Path.cwd()/'level-reduced-control.onnx'),helper_sha256=sha(Path(__file__).resolve().parents[1]/'qnn/attention_fusion_reference.py'),budget_for_real_group=dict(batch=6,channels=256,chunk_queries=2048,current_sample32_bytes=6*256*2048*32*4,level_sample8_bytes=6*256*2048*8*4,partial_output_bytes=6*256*2048*4),decision='reference_prototype_only; current full graph retained',reason='Local grouping controls do not establish full-model task acceptance or native backend scratch/physical-copy savings. Retain original full point reduction pending a separately justified graph candidate; Q5 kernel design can use both references.',scope='18 constructed FP32 ORT cases, own output ownership and observed floating differences; timings coexist with necessary native short validation. No QNN kernel, full model or target acceptance.')
 save(Path.cwd()/'level_reduction_control.json',report);print(json.dumps(dict(status=report['status'],cases=len(controls),max_abs=max(v['level_reduced_difference']['max_abs'] for v in controls))))
if __name__=='__main__':main()
