"""FP32 sampling/weight/reduce reference and limited query-block ORT controls."""
import argparse,copy,json,sys,time
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as H
from validate_query_streaming import fixture,feed,stats
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import sha,save

def fused_reference(inputs,chunk):
 """Stream each level/point and corner; never materialize N*C*Q*32.

 Bilinear, zeros, align_corners=0; level-major/point-major reduction. This is
 a mathematical FP32 host reference, not an SDK device kernel or a new ABI.
 """
 weights=inputs['attention'];batch,_,queries,points=weights.shape;channels=inputs['feature0'].shape[1];result=np.empty((batch,channels,queries),np.float32)
 for start in range(0,queries,chunk):
  end=min(start+chunk,queries);output=np.zeros((batch,channels,end-start),np.float32)
  for level in range(4):
   features=inputs['feature'+str(level)];grid=inputs['grid'+str(level)][:,start:end];height,width=features.shape[-2:]
   for point in range(8):
    gx,gy=grid[:,:,point,0],grid[:,:,point,1];x=(gx+np.float32(1))*np.float32(width/2)-np.float32(.5);y=(gy+np.float32(1))*np.float32(height/2)-np.float32(.5)
    x0=np.floor(x).astype(np.int64);y0=np.floor(y).astype(np.int64);fx=x-x0.astype(np.float32);fy=y-y0.astype(np.float32);sample=np.zeros_like(output)
    for dx,dy,wx,wy in [(0,0,1-fx,1-fy),(1,0,fx,1-fy),(0,1,1-fx,fy),(1,1,fx,fy)]:
     ix=x0+dx;iy=y0+dy;valid=(ix>=0)&(ix<width)&(iy>=0)&(iy<height);factor=(wx*wy*valid)[:,None,:]
     corner=features[np.arange(batch)[:,None],:,np.clip(iy,0,height-1),np.clip(ix,0,width-1)].transpose(0,2,1)
     sample+=corner*factor
    output+=sample*weights[:,0,start:end,level*8+point][:,None,:]
  result[:,:,start:end]=output
 return result

def measure(fn,repeats=5):
 values=[];output=fn()
 for _ in range(repeats):
  start=time.perf_counter();output=fn();values.append(time.perf_counter()-start)
 return output,dict(mean_seconds=float(np.mean(values)),p95_seconds=float(np.percentile(values,95)),samples=values)

def main():
 options=ort.SessionOptions();options.intra_op_num_threads=1;options.log_severity_level=3;rows=[];checks={};budgets=[]
 for queries,batch,channels in [(17,2,3),(10201,2,8)]:
  ref,_=fixture(queries,2048,batch,channels);baseline=ort.InferenceSession(ref.SerializeToString(),options,providers=['CPUExecutionProvider'])
  for seed in (9,21):
   inputs=feed(ref,seed);snapshots={k:v.copy() for k,v in inputs.items()};expected,bt=measure(lambda:baseline.run(None,inputs)[0]);held=expected.copy()
   for chunk in (512,2048,4096):
    _,model=fixture(queries,chunk,batch,channels);session=ort.InferenceSession(model.SerializeToString(),options,providers=['CPUExecutionProvider']);actual,timing=measure(lambda:session.run(None,inputs)[0]);fused,ft=measure(lambda:fused_reference(inputs,chunk),repeats=3)
    rows.append(dict(queries=queries,batch=batch,channels=channels,seed=seed,chunk_queries=chunk,streamed_ort_difference=stats(expected,actual),fused_reference_difference=stats(expected,fused),baseline_ort=bt,streamed_ort=timing,fused_numpy=ft))
    checks[str((queries,seed,chunk))+'_inputs_and_retained_output_owned']=all(np.array_equal(v,snapshots[k]) for k,v in inputs.items()) and np.array_equal(expected,held)
    # Split fixture at sampler boundary, exposing intermediate buffers; merged
    # and split graphs execute the exact same operations/point order.
    split=copy.deepcopy(ref);samples=[n.output[0] for n in split.graph.node if n.op_type=='GridSample'];split.graph.output.extend([H.make_tensor_value_info(n,onnx.TensorProto.FLOAT,[batch,channels,queries,8]) for n in samples]);onnx.checker.check_model(split,full_check=True)
    exposed=ort.InferenceSession(split.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,inputs);host_reduction=(np.concatenate(exposed[1:],axis=-1)*inputs['attention']).sum(-1,dtype=np.float32)
    rows[-1]['exposed_sampler_boundary_difference']=stats(expected,host_reduction)
   checks[str((queries,seed))+'_contracts']=all(np.isfinite(v).all() for v in inputs.values())
  budgets.append(dict(batch=batch,channels=channels,queries=queries,unblocked_sample32_bytes=batch*channels*queries*32*4,unblocked_concat_and_product_bytes=2*batch*channels*queries*32*4,output_bytes=batch*channels*queries*4,blocks=[dict(chunk_queries=n,sample32_bytes=batch*channels*min(n,queries)*32*4,fused_one_point_sample_bytes=batch*channels*min(n,queries)*4) for n in (512,2048,4096)],scope='Selected explicit tensor extents only; inputs/output and NumPy corner/index scratch also exist. No SDK scratch or peak RAM upper bound.'))
 report=dict(status='pass' if all(checks.values()) else 'failed',checks=checks,controls=rows,budgets=budgets,source_sha256=sha(__file__),streaming_source_sha256=sha(Path(__file__).resolve().parents[1]/'qnn/stream_attention.py'),attributes=dict(mode='bilinear',padding_mode='zeros',align_corners=0),scope='Constructed FP32 ORT and streamed NumPy mathematical reference with sampler boundaries and partial chunks. Float differences only recorded. Neither target kernel nor task acceptance; original2048 full graph retained pending target evidence.')
 save(Path.cwd()/'attention_fusion_control.json',report);print(json.dumps(dict(status=report['status'],cases=len(rows),max_fused_abs=max(r['fused_reference_difference']['max_abs'] for r in rows)),indent=2))
 if not all(checks.values()):raise ValueError('fusion contract failed')
if __name__=='__main__':main()
