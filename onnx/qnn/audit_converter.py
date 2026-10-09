"""Compact actual SDK network interface and FP32 representation audit."""
import collections,json,math
from pathlib import Path
import onnx
from session import DTYPES,sha

def describe(model,net_path,native_layouts=None):
 proto=onnx.load(str(model),load_external_data=False);net=json.loads(Path(net_path).read_text());graph=net['graph'];tensors=graph['tensors']
 if 'float_bw=32' not in net['converter_command']:raise ValueError('explicit converter FP32 policy absent')
 histogram=collections.Counter(t['data_type'] for t in tensors.values())
 if any(t['data_type'] in (0x0216,0x0264) or t.get('unquantized_data_type') in (0x0216,0x0264) for t in tensors.values()):raise ValueError('unexpected non-FP32 floating tensor representation')
 rows={}
 for kind,values,code in [('inputs',proto.graph.input,0),('outputs',proto.graph.output,1)]:
  actual={name:row for name,row in tensors.items() if row['type']==code}
  if set(actual)!={v.name for v in values}:raise ValueError('converter boundary names differ')
  rows[kind]=[]
  for v in values:
   logical=[d.dim_value for d in v.type.tensor_type.shape.dim];dtype=str(onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type));native=actual[v.name]
   if DTYPES[native['data_type']]!=dtype:raise ValueError('converter boundary dtype differs: '+v.name)
   policy=(native_layouts or {}).get(v.name);wire='identity' if logical==native['dims'] else 'singleton_axes'
   if policy:
    if policy not in [dict(model='NCHW',native='NHWC'),dict(model='NHWC',native='NCHW')]:raise ValueError('unsupported explicit native layout declaration')
    perm=[0,2,3,1] if policy==dict(model='NCHW',native='NHWC') else [0,3,1,2]
    if len(logical)!=4 or native['dims']!=[logical[i] for i in perm] or dtype!='float32':raise ValueError('declared native layout differs: '+v.name)
    wire='explicit_'+policy['model']+'_to_'+policy['native']
   elif [d for d in logical if d!=1]!=[d for d in native['dims'] if d!=1]:raise ValueError('converter boundary layout or extent changed: '+v.name)
   if any(native.get('is_dynamic_dims',[])) or native['dataFormat']!=0:raise ValueError('converter boundary is not fixed flat buffer')
   rows[kind].append(dict(name=v.name,dtype=dtype,source_shape=logical,native_shape=native['dims'],axis_format=native.get('axis_format'),permute_order_to_src=native.get('permute_order_to_src'),bytes=math.prod(native['dims'])*onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type).itemsize,wire_view=wire))
 return dict(static_tensor_count=sum(t['type']==4 for t in tensors.values()),status='pass',model_sha256=sha(model),net_sha256=sha(net_path),net_bytes=Path(net_path).stat().st_size,command=net['converter_command'],interfaces=rows,tensor_dtype_histogram={str(k):v for k,v in histogram.items()},operator_type_histogram=dict(collections.Counter(n['type'] for n in graph['nodes'].values())),scope='Generated network metadata/FP32/interface only; actual CPU load, frames and task acceptance separate.')
