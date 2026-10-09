"""Persistent QNN CPU session; explicit ABI map and native typed buffers only."""
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import threading
from types import SimpleNamespace
import numpy as np

DTYPES = {0x0008:'int8',0x0016:'int16',0x0032:'int32',0x0064:'int64',
          0x0108:'uint8',0x0116:'uint16',0x0132:'uint32',0x0164:'uint64',
          0x0216:'float16',0x0232:'float32',0x0264:'float64',0x0508:'bool'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def all_finite(value):
    """Check every element with a bounded 256 KiB boolean scratch buffer."""
    iterator=np.nditer(value,flags=['external_loop','buffered','zerosize_ok'],
                       op_flags=['readonly'],order='K',buffersize=262144)
    return all(bool(np.isfinite(chunk).all()) for chunk in iterator)

class NonfiniteOutputError(ValueError):
    def __init__(self,name,value):
        super().__init__('QNN output is nonfinite: '+name)
        self.output_name=name
        self.output_value=value

class IntegerOutputRangeError(ValueError):
    def __init__(self,name,value,lower,upper):
        super().__init__('QNN integer output exceeds declared exact range: '+name)
        self.output_name=name
        self.output_value=value
        self.integer_range=[lower,upper]

class NativeSession:
    def __init__(self, bridge, model_lib, backend_lib, abi, resource_hashes):
        self._lock=threading.Lock(); self._handle=None
        for name,path in [('bridge',bridge),('model_lib',model_lib),('backend_lib',backend_lib)]:
            if name not in resource_hashes or sha(path)!=resource_hashes[name]:
                raise ValueError('QNN resource identity differs: '+name)
        if abi.get('schema')!='qnn-native-abi-v1':
            raise ValueError('unsupported QNN ABI map schema')
        lib=C.CDLL(str(bridge)); self._lib=lib
        lib.q4_qnn_error.restype=C.c_char_p
        lib.q4_qnn_create.argtypes=[C.c_char_p,C.c_char_p];lib.q4_qnn_create.restype=C.c_void_p
        lib.q4_qnn_describe.argtypes=[C.c_void_p];lib.q4_qnn_describe.restype=C.c_char_p
        pointer=C.POINTER(C.c_void_p); sizes=C.POINTER(C.c_uint32)
        lib.q4_qnn_execute.argtypes=[C.c_void_p,pointer,sizes,C.c_uint32,pointer,sizes,C.c_uint32]
        lib.q4_qnn_execute.restype=C.c_int
        lib.q4_qnn_destroy.argtypes=[C.c_void_p];lib.q4_qnn_destroy.restype=None
        self._handle=lib.q4_qnn_create(str(backend_lib).encode(),str(model_lib).encode())
        if not self._handle:raise RuntimeError(lib.q4_qnn_error().decode())
        try:
            self.native_abi=json.loads(lib.q4_qnn_describe(self._handle))
            self._maps={}
            for kind in ('inputs','outputs'):
                source=abi[kind];native=self.native_abi[kind]
                if len(source)!=len(native) or len({r['name'] for r in source})!=len(source) or len({r['native_name'] for r in source})!=len(source):
                    raise ValueError('QNN boundary is not a bijection: '+kind)
                byname={r['name']:r for r in native};self._maps[kind]=[]
                for row in source:
                    actual=byname[row['native_name']];dtype=DTYPES[actual['dtype_code']]
                    if dtype!=row['dtype'] or actual['shape']!=row['native_shape']:
                        raise ValueError('actual QNN dtype/extent differs: '+row['name'])
                    if row['shape']!=actual['shape'] and not (row['shape']==[] and actual['shape']==[1]):
                        logical_axes=[v for v in row['shape'] if v!=1]
                        native_axes=[v for v in actual['shape'] if v!=1]
                        if row.get('wire_view')!='singleton_axes' or logical_axes!=native_axes:
                            raise ValueError('unsupported logical/native extent mapping: '+row['name'])
                    expected=math.prod(row['shape'])*np.dtype(dtype).itemsize
                    if expected<=0 or expected!=actual['bytes']:
                        raise ValueError('native QNN byte size differs: '+row['name'])
                    if dtype=='int64' and 'integer_range' not in row:
                        raise ValueError('CPU QNN int64 needs an explicit exact integer range: '+row['name'])
                    if 'integer_range' in row:
                        interval=row['integer_range']
                        if not isinstance(interval,list) or len(interval)!=2 or any(type(v) is not int for v in interval) or interval[0]>interval[1] or not np.issubdtype(np.dtype(dtype),np.integer):
                            raise ValueError('invalid exact integer range: '+row['name'])
                        limits=np.iinfo(dtype)
                        if interval[0]<limits.min or interval[1]>limits.max or (dtype=='int64' and (interval[0]<-2**31 or interval[1]>2**31-1)):
                            raise ValueError('declared range exceeds verified CPU integer semantics: '+row['name'])
                    self._maps[kind].append(dict(row,bytes=expected,dtype_code=actual['dtype_code']))
            self.resource_hashes=dict(resource_hashes)
        except BaseException:
            self.close();raise
    def get_inputs(self):
        return [SimpleNamespace(name=r['name'],shape=r['shape'],type='tensor('+('float' if r['dtype']=='float32' else 'double' if r['dtype']=='float64' else r['dtype'])+')') for r in self._maps['inputs']]
    def get_outputs(self):
        return [SimpleNamespace(name=r['name'],shape=r['shape'],type='tensor('+('float' if r['dtype']=='float32' else 'double' if r['dtype']=='float64' else r['dtype'])+')') for r in self._maps['outputs']]
    def run(self, output_names, feed):
        with self._lock:
            if not self._handle:raise RuntimeError('QNN session is closed')
            if set(feed)!={r['name'] for r in self._maps['inputs']}:raise ValueError('QNN input names differ')
            inputs=[]
            for r in self._maps['inputs']:
                v=feed[r['name']]
                if not isinstance(v,np.ndarray) or v.dtype!=np.dtype(r['dtype']) or list(v.shape)!=r['shape']:
                    raise ValueError('QNN native input dtype/extent differs: '+r['name'])
                if np.issubdtype(v.dtype,np.floating) and not all_finite(v):
                    raise ValueError('QNN input is nonfinite: '+r['name'])
                if 'integer_range' in r:
                    lower,upper=r['integer_range']
                    if not np.issubdtype(v.dtype,np.integer) or not np.greater_equal(v,lower).all() or not np.less_equal(v,upper).all():
                        raise ValueError('QNN integer input exceeds declared exact range: '+r['name'])
                inputs.append(np.array(v,copy=True,order='C'))
            native_order={r['name']:i for i,r in enumerate(self.native_abi['inputs'])}
            inputs=[v for _,v in sorted((native_order[r['native_name']],v) for r,v in zip(self._maps['inputs'],inputs))]
            native_order_out={r['name']:i for i,r in enumerate(self.native_abi['outputs'])}
            buffers=[np.empty(r['shape'],dtype=np.uint8 if r['dtype']=='bool' else r['dtype']) for r in self._maps['outputs']]
            native_buffers=[v for _,v in sorted((native_order_out[r['native_name']],v) for r,v in zip(self._maps['outputs'],buffers))]
            def pointers(values):return (C.c_void_p*len(values))(*(v.ctypes.data for v in values))
            def sizes(values):return (C.c_uint32*len(values))(*(v.nbytes for v in values))
            code=self._lib.q4_qnn_execute(self._handle,pointers(inputs),sizes(inputs),len(inputs),pointers(native_buffers),sizes(native_buffers),len(native_buffers))
            if code:raise RuntimeError(self._lib.q4_qnn_error().decode())
            result={}
            for r,v in zip(self._maps['outputs'],buffers):
                if r['dtype']=='bool':
                    if not np.isin(v,[0,1]).all():raise ValueError('QNN bool output is noncanonical: '+r['name'])
                    v=v.view(np.bool_)
                if np.issubdtype(v.dtype,np.floating) and not all_finite(v):raise NonfiniteOutputError(r['name'],v)
                if 'integer_range' in r:
                    lower,upper=r['integer_range']
                    if not np.issubdtype(v.dtype,np.integer) or not np.greater_equal(v,lower).all() or not np.less_equal(v,upper).all():
                        raise IntegerOutputRangeError(r['name'],v,lower,upper)
                result[r['name']]=v
            names=output_names if output_names is not None else [r['name'] for r in self._maps['outputs']]
            return [result[name] for name in names]
    def close(self):
        with self._lock:
            if self._handle:self._lib.q4_qnn_destroy(self._handle);self._handle=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
