#!/usr/bin/env python3
"""Prepare two official mini scenes for QNN short-sequence inputs, without model execution."""
import argparse,ast,copy,hashlib,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'onnx/qnn'))
from tool_run import sha,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--reference-sha256',required=True);args=p.parse_args()
    if sha(args.reference)!=args.reference_sha256:raise ValueError('frozen mini manifest differs')
    manifest=json.loads(args.reference.read_text());split=manifest['splits']['mini_val'];config=manifest['assets']['config']
    for path,digest in [(split['info_path'],split['info_sha256']),(config['path'],config['sha256'])]:
        if sha(path)!=digest:raise ValueError('frozen input/config identity differs')
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'onnx/fixed').glob('*.py'))}
    current=json.loads((ROOT/'onnx/fixed/SOURCE_MANIFEST.json').read_text())
    for name,value in current['files'].items():
        expected=value['sha256'] if isinstance(value,dict) else value
        if sha(ROOT/'onnx/fixed'/name)!=expected:raise ValueError('fixed source differs from manifest')
    # Import only frozen deployment adapters plus the existing dataset pipeline.
    sys.path[:0]=[str(ROOT/'onnx/fixed'),str(ROOT)]
    import export
    from prepare_scene import frame_metadata
    from mmcv.parallel import collate,scatter
    from mmdet3d.datasets import build_dataset
    import torch
    torch.set_num_threads(4);torch.manual_seed(0)
    os.chdir(ROOT)
    cfg=export.build_cfg(config['path'],'legacy_cuda');test=copy.deepcopy(cfg.data.test);test.ann_file=split['info_path'];test.data_root=manifest['data_root'];test.test_mode=True;test.file_client_args=dict(backend='disk')
    dataset=build_dataset(test)
    if [i['token'] for i in dataset.data_infos]!=split['tokens']:raise ValueError('mini token order differs')
    # Extract the accepted metadata/feed adapter without importing its old run driver.
    driver=ROOT/'onnx/reference/adapters/frame_feed.py'
    tree=ast.parse(driver.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='frame_feed');namespace={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(driver),'exec'),namespace)
    chosen=[];scenes=[]
    for index,info in enumerate(dataset.data_infos):
        scene=info['scene_token']
        if scene not in scenes:
            if len(scenes)==2:break
            scenes.append(scene)
        if sum(dataset.data_infos[i]['scene_token']==scene for i in chosen)<3:chosen.append(index)
    if len(chosen)!=6 or len(scenes)!=2:raise ValueError('need three consecutive frames from each of two scenes')
    run=OUTPUT
    previous={};rows=[]
    for index in chosen:
        info=dataset.data_infos[index]
        if list(info['cams'])!=split['camera_order']:raise ValueError('camera order differs')
        data=scatter(collate([dataset[index]],samples_per_gpu=1),[-1])[0]
        def tensor(value):
            while not isinstance(value,torch.Tensor):
                if hasattr(value,'data') and not isinstance(value,(list,tuple)):value=value.data
                elif isinstance(value,(list,tuple)) and len(value)==1:value=value[0]
                else:raise ValueError('ambiguous dataset tensor wrapper')
            return value
        image=np.ascontiguousarray(tensor(data['img']).numpy());command=int(tensor(data['command']).item());feed,meta,reset=namespace['frame_feed'](info,image,frame_metadata(info),command,previous)
        path=run/f'frame{len(rows):02d}.inputs.npz'
        with path.open('wb') as stream:np.savez(stream,**feed)
        cameras=[dict(camera=name,path=cam['data_path'],sha256=sha(cam['data_path'])) for name,cam in info['cams'].items()]
        rows.append(dict(sequence_index=len(rows),official_mini_index=index,token=info['token'],scene_token=info['scene_token'],new_scene=reset,metadata=meta,inputs=str(path),inputs_sha256=sha(path),input_shapes={k:list(v.shape) for k,v in feed.items()},input_dtypes={k:str(v.dtype) for k,v in feed.items()},camera_sources=cameras))
        previous=meta;save(run/'frame_progress.json',dict(completed=len(rows),last_token=info['token']));print(json.dumps(dict(prepared=len(rows),token=info['token'])),flush=True)
    save(run/'real_frames.json',dict(status='pass',frames=rows,split='mini_val',official_split_frames=split['frames'],scene_order=scenes,reference_manifest_sha256=args.reference_sha256,config_sha256=config['sha256'],info_sha256=split['info_sha256'],fixed_sources=sources,adapter_source=str(driver),adapter_source_sha256=sha(driver),frame_feed_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),script_sha256=sha(__file__),scope='Six real dataset frame inputs only; no state injected, model execution or task acceptance. Legacy bus delta follows this selected two-scene sequence.'))
if __name__=='__main__':
    OUTPUT=Path.cwd();main()
