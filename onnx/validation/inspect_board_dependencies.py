#!/usr/bin/env python3
"""Inventory installed preboard assets without selecting an unknown board target."""
import ast,importlib.metadata,json,os,platform,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import SDK,sha,save

def command(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
 return dict(argv=args,exit_code=p.returncode,output=p.stdout[:4000])

def compiler_expression(node):
 if isinstance(node,ast.Constant) and isinstance(node.value,str):return node.value
 if isinstance(node,ast.Name) and node.id=='CLANG':return 'clang++-9' if shutil.which('clang++-9') else 'clang++'
 if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):return compiler_expression(node.left)+compiler_expression(node.right)
 raise ValueError('unsupported SDK compiler expression: '+ast.dump(node))

def main():
 root=Path.cwd();generator=SDK/'bin/x86_64-linux-clang/qnn-model-lib-generator';tree=ast.parse(generator.read_text());targets=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='LinuxTarget':
   keys={k.arg:k.value for k in node.keywords};row={}
   for k in ('name','alias','compiler','sys_root'):
    if k in keys:
     row[k]=compiler_expression(keys[k])
   if 'alias' not in row:continue
   compiler=row.get('compiler','');expanded=os.path.expandvars(compiler)
   row['compiler_expanded']=expanded;row['compiler_executable']=shutil.which(expanded) if '$' not in expanded else None
   row['status']='tool_visible_target_unselected' if row['compiler_executable'] else 'toolchain_not_resolved_in_current_environment';targets.append(row)
 targets.append(dict(alias='aarch64-android',compiler='ndk-build',compiler_executable=shutil.which('ndk-build'),status='target_unselected' if shutil.which('ndk-build') else 'toolchain_not_resolved_in_current_environment'))
 libraries=[]
 for directory in sorted((SDK/'lib').iterdir()):
  if not directory.is_dir() or directory.name in ('python','android'):continue
  for name in ('libQnnCpu.so','libQnnHtp.so','libQnnSystem.so'):
   p=directory/name
   if p.is_file():
    elf=command(['readelf','-h',str(p)]);machine=next((line.strip() for line in elf['output'].splitlines() if 'Machine:' in line),None)
    libraries.append(dict(sdk_target_directory=directory.name,path=str(p),sha256=sha(p),bytes=p.stat().st_size,elf_machine=machine,status='installed_asset_only_not_model_or_board_acceptance'))
 host={}
 for package in ('numpy','casadi','onnx','onnxruntime'):
  try:host[package]=importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:host[package]=None
 tools={name:shutil.which(name) for name in ('clang++','g++','aarch64-linux-gnu-g++','ndk-build','adb','qdl','readelf')}
 versions={name:command([path,'--version']) for name,path in tools.items() if path and name in ('clang++','g++','aarch64-linux-gnu-g++')}
 docs=SDK/'docs/QNN/general/htp/htp_backend.html'
 config=dict(schema='qnn-board-configuration-draft-v1',status='required_information_pending',os=None,abi=None,soc_model=None,soc_id=None,dsp_architecture=None,device_sdk_version=None,driver_fastrpc_version=None,backend=None,precision_policy='float32; quantization and voluntary FP16 deferred',device_library_search_path=None,host_python_numpy_support=None,collision_solver_casadi_support=None)
 save(root/'board_configuration.draft.json',config)
 report=dict(status='pass',inspection_status='complete',board_readiness='not_accepted',source_generator=str(generator),source_generator_sha256=sha(generator),sdk=str(SDK),host=dict(machine=platform.machine(),python=platform.python_version(),packages=host),visible_tools=tools,actual_compiler_version_checks=versions,installed_model_lib_targets=targets,installed_runtime_libraries=libraries,htp_document=dict(path=str(docs),sha256=sha(docs),evidence_kind='installed_document_only') if docs.is_file() else None,unresolved=[k for k,v in config.items() if v is None],next_actions=['Confirm target OS/ABI/SoC/DSP/driver/SDK before selecting any SDK target or context configuration.','Build model and API bridge with the confirmed toolchain in an ordinary Q4 run and bind actual output hashes.','Verify ARM Python/NumPy and CasADi/optimizer dependencies independently; current x86 host versions are not ARM deployment proof.','Execute and score the exact pinned QNN profile on board; CPU acceptance and installed libraries do not establish HTP support or performance.'],draft_sha256=sha(root/'board_configuration.draft.json'),script_sha256=sha(__file__),scope='Actual installed-file/toolchain/version inventory only; no guessed 8797 configuration, target model build, offline HTP context, quantization or board execution.')
 save(root/'dependency_inventory.json',report)
if __name__=='__main__':main()
