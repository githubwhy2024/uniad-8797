"""Bind a portable planning runtime to its actually scored development profile."""
import ast
import copy
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from tool_run import ROOT, sha
from session import all_finite


IMPORTS = 'import copy,json,math,os,resource,tempfile,threading,time\nfrom pathlib import Path\nfrom types import SimpleNamespace\nimport numpy as np\nfrom session import NativeSession,NonfiniteOutputError,IntegerOutputRangeError,sha,all_finite\n'
BORROWED = '\nimport borrowed_session\nif Path(borrowed_session.__file__).resolve() != Path(__file__).parent / "borrowed_session.py":\n    raise ValueError("borrowed module must come from the copied runtime")\nNativeSession = borrowed_session.BorrowedNativeSession\n'


def portable_resource_policy(policy):
    """Keep the scored policy, omitting only the development evidence location."""
    if policy is None:return None
    result=copy.deepcopy(policy)
    result['source'].pop('load_control',None)
    return result


def definition(source, name, kind):
    nodes = [n for n in ast.parse(source).body if isinstance(n, kind) and
             (getattr(n, 'name', None) == name if kind is not ast.Assign else
              any(isinstance(t, ast.Name) and t.id == name for t in n.targets))]
    if len(nodes) != 1:
        raise ValueError('scored implementation definition differs: ' + name)
    return nodes[0]


def generated_runtime(partition, ranges, atomic_save, borrowed):
    body = ast.parse(IMPORTS).body + [copy.deepcopy(n) for n in (ranges, atomic_save, partition)]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    return ast.unparse(module) + '\n' + (BORROWED if borrowed else '')


def generated_resource_runtime(partition,ranges,atomic_save,resource_source):
    """Export the same resource implementation against the bound base class.

    Developer imports are removed. Its source-file guard becomes an exact
    base-class AST guard because the portable file also contains journal/range
    definitions. The verifier independently regenerates the complete module.
    No resource operation, validation or execution scope is rewritten here.
    """
    base=generated_runtime(partition,ranges,atomic_save,True)
    imports='import ast,hashlib,inspect,textwrap\nfrom graph_resources import GraphResources,ResourceKey\nfrom borrowed_session import BorrowedNativeSession,TRANSPORT\nsource=SimpleNamespace(**globals())\n'
    expected=hashlib.sha256(ast.dump(partition).encode()).hexdigest()
    replacement=ast.parse("if hashlib.sha256(ast.dump(ast.parse(textwrap.dedent(inspect.getsource(source.PartitionSession))).body[0]).encode()).hexdigest()!=%r:\n    raise ValueError('portable resource base class differs')"%expected).body[0]
    nodes=[];guards=0
    for node in ast.parse(resource_source).body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):continue
        if isinstance(node,ast.If) and ast.dump(node.test)==ast.dump(ast.parse('source.sha(source.__file__)!=SOURCE_SHA256',mode='eval').body):
            nodes.append(replacement);guards+=1
        else:nodes.append(node)
    if guards!=1:raise ValueError('resource source guard extraction differs')
    module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    return base+imports+ast.unparse(module)+'\n'


def historical_source(scored_run, relative, expected):
    run = Path(scored_run)
    terminal = json.loads((run / 'result.json').read_text())
    source_path = run / 'source_identity.json'
    if sha(source_path) != terminal['source_identity_sha256']:
        raise ValueError('scored implementation source identity differs')
    identity = json.loads(source_path.read_text())
    if identity['source_files'][relative] != expected:
        raise ValueError('scored source/profile identity differs')
    source = subprocess.check_output(['git', 'show', identity['source_commit'] + ':' + relative], cwd=ROOT)
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError('scored implementation snapshot unavailable')
    return source.decode()


def bound_source(profile, role, scored_run):
    row = profile['assets'][role]
    path = Path(row['path'])
    if path.is_file() and sha(path) == row['sha256'] and path.stat().st_size == row['bytes']:
        return path.read_text()
    if not path.is_relative_to(ROOT):
        raise ValueError('scored implementation source changed: ' + role)
    return historical_source(scored_run, str(path.relative_to(ROOT)), row['sha256'])


def save_source(profile, scored_run):
    if 'atomic_save_source' in profile['assets']:
        return bound_source(profile, 'atomic_save_source', scored_run)
    # Historical profiles did not expose atomic_save separately. Their durable
    # source identity still binds the complete tool file to a Git snapshot.
    run = Path(scored_run)
    terminal = json.loads((run / 'result.json').read_text())
    source_path = run / 'source_identity.json'
    if sha(source_path) != terminal['source_identity_sha256']:
        raise ValueError('scored atomic save source identity differs')
    identity = json.loads(source_path.read_text())
    relative = 'onnx/qnn/tool_run.py'
    expected = identity['source_files'][relative]
    current = ROOT / relative
    if sha(current) == expected:
        return current.read_text()
    source = subprocess.check_output(['git', 'show', identity['source_commit'] + ':' + relative], cwd=ROOT)
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError('scored atomic save snapshot unavailable')
    return source.decode()


def verify_scored_bundle(bundle, manifest, profile, scored_run):
    """Validate execution identity before portable execution or promotion.

    Only resource locations may differ. Pool schemas, port mappings, lifetimes,
    native ABIs, implementation AST, math precision and policies stay bound.
    This function neither runs a model nor promotes a package.
    """
    bundle = Path(bundle).absolute()
    checks = {}
    def require(label, condition):
        if not condition:
            raise ValueError('portable/scored identity differs: ' + label)
        checks[label] = True
    require('actual_scored_profile',profile==json.loads((Path(scored_run)/'profile.json').read_text()))
    require('model', manifest['lineage']['native_model_sha256'] == profile['assets']['native_model']['sha256'])
    require('public_abi', manifest['profile']['abi'] == profile['abi'])
    require('host_policy', manifest['host_policy'] == profile['host_policy'])
    require('backend', manifest['backend'] == dict(type='QNN_CPU', precision=profile['backend']['precision'], target='x86_64-linux-clang') and profile['backend']['type'] == 'QNN_CPU')
    require('runtime_versions', manifest['runtime_versions'] == dict(profile['runtime_versions'], machine=platform.machine()))
    require('transport_policy', manifest.get('transport_policy') == profile.get('transport_policy'))
    require('graph_resource_policy',manifest.get('graph_resource_policy')==portable_resource_policy(profile.get('graph_resource_policy')))
    expected_parts = [{k: copy.deepcopy(row[k]) for k in ('index', 'inputs', 'outputs', 'drop_after')} for row in profile['plan']['parts']]
    require('cut_dependency_lifetime_table', manifest['profile']['plan']['parts'] == expected_parts)
    require('part_count', len(manifest['profile']['parts']) == len(profile['parts']) == len(expected_parts))
    for portable, scored in zip(manifest['profile']['parts'], profile['parts']):
        require('part_' + str(scored['index']) + '_abi', portable['index'] == scored['index'] and portable['abi'] == scored['abi'] and portable['library_sha256'] == scored['library_sha256'])
        require('part_' + str(scored['index']) + '_path', portable['library'] == manifest['files']['part_' + str(scored['index'])]['path'] and manifest['files']['part_' + str(scored['index'])]['sha256'] == scored['library_sha256'])
        if profile.get('graph_resource_policy') is not None:require('part_'+str(scored['index'])+'_resource_source',portable['native_model_sha256']==scored['native_model_sha256'])
    roles = [('bridge', 'bridge'), ('backend_lib', 'backend_lib'), ('host', 'host'),
             ('state_contract', 'state_contract'), ('initial_state', 'initial_state'),
             ('collision_optimizer', 'collision_optimizer'), ('session_adapter', 'adapter')]
    if profile.get('transport_policy') is not None:
        roles.append(('borrowed_input_adapter', 'borrowed_input_adapter'))
    if profile.get('graph_resource_policy') is not None:
        roles.append(('graph_resource_helper','graph_resource_helper'))
        require('resource_source_lineage',manifest['lineage']['resource_source_sha256']==profile['assets']['resource_adapter']['sha256'])
    pool = copy.deepcopy(profile['plan'].get('shared_constants'))
    expected_pool = copy.deepcopy(pool)
    if expected_pool is not None:
        expected_pool.pop('path')
        roles.append(('shared_constant_pool', 'shared_constant_pool'))
    require('shared_pool_schema_and_views', manifest['profile'].get('shared_constants') == expected_pool)
    require('shared_pool_presence', ('shared_constant_pool' in manifest['files']) == (pool is not None))
    for role, source_role in roles:
        require(role + '_source_binding', manifest['files'][role]['sha256'] == profile['assets'][source_role]['sha256'] and manifest['files'][role]['bytes'] == profile['assets'][source_role]['bytes'])
    partition = definition(bound_source(profile, 'partition_adapter', scored_run), 'PartitionSession', ast.ClassDef)
    ranges = definition(bound_source(profile, 'profile', scored_run), 'RANGES', ast.Assign)
    atomic_save = definition(save_source(profile, scored_run), 'save', ast.FunctionDef)
    expected_text = generated_runtime(partition, ranges, atomic_save, profile.get('transport_policy') is not None)
    if profile.get('graph_resource_policy') is not None:expected_text=generated_resource_runtime(partition,ranges,atomic_save,bound_source(profile,'resource_adapter',scored_run))
    generated = bundle / manifest['files']['partition_adapter']['path']
    require('generated_execution_ast', ast.dump(ast.parse(generated.read_text())) == ast.dump(ast.parse(expected_text)))
    for key, node in [('partition_class_ast_sha256', partition), ('ranges_ast_sha256', ranges), ('atomic_save_ast_sha256', atomic_save)]:
        require(key, manifest['lineage'][key] == hashlib.sha256(ast.dump(node).encode()).hexdigest())
    require('partition_source_sha256', manifest['lineage']['partition_source_sha256'] == profile['assets']['partition_adapter']['sha256'])
    for role, row in manifest['files'].items():
        path = bundle / row['path']
        require(role + '_ordinary_file', not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts and path.resolve() == path and path.is_file())
        require(role + '_actual_bytes', path.stat().st_size == row['bytes'] and sha(path) == row['sha256'])
    if pool is not None:
        path = bundle / manifest['files']['shared_constant_pool']['path']
        rows = pool['entries']
        require('shared_names_unique', len({row['source_name'] for row in rows}) == len(rows))
        with np.load(path, allow_pickle=False) as archive:
            require('shared_key_closure', set(archive.files) == {row['key'] for row in rows})
            buffers = {key: archive[key] for key in archive.files}
        hashes = {}
        for key, data in buffers.items():
            require(key + '_physical_buffer', data.ndim == 1 and data.flags.c_contiguous and all_finite(data))
            hashes[key] = hashlib.sha256(memoryview(data).cast('B')).hexdigest()
        for row in rows:
            data = buffers[row['key']]
            require(row['source_name'] + '_view', data.dtype == np.dtype(row['dtype']) and data.nbytes == row['bytes'] and data.size == __import__('math').prod(row['shape']) and hashes[row['key']] == row['data_sha256'] and row['key'] == 'sha_' + hashes[row['key']])
        require('shared_byte_accounting', pool['unique_data_bytes'] == sum(v.nbytes for v in buffers.values()) and pool['logical_view_bytes'] == sum(row['bytes'] for row in rows))
    return dict(status='pass', checks=checks, scored_profile_sha256=sha(Path(scored_run) / 'profile.json'), scope='Actual full execution identity and resource closure only; no neural/task/target acceptance created.')
