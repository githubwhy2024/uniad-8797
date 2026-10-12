"""Account for retained cut values and explicit user-buffer copies, without NN."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--short-run', type=Path, required=True)
    args = parser.parse_args()
    run = args.short_run.absolute()
    profile_path = run / 'profile.json'
    status, result = [json.loads((run / name).read_text()) for name in ('status.json', 'result.json')]
    if status['status'] != 'pass' or result['status'] != 'pass' or status['result_sha256'] != sha(run / 'result.json') or result['artifacts']['profile.json']['sha256'] != sha(profile_path):
        raise ValueError('memory accounting source must be a bound terminal profile')
    profile = json.loads(profile_path.read_text())
    sizes = {'float32': 4, 'float64': 8, 'int64': 8, 'int32': 4, 'bool': 1}
    initial = {row['name']: math.prod(row['shape']) * sizes[row['dtype']] for row in profile['abi']['inputs']}
    live = dict(initial)
    initial_names = set(initial)
    caller_retained = sum(initial.values())
    rows = []
    for part in profile['plan']['parts']:
        for item in part['inputs']:
            if live.get(item['source_name']) != item['bytes']:
                raise ValueError('cut input is not a live source extent: ' + item['source_name'])
        new_names = [row['source_name'] for row in part['outputs']]
        if len(set(new_names)) != len(new_names) or any(name in live for name in new_names):
            raise ValueError('cut outputs overwrite live storage')
        frontier = caller_retained + sum(size for name, size in live.items() if name not in initial_names)
        copied = sum(row['bytes'] for row in part['inputs'])
        allocated = sum(row['bytes'] for row in part['outputs'])
        rows.append(dict(index=part['index'], caller_and_live_value_bytes=frontier,
                         input_pack_copy_bytes=copied, fresh_output_bytes=allocated,
                         baseline_explicit_buffer_bytes=frontier + copied + allocated,
                         borrowed_contiguous_buffer_bytes=frontier + allocated))
        live.update({row['source_name']: row['bytes'] for row in part['outputs']})
        for name in part['drop_after']:
            # Root constants may be listed here without a Python values entry.
            # Match the runtime's values.pop(name, None).
            live.pop(name, None)
    total_input = sum(row['input_pack_copy_bytes'] for row in rows)
    total_output = sum(row['fresh_output_bytes'] for row in rows)
    report = dict(status='pass', profile_sha256=sha(profile_path), parts=len(rows),
                  initial_caller_buffers_bytes=caller_retained,
                  cumulative_input_pack_copy_bytes_per_frame=total_input,
                  cumulative_fresh_output_extents_bytes_per_frame=total_output,
                  largest_explicit_baseline_buffers=max(rows, key=lambda row: row['baseline_explicit_buffer_bytes']),
                  largest_explicit_borrowed_buffers=max(rows, key=lambda row: row['borrowed_contiguous_buffer_bytes']),
                  rows=rows, measured_peak_rss_kib=json.loads((run / 'neural_result.json').read_text())['peak_rss_kib'],
                  profile_binding=dict(path=str(profile_path), sha256=sha(profile_path)),
                  scope='Static accounting of Python-visible retained source arrays and native user-buffer extents. '
                        'Caller input arrays stay retained for the full synchronous call. Assumes distinct newly '
                        'allocated outputs as in source NativeSession.run. Excludes SDK activations/scratch/allocator '
                        'pools, weights, Host copies, interpreter and page cache; neither figure is predicted process '
                        'RSS or board memory. Borrowed figure assumes suitable contiguous/aligned/writable inputs.')
    out = Path.cwd() / 'partition_memory.json'
    if out.exists():
        raise ValueError('preserve existing memory observation')
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('largest_explicit_baseline_buffers', 'largest_explicit_borrowed_buffers')}))


if __name__ == '__main__':
    main()
