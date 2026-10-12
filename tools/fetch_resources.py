#!/usr/bin/env python3
"""Fetch the pinned baseline assets from this repository's GitHub Release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def destination(relative):
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError('resource destination escapes project')
    result = ROOT / value
    if result.resolve() != result or not result.is_relative_to(ROOT):
        raise ValueError('ordinary project resource path required')
    return result


def matches(path, row):
    return path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']


def atomic_copy(source, target, row):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.part')
    if temporary.is_symlink():
        raise ValueError('resource staging path is a symlink')
    try:
        with temporary.open('wb') as out:
            shutil.copyfileobj(source, out, 8 * 1024**2)
        if not matches(temporary, row):
            raise ValueError('resource size/hash differs: ' + str(target.relative_to(ROOT)))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-dir', type=Path, help='Use an already downloaded Release directory')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'onnx/resources/manifest.json').read_text())
    resources = manifest['resources']
    missing = [name for name, row in resources.items() if not matches(destination(row['path']), row)]
    if args.verify_only:
        if missing:
            raise SystemExit('Missing or changed resources: ' + ', '.join(missing))
        print('All baseline resources match their pinned sizes and SHA256 hashes.')
        return
    for asset in manifest['release_assets']:
        selected = [name for name in asset['resources'] if name in missing]
        if not selected:
            continue
        if asset.get('archive'):
            target = destination('.local/downloads/' + asset['name'])
        else:
            target = destination(resources[asset['resources'][0]]['path'])
        if not matches(target, asset):
            if args.from_dir:
                source_path = args.from_dir / asset['name']
                with source_path.open('rb') as stream:
                    atomic_copy(stream, target, asset)
            else:
                url = manifest['release_base_url'] + '/' + manifest['release_tag'] + '/' + asset['name']
                print('Downloading ' + asset['name'], flush=True)
                with urllib.request.urlopen(url, timeout=120) as stream:
                    atomic_copy(stream, target, asset)
        if asset.get('archive'):
            expected = {resources[name]['path']: resources[name] for name in asset['resources']}
            with zipfile.ZipFile(target) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or set(names) != set(expected):
                    raise ValueError('archive resource coverage differs')
                for name in selected:
                    row = resources[name]
                    info = archive.getinfo(row['path'])
                    if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('regular archive file required')
                    with archive.open(info) as stream:
                        atomic_copy(stream, destination(row['path']), row)
    for name, row in resources.items():
        if not matches(destination(row['path']), row):
            raise SystemExit('Resource verification failed: ' + name)
    print('All baseline resources are ready.')


if __name__ == '__main__':
    main()
