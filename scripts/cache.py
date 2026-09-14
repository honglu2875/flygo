#!/usr/bin/env python3
"""Prepare a frozen feature cache or reclaim unused leased caches under pressure."""
import argparse
import json
from pathlib import Path
import time

from flygo.data.cache import cache_directory, trim_caches
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release
from flygo.runtime import pin
from flygo.storage import StorageBudget, GIB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'trim'])
    parser.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    parser.add_argument('--release', default='v0-1m')
    parser.add_argument('--cpus', default='117,118,119')
    parser.add_argument('--files', type=int, default=0, help='Proposed file allocation in bytes for trim')
    parser.add_argument('--heap', type=int, default=0, help='Proposed heap allocation in bytes for trim')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); pin(list(map(int, args.cpus.split(','))))
    if args.action == 'trim':
        result = trim_caches(args.root, files=args.files, heap=args.heap)
    else:
        started = time.time()
        with StorageBudget(args.root).reserve(files=1024**2, heap=3*GIB, purpose='prepare frozen features'):
            manifest, arrays, indexes = load_release(args.root, args.root/'releases'/args.release/'manifest.json', cache=True)
            path = cache_directory(args.root, manifest['dataset_id'])
            receipt = json.loads((path/'manifest.json').read_text())
            result = dict(status='passed', dataset_id=manifest['dataset_id'], cache_id=receipt['cache_id'],
                          path=str(path), positions=len(arrays['features']),
                          splits={k:len(v) for k,v in indexes.items()}, seconds=time.time()-started)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
