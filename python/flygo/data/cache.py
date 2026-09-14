"""Immutable feature caches with reader leases and pressure-driven reclamation.

Only the v2 namespace is evictable. Legacy caches may have readers running an
older immutable environment, so they remain protected and can still be reused.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import fcntl
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import time
import uuid
import weakref

import numpy as np

from .corpus import atomic_json, identity
from ..qualify import sha256
from ..storage import StorageBudget, StoragePressure, GIB


LEGACY_VERSION = 'go-9x9-h4-fp32-d4-v1'
CACHE_VERSION = 'go-9x9-h4-fp32-d4-v2'
LEASE_PROTOCOL = 'shared-flock-mmap-lifetime-v1'


def cache_directory(root: Path, dataset_id: str) -> Path:
    if not re.fullmatch(r'[0-9a-f]{64}', dataset_id):
        raise ValueError('Expected a SHA-256 dataset identity')
    parent = root / 'cache/features'
    current, legacy = (parent / version / dataset_id for version in (CACHE_VERSION, LEGACY_VERSION))
    # Reuse already qualified bytes; a running old reader cannot be retrofitted
    # with a lease, so reuse never makes a legacy directory evictable.
    return legacy if legacy.exists() and not current.exists() else current


def _receipt(directory: Path) -> dict:
    if directory.is_symlink():
        raise ValueError('Feature cache must be an owned directory')
    receipt = json.loads((directory / 'manifest.json').read_text())
    if (receipt['version'] != directory.parent.name or receipt['dataset_id'] != directory.name
            or identity({k: v for k, v in receipt.items() if k != 'cache_id'}) != receipt['cache_id']):
        raise ValueError('Feature cache identity mismatch')
    if receipt['version'] == CACHE_VERSION and receipt.get('lease_protocol') != LEASE_PROTOCOL:
        raise ValueError('Feature cache lease protocol mismatch')
    if any(Path(name).name != name or not name.endswith('.npy') for name in receipt['files']):
        raise ValueError('Feature cache contains an invalid file name')
    if any(p.is_symlink() or not p.is_file() for p in directory.iterdir()):
        raise ValueError('Feature cache contains unsupported entries')
    if {p.name for p in directory.iterdir()} != {'manifest.json', *receipt['files']}:
        raise ValueError('Feature cache contains unregistered files')
    return receipt


def trim_caches(root: Path, *, files=0, heap=0, budget=None) -> dict:
    """Reclaim oldest unused v2 caches only while the proposed request cannot fit.

    Reader leases and the writer use the same persistent lock inode. Never
    delete a lock file: a new inode would not protect existing memory maps.
    The caller must still reserve its allocation after this advisory check.
    """
    budget = budget or StorageBudget(root)
    result = dict(status='admitted', evicted=[], busy=[], protected=[])
    parent = root / 'cache/features' / CACHE_VERSION

    def admitted():
        try:
            result['snapshot'] = budget.check(files=files, heap=heap)
            result.pop('reason', None)
            result['status'] = 'admitted'
            return True
        except StoragePressure as error:
            result.update(status='pressure', reason=str(error))
            return False

    if admitted() or not parent.exists():
        return result
    if parent.is_symlink():
        raise ValueError('Feature cache namespace must not be a symlink')
    if files > budget.limits.files_cap:
        return result

    def age(directory):
        access = directory.with_suffix('.used')
        try:
            return access.stat().st_mtime if access.exists() else directory.stat().st_mtime
        except FileNotFoundError:
            # Another trimmer may remove a candidate between discovery and sort.
            # The persistent lock and existence check below remain authoritative.
            return float('inf')

    candidates = [p for p in parent.iterdir() if re.fullmatch(r'[0-9a-f]{64}', p.name)]
    for directory in sorted(candidates, key=lambda p: (age(p), p.name)):
        with directory.with_suffix('.lock').open('a+b') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                result['busy'].append(directory.name)
                continue
            if not directory.exists():
                continue
            try:
                receipt = _receipt(directory)
            except (OSError, ValueError, KeyError, TypeError) as error:
                result['protected'].append(dict(dataset_id=directory.name, reason=str(error)))
                continue
            # Candidate membership and the lease define ownership. Raw games,
            # releases, checkpoints, legacy caches and unrelated paths are never
            # candidates, even when admission remains impossible.
            shutil.rmtree(directory)
            result['evicted'].append(dict(dataset_id=directory.name, cache_id=receipt['cache_id']))
        if admitted():
            return result
    admitted()
    return result


@contextmanager
def _reservation(root, **request):
    budget = StorageBudget(root)
    with ExitStack() as stack:
        try:
            stack.enter_context(budget.reserve(**request))
        except StoragePressure:
            result = trim_caches(root, files=request['files'], heap=request['heap'], budget=budget)
            if result['status'] != 'admitted':
                raise StoragePressure('Unused caches cannot satisfy admission: ' + result['reason'])
            # Concurrent reservations can still refuse this request. Recheck
            # atomically; never weaken the shared memory floors.
            stack.enter_context(budget.reserve(**request))
        yield


def load_cached(root: Path, manifest: dict, build):
    """Return read-only maps; shared locks survive arrays and derived views."""
    from .loader import LAYOUT
    directory = cache_directory(root, manifest['dataset_id'])
    directory.parent.mkdir(parents=True, exist_ok=True)
    with directory.with_suffix('.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        if not directory.exists():
            fcntl.flock(lock, fcntl.LOCK_UN)
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not directory.exists():
                size = manifest['positions'] * sum(np.dtype(dtype).itemsize * int(np.prod(shape))
                                                   for dtype, shape in LAYOUT.values())
                with _reservation(root, files=size + manifest['positions'] * 24 + 16 * 1024**2,
                                  heap=2 * GIB, purpose='shared feature cache ' + manifest['dataset_id']):
                    temporary = directory.with_name('.' + directory.name + '.' + uuid.uuid4().hex + '.staging')
                    temporary.mkdir()
                    try:
                        arrays, indexes = build(temporary)
                        for array in arrays.values():
                            array.flush()
                        for name, index in indexes.items():
                            np.save(temporary / ('index-' + name + '.npy'), index, allow_pickle=False)
                        files = {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p))
                                 for p in sorted(temporary.glob('*.npy'))}
                        receipt = dict(version=CACHE_VERSION, lease_protocol=LEASE_PROTOCOL,
                                       dataset_id=manifest['dataset_id'], positions=manifest['positions'],
                                       arrays=list(arrays), indexes=list(indexes), files=files, created=time.time())
                        receipt['cache_id'] = identity(receipt)
                        atomic_json(temporary / 'manifest.json', receipt)
                        del arrays
                        temporary.rename(directory)
                    finally:
                        if temporary.exists():
                            shutil.rmtree(temporary)
            fcntl.flock(lock, fcntl.LOCK_SH)
        receipt = _receipt(directory)
        if receipt['positions'] != manifest['positions']:
            raise ValueError('Feature cache position count mismatch')
        if set(receipt['arrays']) != set(LAYOUT) or not set(receipt['indexes']) <= {
                'train', 'validation', 'test', 'validation_novel'}:
            raise ValueError('Feature cache column contract mismatch')
        expected = {name + '.npy' for name in receipt['arrays']} | {
            'index-' + name + '.npy' for name in receipt['indexes']}
        if set(receipt['files']) != expected:
            raise ValueError('Feature cache file contract mismatch')
        for name, record in receipt['files'].items():
            if sha256(directory / name) != record['sha256']:
                raise ValueError('Feature cache hash mismatch: ' + name)

        def load(name):
            array = np.load(directory / (name + '.npy'), mmap_mode='r', allow_pickle=False)
            if not isinstance(array.base, mmap.mmap):
                raise ValueError('Expected a directly memory-mapped cache array')
            # Duplicates share this flock's open-file description. The kernel
            # releases it after the last mapping/view (or process) dies. Attach
            # to the backing mmap, so np.asarray does not lose the lease.
            weakref.finalize(array.base, os.close, os.dup(lock.fileno()))
            return array

        arrays = {name: load(name) for name in receipt['arrays']}
        indexes = {name: load('index-' + name) for name in receipt['indexes']}
        directory.with_suffix('.used').touch()
        return arrays, indexes
