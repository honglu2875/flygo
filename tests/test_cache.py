import gc
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from flygo.data.cache import CACHE_VERSION, LEGACY_VERSION, cache_directory, load_cached, trim_caches
from flygo.data.corpus import atomic_json, identity
from flygo.data.loader import LAYOUT
from flygo.storage import Limits, StorageBudget, allocated_bytes


def fixture(root, label, budget, positions=128):
    manifest = dict(dataset_id=identity(label), positions=positions)

    def build(directory):
        arrays = {}
        for i, (name, (dtype, shape)) in enumerate(LAYOUT.items()):
            arrays[name] = np.lib.format.open_memmap(directory / (name + '.npy'), mode='w+',
                                                     dtype=dtype, shape=(positions, *shape))
            arrays[name][...] = i
        indexes = dict(train=np.arange(positions // 2, dtype=np.int64),
                       validation=np.arange(positions // 2, positions, dtype=np.int64),
                       validation_novel=np.arange(positions // 2, positions, dtype=np.int64))
        return arrays, indexes

    with patch('flygo.data.cache.StorageBudget', return_value=budget):
        return manifest, load_cached(root, manifest, build)


def budget_for(root, cap=32 * 1024**2):
    return StorageBudget(root, Limits(files_cap=cap, free_files_floor=0, available_memory_floor=0))


def request_pressure(root, budget):
    return budget.limits.files_cap - allocated_bytes(root) + 128 * 1024


class FeatureCacheLeases(unittest.TestCase):
    def test_pressure_skips_live_views_and_evicts_oldest_unused_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); budget = budget_for(root)
            active, held = fixture(root, 'active', budget)
            old, unused = fixture(root, 'old', budget)
            newer, recent = fixture(root, 'newer', budget)
            view = np.asarray(held[0]['features'])[1:3]
            del held, unused, recent
            gc.collect()
            for when, manifest in enumerate((active, old, newer), 1):
                os.utime(cache_directory(root, manifest['dataset_id']).with_suffix('.used'), (when, when))
            sentinels = [root / 'corpus/game.npz', root / 'runs/checkpoint.npz', root / 'cache/unrelated/data']
            for path in sentinels:
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'preserve')
            result = trim_caches(root, files=request_pressure(root, budget), budget=budget)
            self.assertEqual(result['status'], 'admitted')
            self.assertEqual(result['busy'], [active['dataset_id']])
            self.assertEqual([r['dataset_id'] for r in result['evicted']], [old['dataset_id']])
            self.assertTrue(cache_directory(root, active['dataset_id']).exists())
            self.assertTrue(cache_directory(root, newer['dataset_id']).exists())
            self.assertTrue(all(p.read_bytes() == b'preserve' for p in sentinels))
            np.testing.assert_array_equal(view, 0)
            del view; gc.collect()
            result = trim_caches(root, files=request_pressure(root, budget), budget=budget)
            self.assertEqual(result['evicted'][0]['dataset_id'], active['dataset_id'])

    def test_cache_build_reclaims_only_unused_cache_under_real_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); budget = budget_for(root, 26 * 1024**2)
            active, held = fixture(root, 'active', budget, 1024)
            unused, maps = fixture(root, 'unused', budget, 1024)
            del maps; gc.collect()
            new, maps = fixture(root, 'new', budget, 1024)
            self.assertTrue(cache_directory(root, active['dataset_id']).exists())
            self.assertFalse(cache_directory(root, unused['dataset_id']).exists())
            self.assertTrue(cache_directory(root, new['dataset_id']).exists())
            np.testing.assert_array_equal(held[0]['features'], maps[0]['features'])
            budget.check()

    def test_legacy_and_unregistered_files_remain_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); budget = budget_for(root)
            old, maps = fixture(root, 'legacy', budget)
            current = cache_directory(root, old['dataset_id'])
            del maps; gc.collect()
            legacy = root / 'cache/features' / LEGACY_VERSION / old['dataset_id']
            legacy.parent.mkdir(parents=True); current.rename(legacy)
            receipt = json.loads((legacy / 'manifest.json').read_text())
            receipt['version'] = LEGACY_VERSION; receipt.pop('lease_protocol'); receipt.pop('cache_id')
            receipt['cache_id'] = identity(receipt); atomic_json(legacy / 'manifest.json', receipt)
            arrays, indexes = load_cached(root, old, lambda _: self.fail('Legacy cache must be reused'))
            self.assertEqual(cache_directory(root, old['dataset_id']), legacy)
            del arrays, indexes; gc.collect()
            suspect, maps = fixture(root, 'unregistered', budget)
            suspect_path = cache_directory(root, suspect['dataset_id'])
            del maps; gc.collect()
            (suspect_path / 'unregistered.txt').write_text('preserve')
            result = trim_caches(root, files=request_pressure(root, budget), budget=budget)
            self.assertEqual(result['status'], 'pressure')
            self.assertFalse(result['evicted'])
            self.assertTrue(legacy.exists())
            self.assertEqual((suspect_path / 'unregistered.txt').read_text(), 'preserve')

    def test_another_process_holds_a_lease_until_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); budget = budget_for(root)
            manifest, maps = fixture(root, 'process', budget)
            del maps; gc.collect()
            code = '''import json,sys,numpy as np
from pathlib import Path
from flygo.data.cache import load_cached
a,i=load_cached(Path(sys.argv[1]),json.loads(sys.argv[2]),lambda _:None)
view=np.asarray(a['features'])[1:]
del a,i
print('ready',flush=True)
sys.stdin.buffer.read(1)
assert view[0,0,0,0]==0
'''
            process = subprocess.Popen([sys.executable, '-B', '-c', code, str(root), json.dumps(manifest)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    self.assertTrue(selector.select(10), 'Reader did not start')
                self.assertEqual(process.stdout.readline(), b'ready\n')
                request = request_pressure(root, budget)
                result = trim_caches(root, files=request, budget=budget)
                self.assertEqual(result['status'], 'pressure')
                self.assertEqual(result['busy'], [manifest['dataset_id']])
                process.kill(); process.wait(timeout=10)
                result = trim_caches(root, files=request, budget=budget)
                self.assertEqual(result['status'], 'admitted')
                self.assertEqual(result['evicted'][0]['dataset_id'], manifest['dataset_id'])
                self.assertTrue((root / 'cache/features' / CACHE_VERSION /
                                 (manifest['dataset_id'] + '.lock')).exists())
            finally:
                if process.poll() is None: process.kill(); process.wait(timeout=10)
                for stream in (process.stdin, process.stdout, process.stderr): stream.close()


if __name__ == '__main__':
    unittest.main()
