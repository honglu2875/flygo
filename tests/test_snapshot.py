"""Concurrent deployment must never publish an incomplete or misidentified source."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts import cluster
from flygo.storage import GIB, Limits, StorageBudget


class Snapshot(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'runtime'
        self.repo = Path(self.temporary.name) / 'repo'
        source = self.repo / 'python/flygo/__init__.py'
        source.parent.mkdir(parents=True); source.write_text('version = 1\n')
        installed = self.root / 'venv/lib/python3.12/site-packages'
        for name in ('flygo', 'numpy', 'numpy.libs', 'numpy-1.0.dist-info'):
            (installed / name).mkdir(parents=True)
        (installed / 'flygo/_native.test.so').write_bytes(b'native fixture')
        (installed / 'numpy/__init__.py').write_text('numpy fixture')
        (installed / 'numpy-1.0.dist-info/METADATA').write_text('Name: numpy\nVersion: 1.0\n')
        budget = StorageBudget(self.root, Limits(files_cap=4*GIB, free_files_floor=0, available_memory_floor=0))
        for context in (patch.object(cluster, 'REPO', self.repo),
                        patch.object(cluster, 'GIB', 1 << 20),
                        patch.object(cluster, 'StorageBudget', return_value=budget)):
            context.start(); self.addCleanup(context.stop)

    def test_concurrent_launch_waits_for_atomic_publication(self):
        entered, release = threading.Event(), threading.Event()
        original = shutil.copytree

        def delayed(source, *args, **kwargs):
            if Path(source).name == 'numpy':
                entered.set()
                if not release.wait(5): raise TimeoutError('Fixture was not released')
            return original(source, *args, **kwargs)

        with patch.object(cluster.shutil, 'copytree', side_effect=delayed), ThreadPoolExecutor(2) as pool:
            first = pool.submit(cluster.snapshot, self.root)
            try:
                self.assertTrue(entered.wait(5))
                second = pool.submit(cluster.snapshot, self.root)
                self.assertFalse(any(p.is_dir() and not p.name.startswith('.')
                                     for p in (self.root/'environments').iterdir()))
                self.assertFalse(second.done())
            finally:
                release.set()
            self.assertEqual(first.result(), second.result())
        self.assertTrue((first.result() / 'snapshot.json').is_file())

    def test_failure_leaves_no_published_environment_and_retry_succeeds(self):
        with patch.object(cluster.shutil, 'copytree', side_effect=OSError('copy failed')):
            with self.assertRaises(OSError): cluster.snapshot(self.root)
        self.assertFalse(any(p.is_dir() for p in (self.root / 'environments').iterdir()))
        self.assertTrue((cluster.snapshot(self.root) / 'snapshot.json').is_file())

    def test_edit_during_copy_cannot_change_the_frozen_source_bytes(self):
        original = shutil.copytree
        source = self.repo / 'python/flygo/__init__.py'

        def edit(*args, **kwargs):
            source.write_text('version = 2\n')
            return original(*args, **kwargs)

        with patch.object(cluster.shutil, 'copytree', side_effect=edit):
            first = cluster.snapshot(self.root)
        self.assertEqual((first / 'site-packages/flygo/__init__.py').read_text(), 'version = 1\n')
        second = cluster.snapshot(self.root)
        self.assertNotEqual(first, second)
        self.assertEqual((second / 'site-packages/flygo/__init__.py').read_text(), 'version = 2\n')


if __name__ == '__main__': unittest.main()
