"""Keep incomplete, conflicting or corrupted evidence out of the paired analysis."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from collect_clipping import archive_members, extract_verified, inspect
from flygo.storage import Limits, StorageBudget


class ClippingCollection(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); self.archive = self.root / 'evidence.tar'

    def bundle(self, files, *, extra=None, duplicate=False, override=None):
        manifest = dict(files={name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                              for name, data in files.items()})
        if override:
            manifest['files'].update(override)
        payload = json.dumps(manifest).encode(); name = 'archive/manifest.json'
        with tarfile.open(self.archive, 'w') as tar:
            for path, data in [*files.items(), (name, payload), *(extra or [])]:
                member = tarfile.TarInfo(path); member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
                if duplicate:
                    tar.addfile(member, io.BytesIO(data)); duplicate = False
        return name, hashlib.sha256(payload).hexdigest()

    def extract(self, name, digest):
        def budget(root):
            return StorageBudget(root, Limits(files_cap=1 << 25, free_files_floor=0, available_memory_floor=0))
        with patch('collect_clipping.StorageBudget', side_effect=budget):
            return extract_verified(self.root, self.archive, name, digest)

    def test_real_archive_roundtrip_and_idempotent_extraction(self):
        files = {'runs/a/metrics.jsonl': b'fixed metrics', 'runs/b/response.npz': bytes(range(256)) * 8192}
        name, digest = self.bundle(files)
        self.assertEqual(self.extract(name, digest)['files'], 2)
        for path, data in files.items():
            self.assertEqual((self.root / path).read_bytes(), data)
        self.assertEqual(self.extract(name, digest)['extracted'], 0)

    def test_conflict_does_not_overwrite_or_publish_other_members(self):
        name, digest = self.bundle({'new': b'payload', 'existing': b'expected'})
        (self.root / 'existing').write_bytes(b'conflict')
        with self.assertRaisesRegex(ValueError, 'Immutable evidence conflict'):
            self.extract(name, digest)
        self.assertEqual((self.root / 'existing').read_bytes(), b'conflict')
        self.assertFalse((self.root / 'new').exists())

    def test_member_corruption_manifest_change_and_duplicate_are_rejected(self):
        name, digest = self.bundle({'payload': b'abc'}, override={'payload': dict(bytes=3, sha256='0' * 64)})
        with self.assertRaisesRegex(ValueError, 'checksum differs'):
            self.extract(name, digest)
        self.assertFalse((self.root / 'payload').exists())
        name, digest = self.bundle({'payload': b'abc'})
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            archive_members(self.root, self.archive, name, '0' * 64)
        name, digest = self.bundle({'payload': b'abc'}, duplicate=True)
        with self.assertRaisesRegex(ValueError, 'unique regular'):
            archive_members(self.root, self.archive, name, digest)

    def test_escaping_paths_and_symlinks_and_unlisted_entries_are_rejected(self):
        name, digest = self.bundle({'../outside': b'bad'})
        with self.assertRaises(ValueError):
            archive_members(self.root, self.archive, name, digest)
        name, digest = self.bundle({'escape/file': b'bad'})
        (self.root / 'escape').symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            archive_members(self.root, self.archive, name, digest)
        name, digest = self.bundle({'good': b'good'}, extra=[('unlisted', b'bad')])
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            archive_members(self.root, self.archive, name, digest)

    def test_completed_status_does_not_release_a_live_learner_or_followup(self):
        plan = dict(updates=4000, jobs=[dict(host=1, run_id='trial')]); analysis = dict(followup='runs/followup')
        for name in ['runs/trial', 'runs/followup/trial']:
            path = self.root / name; path.mkdir(parents=True)
            (path / 'status.json').write_text(json.dumps(dict(state='complete', step=4000)))
        with patch('finish_study.running_train', return_value=123), patch('finish_study.dependency_complete', return_value=True):
            self.assertFalse(inspect(self.root, analysis, plan, 1)['ready'])
        with patch('finish_study.running_train', return_value=None), patch('finish_study.dependency_complete', return_value=False):
            self.assertFalse(inspect(self.root, analysis, plan, 1)['ready'])
        with patch('finish_study.running_train', return_value=None), patch('finish_study.dependency_complete', return_value=True):
            self.assertTrue(inspect(self.root, analysis, plan, 1)['ready'])
        (self.root / 'runs/trial/status.json').write_text(json.dumps(dict(state='failed')))
        with self.assertRaisesRegex(RuntimeError, 'registered endpoint failed'):
            inspect(self.root, analysis, plan, 1)

    def test_queued_wave_waits_without_inventing_a_completed_endpoint(self):
        job = dict(host=1, run_id='queued', wait_for=['followup/earlier'])
        plan = dict(updates=4000, jobs=[job]); analysis = dict(followup='runs/followup')
        with patch('finish_study.running_train', return_value=None):
            observed = inspect(self.root, analysis, plan, 1)
            self.assertFalse(observed['ready'])
            self.assertIsNone(observed['records'][0]['trainer_state'])
            job.pop('wait_for')
            with self.assertRaisesRegex(RuntimeError, 'unqueued registered trainer is missing'):
                inspect(self.root, analysis, plan, 1)
        with patch('finish_study.running_train', return_value=123):
            self.assertFalse(inspect(self.root, analysis, plan, 1)['ready'])


if __name__ == '__main__': unittest.main()
