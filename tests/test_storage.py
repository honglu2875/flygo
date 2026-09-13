import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from flygo.storage import Limits, StorageBudget, StoragePressure, allocated_bytes


class StorageTests(unittest.TestCase):
    def test_concurrent_processes_share_cap_and_release_after_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = StorageBudget(root, Limits(1 << 20, 0, 0))
            code = '''
from pathlib import Path
import sys
from flygo.storage import Limits, StorageBudget, StoragePressure
budget = StorageBudget(Path(sys.argv[1]), Limits(1 << 20, 0, 0))
try:
    with budget.reserve(files=800000, heap=0, purpose="child"):
        print("admitted")
except StoragePressure:
    print("blocked")
'''
            with self.assertRaisesRegex(RuntimeError, "worker failure"):
                with budget.reserve(files=800000, heap=0, purpose="parent"):
                    output = subprocess.check_output([sys.executable, "-c", code, temp], text=True, timeout=10)
                    self.assertEqual(output.strip(), "blocked")
                    raise RuntimeError("worker failure")
            output = subprocess.check_output([sys.executable, "-c", code, temp], text=True, timeout=10)
            self.assertEqual(output.strip(), "admitted")
            self.assertEqual(json.loads((root / "control/storage.json").read_text())["reservations"], {})

    def test_live_pressure_and_dead_process_reservations_do_not_delete_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = StorageBudget(root, Limits(1 << 20, 100, 200))
            keep = root / "corpus.bin"
            keep.write_bytes(b"unique corpus")
            with patch.object(budget, "_snapshot", return_value=dict(files_used=100, files_free=150,
                                                                    memory_available=1000)):
                with self.assertRaisesRegex(StoragePressure, "filesystem"):
                    with budget.reserve(files=100, heap=0, purpose="too much"):
                        self.fail("unexpected admission")
            with patch.object(budget, "_snapshot", return_value=dict(files_used=100, files_free=1000,
                                                                    memory_available=250)):
                with self.assertRaisesRegex(StoragePressure, "RAM"):
                    with budget.reserve(files=10, heap=100, purpose="too much"):
                        self.fail("unexpected admission")
            code = '''
from pathlib import Path
import os, sys
from flygo.storage import Limits, StorageBudget
with StorageBudget(Path(sys.argv[1]), Limits(1 << 20, 100, 200)).reserve(files=900000, heap=0, purpose="crash"):
    os._exit(0)
'''
            subprocess.run([sys.executable, "-c", code, temp], check=True, timeout=10)
            self.assertEqual(budget.check()["reserved_files"], 0)
            self.assertEqual(keep.read_bytes(), b"unique corpus")

    def test_hardlinks_count_once_and_shared_policy_cannot_change_silently(self):
        import os
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a").write_bytes(b"x" * 8192)
            one = allocated_bytes(root)
            os.link(root / "a", root / "b")
            self.assertEqual(allocated_bytes(root), one)
            StorageBudget(root, Limits(1 << 20, 0, 0)).check()
            with self.assertRaisesRegex(ValueError, "identical limits"):
                StorageBudget(root, Limits(2 << 20, 0, 0)).check()
