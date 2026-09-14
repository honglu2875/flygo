"""An exiting or failed evaluator must not release its CPU lane to a successor."""
import fcntl
import json
from pathlib import Path
import tempfile
import unittest

from scripts.finish_study import dependency_complete


class StudyDependencies(unittest.TestCase):
    def test_completion_waits_for_worker_lock_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            self.assertFalse(dependency_complete(directory))
            (directory/'status.json').write_text(json.dumps({'state':'complete'}))
            with (directory/'worker.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                self.assertFalse(dependency_complete(directory))
            self.assertTrue(dependency_complete(directory))

    def test_failed_dependency_stops_the_successor(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            for state in ('failed','stopped'):
                (directory/'status.json').write_text(json.dumps({'state':state}))
                with self.assertRaises(RuntimeError):dependency_complete(directory)


if __name__=='__main__':unittest.main()
