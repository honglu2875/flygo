"""A checkpoint payload alone is insufficient: interrupted receipt copies must retry."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.replicate_checkpoints import push_local


class CheckpointReplicas(unittest.TestCase):
    def test_interrupted_receipt_is_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            path=root/'runs/trial/checkpoints/step-00000001.npz'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'owned checkpoint fixture')
            receipt_path=path.with_suffix('.json')
            receipt_path.write_text(json.dumps(dict(path=str(path),replica_status='local_only')))
            copied={}
            failed=False
            def transfer(source,*_):
                nonlocal failed
                if source.suffix=='.json' and not failed:
                    failed=True
                    raise RuntimeError('Connection lost before receipt publication')
                copied[source.name]=source.read_bytes()
            with patch('scripts.replicate_checkpoints.replicate_file',side_effect=transfer):
                with self.assertRaises(RuntimeError):push_local(root,'trial')
                self.assertEqual(json.loads(receipt_path.read_text())['replica_status'],'retry')
                self.assertNotIn(receipt_path.name,copied)
                self.assertEqual(push_local(root,'trial'),[str(path)])
                self.assertEqual(copied[path.name],path.read_bytes())
                self.assertEqual(copied[receipt_path.name],receipt_path.read_bytes())
                self.assertEqual(json.loads(receipt_path.read_text())['replica_status'],'verified')
                self.assertEqual(push_local(root,'trial'),[])


if __name__=='__main__':unittest.main()
