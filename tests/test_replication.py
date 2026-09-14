"""A checkpoint payload alone is insufficient: interrupted receipt copies must retry."""
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.replicate_checkpoints import push_local
from flygo.replication import replicate_bundle
from flygo.qualify import sha256
from flygo.storage import StorageBudget,Limits

RUN=subprocess.run


class BundleReplicas(unittest.TestCase):
    """Exercise the real remote inventory/extraction code on two local roots."""
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)/'local';self.peer=Path(self.temporary.name)/'peer'
        self.root.mkdir();(self.peer/'environments/fixture/site-packages').mkdir(parents=True)
        self.transferred=[]

    def remote(self,command,**kwargs):
        self.assertEqual(command[0],'ssh')
        executable,flag,code=shlex.split(command[-1]);self.assertEqual(flag,'-c')
        code=code.replace(str(self.root),str(self.peer))
        code=code.replace('from flygo.storage import StorageBudget',
            'from flygo.storage import StorageBudget as Budget,Limits\n'
            'StorageBudget=lambda root: Budget(root,Limits(files_cap=1<<30,free_files_floor=0,available_memory_floor=0))')
        return RUN([sys.executable,'-c',code],**kwargs)

    def transfer(self,path,root,peer,**kwargs):
        target=self.peer/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target);self.transferred.append(path)
        return dict(status='passed',sha256=sha256(path),bytes=path.stat().st_size)

    def replicate(self,paths):
        budget=lambda root:StorageBudget(root,Limits(files_cap=1<<30,free_files_floor=0,available_memory_floor=0))
        with patch('flygo.replication.subprocess.run',side_effect=self.remote), \
             patch('flygo.replication.replicate_file',side_effect=self.transfer), \
             patch('flygo.replication.StorageBudget',side_effect=budget):
            return replicate_bundle(paths,self.root,'fixture')

    def test_missing_only_and_idempotent(self):
        a=self.root/'existing';b=self.root/'nested/file with spaces $(literal)'
        a.write_bytes(b'existing');b.parent.mkdir();b.write_bytes(b'new')
        (self.peer/'existing').write_bytes(a.read_bytes())
        result=self.replicate([a,b])
        self.assertEqual((result['files'],result['transferred_files'],result['verified_existing_files']),(2,1,1))
        self.assertEqual((self.peer/b.relative_to(self.root)).read_bytes(),b.read_bytes())
        self.assertEqual(self.replicate([a,b])['transferred_files'],0)
        self.assertEqual(len(self.transferred),1)
        self.assertEqual(list((self.root/'tmp').iterdir()),[])
        self.assertEqual(list((self.peer/'tmp').iterdir()),[])

    def test_conflicting_existing_content_is_not_overwritten(self):
        p=self.root/'immutable';p.write_bytes(b'abc');(self.peer/p.name).write_bytes(b'xyz')
        with self.assertRaisesRegex(RuntimeError,'hash conflict'):self.replicate([p])
        self.assertEqual((self.peer/p.name).read_bytes(),b'xyz')
        self.assertFalse(self.transferred)

    def test_escaping_peer_symlink_is_rejected(self):
        p=self.root/'immutable';p.write_bytes(b'abc')
        outside=Path(self.temporary.name)/'outside';outside.write_bytes(b'abc')
        (self.peer/p.name).symlink_to(outside)
        with self.assertRaises(RuntimeError):self.replicate([p])
        self.assertFalse(self.transferred)

    def test_empty_and_duplicate_bundles(self):
        self.assertEqual(self.replicate([])['transferred_files'],0)
        p=self.root/'immutable';p.write_bytes(b'abc')
        with self.assertRaises(ValueError):self.replicate([p,p])


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
