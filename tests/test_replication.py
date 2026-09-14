"""A checkpoint payload alone is insufficient: interrupted receipt copies must retry."""
import json
import hashlib
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.replicate_checkpoints import push_local,relay
from flygo.replication import replicate_bundle,replicate_stream
from flygo.qualify import sha256
from flygo.storage import StorageBudget,Limits

RUN=subprocess.run
POPEN=subprocess.Popen


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

    def send(self,payload,*,size=None,digest=None,path=Path('streamed checkpoint')):
        with tempfile.TemporaryFile() as stream:
            stream.write(payload);stream.seek(0)
            with patch('flygo.replication.subprocess.run',side_effect=self.remote):
                result=replicate_stream(stream,self.root,path,'fixture',size=len(payload) if size is None else size,
                    digest=digest or hashlib.sha256(payload).hexdigest())
            self.assertEqual(stream.tell(),len(payload))
            return result

    def test_stream_consumes_existing_replica_and_rejects_incomplete_or_extra_data(self):
        payload=b'fixed neural weights'*(1<<16)
        self.assertEqual(self.send(payload)['bytes'],len(payload))
        self.assertEqual(self.send(payload)['status'],'passed')
        self.assertEqual((self.peer/'streamed checkpoint').read_bytes(),payload)
        for size,message in [(len(payload)+1,'Incomplete replica transfer'),(len(payload)-1,'exceeds declared size')]:
            with self.assertRaisesRegex(RuntimeError,message):self.send(payload,size=size,path=Path('incomplete'))
            self.assertFalse((self.peer/'incomplete').exists())
        with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
            self.send(payload,digest='0'*64,path=Path('wrong hash'))
        self.assertFalse((self.peer/'wrong hash').exists())
        self.assertFalse(list(self.peer.glob('*.incoming')))

    def test_stream_does_not_overwrite_a_conflict_or_escape_the_root(self):
        (self.peer/'streamed checkpoint').write_bytes(b'existing')
        with self.assertRaisesRegex(RuntimeError,'different contents'):self.send(b'new')
        self.assertEqual((self.peer/'streamed checkpoint').read_bytes(),b'existing')
        with self.assertRaises(ValueError):self.send(b'new',path=Path('../escaped'))


class CheckpointReplicas(unittest.TestCase):
    def test_worker_relay_recovers_receipt_failure_without_a_root_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'root';source=Path(temporary)/'source';peer=Path(temporary)/'peer'
            root.mkdir();relative=Path('runs/trial/checkpoints/step-00000042.npz')
            payload=b'checkpoint arrays and sampler state'*(1<<15)
            checkpoint=source/relative;checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(payload)
            original=dict(path=str(root/relative),step=42,bytes=len(payload),sha256=sha256(checkpoint),replica_status='local_only')
            checkpoint.with_suffix('.json').write_text(json.dumps(original))
            (source/'runs/trial/latest.json').write_text(json.dumps(original))
            (peer/'environments/fixture/site-packages').mkdir(parents=True)
            interrupted=False
            def ssh(command,*args,**kwargs):
                nonlocal interrupted
                self.assertEqual(command[0],'ssh')
                destination=source if command[-2].endswith('-1') else peer
                executable,flag,code=shlex.split(command[-1]);self.assertEqual(flag,'-c')
                code=code.replace(str(root),str(destination))
                code=code.replace('from flygo.storage import StorageBudget',
                    'from flygo.storage import StorageBudget as Budget,Limits\n'
                    'StorageBudget=lambda root: Budget(root,Limits(files_cap=1<<30,free_files_floor=0,available_memory_floor=0))')
                if destination==source and "latest=p.parent.parent/'latest.json'" in code and not interrupted:
                    code='raise RuntimeError("interrupted source receipt")';interrupted=True
                return POPEN([sys.executable,'-c',code],*args,**kwargs)
            budget=StorageBudget(root,Limits(files_cap=1<<30,free_files_floor=0,available_memory_floor=0))
            with patch('subprocess.Popen',side_effect=ssh),patch('scripts.replicate_checkpoints.StorageBudget',return_value=budget):
                with self.assertRaises(subprocess.CalledProcessError):relay(root,1,2,'trial')
                self.assertEqual((peer/relative).read_bytes(),payload)
                self.assertEqual(json.loads(checkpoint.with_suffix('.json').read_text())['replica_status'],'local_only')
                self.assertEqual(relay(root,1,2,'trial'),[str(root/relative)])
                self.assertEqual(relay(root,1,2,'trial'),[])
            self.assertFalse((root/relative).exists())
            self.assertEqual(json.loads(checkpoint.with_suffix('.json').read_text())['replica_status'],'verified')
            self.assertEqual(json.loads((source/'runs/trial/latest.json').read_text())['replica_status'],'verified')
            self.assertEqual(json.loads((peer/relative.with_suffix('.json')).read_text())['sha256'],original['sha256'])

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
