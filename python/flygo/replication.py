"""Copy one owned immutable artifact through a remote, host-wide RAM reservation."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import uuid

from .qualify import sha256
from .storage import StorageBudget


def replicate_stream(stream,root:Path,relative:Path,peer:str,*,digest:str,size:int,timeout=180):
    """Receive an exact immutable payload from a file or pipe, with bounded memory.

    Existing replicas still consume and verify the stream so a producer cannot
    block on a full pipe. The caller must also check its producer's exit status.
    """
    relative=Path(relative)
    if (relative.is_absolute() or not relative.parts or '..' in relative.parts
            or type(size) is not int or size<0 or not isinstance(digest,str)
            or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('Expected a relative artifact path, nonnegative size and SHA-256')
    code='''import pathlib,sys,json,hashlib,uuid,os
root=pathlib.Path(ROOT)
sites=sorted((root/'environments').glob('*/site-packages'))
assert sites,'No qualified FlyGo runtime on peer'
sys.path.insert(0,str(sites[0]))
from flygo.storage import StorageBudget
target=root/RELATIVE
target.resolve().relative_to(root.resolve())
def hash_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def receive(output=None):
 h=hashlib.sha256();remaining=SIZE
 while remaining:
  c=sys.stdin.buffer.read(min(1024*1024,remaining))
  assert c,'Incomplete replica transfer'
  if output is not None:output.write(c)
  h.update(c);remaining-=len(c)
 assert not sys.stdin.buffer.read(1),'Replica transfer exceeds declared size'
 assert h.hexdigest()==DIGEST,'Replica transfer hash mismatch'
if target.exists():
 assert target.is_file() and target.stat().st_size==SIZE and hash_file(target)==DIGEST,'Immutable peer file has different contents'
 receive()
else:
 block=os.statvfs(root).f_frsize
 with StorageBudget(root).reserve(files=((SIZE+block-1)//block)*block,heap=4*1024**2,purpose='verified replica '+str(RELATIVE)):
  target.parent.mkdir(parents=True,exist_ok=True)
  temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.incoming')
  try:
   with temporary.open('wb') as f:
    receive(f)
   try:os.link(temporary,target)
   except FileExistsError:assert hash_file(target)==DIGEST,'Conflicting immutable replica'
  finally:
   if temporary.exists():temporary.unlink()
print(json.dumps(dict(status='passed',sha256=DIGEST,bytes=SIZE)))
'''
    for key,value in dict(ROOT=str(root),RELATIVE=str(relative),DIGEST=digest,SIZE=size).items():
        code=code.replace(key,repr(value))
    command=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10',peer,
             'python3 -c '+shlex.quote(code)]
    result=subprocess.run(command,stdin=stream,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError('Peer replication failed: '+result.stderr[-2000:])
    record=json.loads(result.stdout)
    if record!=dict(status='passed',sha256=digest,bytes=size):
        raise ValueError('Peer verification record differs from the payload')
    return record


def replicate_file(path:Path,root:Path,peer:str,*,timeout=180):
    relative=path.resolve().relative_to(root.resolve())
    digest=sha256(path);size=path.stat().st_size
    with path.open('rb') as stream:
        return replicate_stream(stream,root,relative,peer,digest=digest,size=size,timeout=timeout)


def replicate_bundle(paths, root: Path, peer: str, *, timeout=300):
    """Verify existing peer files, then archive only missing immutable entries."""
    root = root.resolve()
    paths = [Path(path).resolve() for path in paths]
    for path in paths:
        path.resolve().relative_to(root.resolve())
        if not path.is_file():
            raise ValueError('Bundle entries must be individual regular files')
    if len(set(paths)) != len(paths):
        raise ValueError('Bundle entries must be unique')
    manifest = [dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha256(path))
                for path in paths]
    inventory = '''import pathlib,sys,json,hashlib
root=pathlib.Path(ROOT).resolve()
manifest=json.load(sys.stdin);missing=[]
for index,record in enumerate(manifest):
 target=root/record['path']
 target.resolve().relative_to(root)
 if not target.exists():
  missing.append(index);continue
 assert target.is_file(),'Bundle target is not a regular file'
 assert target.stat().st_size==record['bytes'],'Immutable bundle size conflict: '+str(target)
 h=hashlib.sha256()
 with target.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==record['sha256'],'Immutable bundle hash conflict: '+str(target)
print(json.dumps(dict(status='passed',missing=missing,files=len(manifest))))
'''.replace('ROOT', repr(str(root)))
    checked = subprocess.run(['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10',
                              peer,'python3 -c '+shlex.quote(inventory)], input=json.dumps(manifest),
                             capture_output=True,text=True,timeout=timeout)
    if checked.returncode:
        raise RuntimeError('Bundle inventory failed: '+checked.stderr[-2000:])
    checked = json.loads(checked.stdout)
    missing = checked['missing']
    if (checked.get('status') != 'passed' or checked.get('files') != len(paths)
            or len(set(missing)) != len(missing)
            or any(type(index) is not int or not 0 <= index < len(paths) for index in missing)):
        raise ValueError('Invalid peer bundle inventory')
    total = len(paths)
    if not missing:
        return dict(status='passed',files=total,transferred_files=0,verified_existing_files=total)
    paths = [paths[index] for index in missing]
    size = sum(path.stat().st_size for path in paths) + len(paths) * 4096 + 10240
    archive = root / 'tmp' / ('replica-' + uuid.uuid4().hex + '.tar')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with StorageBudget(root).reserve(files=size, heap=16 * 1024**2, purpose='immutable replica bundle'):
        try:
            with tarfile.open(archive, 'w') as bundle:
                for path in paths:
                    bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)
            transfer = replicate_file(archive, root, peer, timeout=timeout)
            code = '''import pathlib,sys,tarfile,hashlib,uuid,os,shutil,json
root=pathlib.Path(ROOT)
sys.path.insert(0,str(sorted((root/'environments').glob('*/site-packages'))[0]))
from flygo.storage import StorageBudget
archive=pathlib.Path(ARCHIVE)
def digest(stream):
 h=hashlib.sha256()
 for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 return h.digest()
with tarfile.open(archive) as bundle:
 members=bundle.getmembers()
 assert all(m.isfile() for m in members),'Only regular files are supported'
 with StorageBudget(root).reserve(files=sum(m.size for m in members),heap=16*1024**2,purpose='immutable bundle extraction'):
  for member in members:
   target=root/member.name
   target.resolve().relative_to(root.resolve())
   source=bundle.extractfile(member)
   if target.exists():
    with target.open('rb') as stream:assert digest(stream)==digest(source),'Immutable bundle conflict: '+str(target)
    continue
   target.parent.mkdir(parents=True,exist_ok=True)
   temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.incoming')
   try:
    with temporary.open('wb') as stream:shutil.copyfileobj(source,stream)
    try:os.link(temporary,target)
    except FileExistsError:
     with target.open('rb') as a,temporary.open('rb') as b:assert digest(a)==digest(b),'Immutable publication conflict'
   finally:temporary.unlink(missing_ok=True)
archive.unlink()
print(json.dumps(dict(status='passed',files=len(members))))
'''.replace('ROOT', repr(str(root))).replace('ARCHIVE', repr(str(archive)))
            result = subprocess.run(['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                     peer, 'python3 -c ' + shlex.quote(code)], capture_output=True,
                                    text=True, timeout=timeout)
            if result.returncode:
                raise RuntimeError('Bundle extraction failed: ' + result.stderr[-2000:])
            extracted=json.loads(result.stdout)
            if extracted.get('status')!='passed' or extracted.get('files')!=len(paths):
                raise ValueError('Bundle extraction result differs from the manifest')
            return {**extracted, 'files':total, 'transferred_files':len(paths),
                    'verified_existing_files':total-len(paths), 'archive_sha256':transfer['sha256']}
        finally:
            archive.unlink(missing_ok=True)
