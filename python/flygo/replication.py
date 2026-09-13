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


def replicate_file(path:Path,root:Path,peer:str,*,timeout=180):
    relative=path.resolve().relative_to(root.resolve())
    digest=sha256(path);size=path.stat().st_size
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
if target.exists():
 assert hash_file(target)==DIGEST,'Immutable peer file has different contents'
else:
 with StorageBudget(root).reserve(files=SIZE,heap=4*1024**2,purpose='verified replica '+str(RELATIVE)):
  target.parent.mkdir(parents=True,exist_ok=True)
  temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.incoming')
  try:
   h=hashlib.sha256();remaining=SIZE
   with temporary.open('wb') as f:
    while remaining:
     c=sys.stdin.buffer.read(min(1024*1024,remaining))
     assert c,'Incomplete replica transfer'
     f.write(c);h.update(c);remaining-=len(c)
   assert h.hexdigest()==DIGEST,'Replica transfer hash mismatch'
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
    with path.open('rb') as stream:
        result=subprocess.run(command,stdin=stream,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError('Peer replication failed: '+result.stderr[-2000:])
    return json.loads(result.stdout)


def replicate_bundle(paths, root: Path, peer: str, *, timeout=300):
    """Copy an immutable file set with admission for both archive and extraction."""
    paths = [Path(path) for path in paths]
    for path in paths:
        path.resolve().relative_to(root.resolve())
        if not path.is_file():
            raise ValueError('Bundle entries must be individual regular files')
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
            return {**json.loads(result.stdout), 'archive_sha256': transfer['sha256']}
        finally:
            archive.unlink(missing_ok=True)
