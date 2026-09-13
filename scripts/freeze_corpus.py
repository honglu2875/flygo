#!/usr/bin/env python3
"""Freeze audited games as a balanced release or a clone of current production."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import uuid

import numpy as np

from flygo.data.corpus import atomic_json, audit_game, identity
from flygo.qualify import sha256
from flygo.replication import replicate_bundle, replicate_file
from flygo.runtime import cpu_profile, pin
from flygo.storage import GIB, StorageBudget

SSH = ['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10']


class NotReady(ValueError):
    """The immutable producer has not filled every requested stratum yet."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    parser.add_argument('--run-id',default='expert-v1')
    parser.add_argument('--release',default='pilot-v1')
    parser.add_argument('--positions',type=int,default=10000)
    parser.add_argument('--selection',choices=('balanced','all-complete'),default='balanced')
    parser.add_argument('--cutoff',type=float,help='For all-complete: include files written before this Unix time; defaults to launch time')
    parser.add_argument('--cpus',help='Comma-separated spare physical CPUs')
    parser.add_argument('--wait',action='store_true',help='Wait for sufficient balanced games, then audit and freeze once')
    args = parser.parse_args()
    if args.selection=='all-complete':
        if args.wait:parser.error('--wait applies to balanced releases')
        args.cutoff=args.cutoff or time.time()
    pin(list(map(int,args.cpus.split(','))) if args.cpus else cpu_profile()['research_cpus'][:4])
    watch=args.root/'runs'/('freeze-'+args.release)
    while True:
        try:
            freeze(args)
            if args.wait:
                atomic_json(watch/'status.json',dict(state='complete',release=args.release,updated=time.time()))
            return
        except NotReady as error:
            if not args.wait:raise
            atomic_json(watch/'status.json',dict(state='waiting_games',reason=str(error),target_positions=args.positions,updated=time.time()))
            # Bounded waits also allow a user to stop the watcher promptly.
            for _ in range(3):
                if (watch/'stop').exists():
                    atomic_json(watch/'status.json',dict(state='stopped',updated=time.time()))
                    return
                time.sleep(20)
        except BaseException as error:
            if args.wait:atomic_json(watch/'status.json',dict(state='failed',error=repr(error),updated=time.time()))
            raise


def freeze(args):
    root=args.root
    contract=json.loads((root/'runs'/args.run_id/'contract.json').read_text())
    contract_id=identity(contract)
    corpus=root/'corpora'/contract_id
    release=root/'releases'/args.release
    if release.exists():
        raise ValueError('Release names are immutable; choose a new name')
    with StorageBudget(root).reserve(files=4*GIB,heap=2*GIB,purpose='gather and freeze expert corpus'):
        for host in range(1,4):
            subprocess.run(['rsync','-a','--ignore-existing','--include=*.npz','--include=*/','--exclude=*',
                '-e',shlex.join(SSH),f'cubic27@t1v-n-a09f5679-w-{host}:{corpus}/host-{host}',str(corpus)+'/'],check=True)
        strata=defaultdict(list)
        for path in sorted(corpus.glob('host-*/worker-*/*.npz')):
            if args.selection=='all-complete' and path.stat().st_mtime>args.cutoff:
                continue
            with np.load(path,allow_pickle=False) as data:
                metadata=json.loads(data['metadata'].tobytes())
            if metadata['contract_id']!=contract_id:
                raise ValueError('Corpus contract mismatch')
            if args.selection=='all-complete' and not metadata['terminal']:
                continue
            strata[(metadata['opponent_index'],metadata['expert_color'])].append((path,metadata))
        if len(strata)!=len(contract['opponents'])*2:
            raise NotReady('Both expert colors must be represented for every opponent')
        # Stable random order, independent of host completion order. Round-robin
        # balances games, while naturally retaining their different lengths.
        for rows in strata.values():
            rows.sort(key=lambda item:identity([args.release,item[1]['game_id']]))
        if args.selection=='all-complete':
            chosen=sorted((item for rows in strata.values() for item in rows),key=lambda item:item[1]['game_id'])
            positions=sum(metadata['rows'] for _,metadata in chosen)
        else:
            chosen=[]; positions=0; index=0
            while positions<args.positions:
                for key in sorted(strata):
                    if index>=len(strata[key]):
                        raise NotReady(f'Wait for more complete games in stratum {key}; currently {positions} selectable positions')
                    item=strata[key][index];chosen.append(item);positions+=item[1]['rows']
                index+=1
        staging=release.with_name('.'+release.name+'.'+uuid.uuid4().hex+'.staging')
        staging.mkdir(parents=True)
        atomic_json(root/'runs'/('freeze-'+args.release)/'status.json',
                    dict(state='auditing',games=len(chosen),positions=positions,updated=time.time()))
        records=[]; splits=Counter();colors=Counter();opponents=Counter();phases=Counter();
        unique_states=set(); saturated=0; truncated=0; bytes_total=0
        input_sets=defaultdict(set)
        for path,metadata in chosen:
            checked=audit_game(path)
            if checked!=metadata:
                raise ValueError('Metadata changed during freeze')
            digest=sha256(path)
            record_path=path.relative_to(root)
            if args.selection=='all-complete':
                relative=Path('games')/path.relative_to(corpus)
                clone=staging/relative
                clone.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,clone)
                if sha256(clone)!=digest:raise ValueError('Snapshot clone hash mismatch')
                record_path=release.relative_to(root)/relative
            records.append(dict(path=str(record_path),sha256=digest,bytes=path.stat().st_size,**metadata))
            bytes_total+=path.stat().st_size
            splits[metadata['split']]+=metadata['rows'];colors[metadata['expert_color']]+=1
            opponents[metadata['opponent_index']]+=1;truncated+=not metadata['terminal']
            with np.load(path,allow_pickle=False) as data:
                saturated+=int((np.abs(data['raw_value'])>0.98).sum())
                for ply,stones in enumerate(data['stones']):
                    key=hashlib.sha256(stones.tobytes()+bytes([ply%2])).hexdigest()
                    unique_states.add(key);input_sets[metadata['split']].add(key)
                    phases['opening' if ply<16 else 'middle' if ply<50 else 'late']+=1
            if len(records)%1000==0:
                atomic_json(root/'runs'/('freeze-'+args.release)/'status.json',
                            dict(state='auditing',games=len(chosen),audited_games=len(records),positions=positions,updated=time.time()))
        manifest=dict(schema_version=1,release=args.release,contract_id=contract_id,contract=contract,
                      created=time.time(),positions=positions,games=len(records),records=records,
                      producer=json.loads((root/'runs'/args.run_id/'cluster.json').read_text()),
                      sampling=('all complete games in production before the recorded file-time cutoff'
                                if args.selection=='all-complete' else 'equal complete-game counts per opponent/expert-color stratum'),
                      copies=['t1v-n-a09f5679-w-0','t1v-n-a09f5679-w-1'],
                      audit=dict(splits=dict(splits),expert_colors=dict(colors),opponents=dict(opponents),
                                 phases=dict(phases),truncated_games=truncated,unique_board_turn_states=len(unique_states),
                                 saturated_values=saturated,bytes=bytes_total,bytes_per_position=bytes_total/positions,
                                 board_turn_overlap_train_validation=len(input_sets['train']&input_sets['validation']),
                                 overlap_note='Board/turn diagnostic only; loader builds exact history/legal input hashes and novel validation'))
        if args.selection=='all-complete':
            manifest['snapshot']=dict(cutoff_unix=args.cutoff,cloned_files=True,
                split_contract='D4-canonical first-eight-action family hash; 90/5/5 train/validation/test, unchanged from production',
                scope='Production-distribution research snapshot; not the balanced V0 acceptance release')
        manifest['dataset_id']=identity(manifest)
        atomic_json(staging/'manifest.json',manifest)
        staging.rename(release)
        peer='cubic27@t1v-n-a09f5679-w-1'
        replicate_bundle([root/record['path'] for record in records]+[release/'manifest.json'],root,peer)
        verifier='''import hashlib,json,pathlib
p=pathlib.Path(ROOT); m=json.loads((p/REL/'manifest.json').read_text())
for r in m['records']:
 h=hashlib.sha256()
 with (p/r['path']).open('rb') as f:
  while True:
   chunk=f.read(1024*1024)
   if not chunk:break
   h.update(chunk)
 assert h.hexdigest()==r['sha256'],r['path']
print(json.dumps(dict(status='passed',games=len(m['records']),dataset_id=m['dataset_id'])))
'''.replace('ROOT',repr(str(root))).replace('REL',repr(str(release.relative_to(root))))
        verification=json.loads(subprocess.check_output(SSH+[peer,'python3 -c '+shlex.quote(verifier)],text=True))
        atomic_json(release/'replication.json',verification)
        replicate_file(release/'replication.json',root,peer)
        atomic_json(root/'runs'/('freeze-'+args.release)/'status.json',
                    dict(state='complete',release=args.release,dataset_id=manifest['dataset_id'],
                         games=len(records),positions=positions,updated=time.time()))
        print(json.dumps({k:v for k,v in manifest.items() if k not in ('records','contract','producer')},indent=2))


if __name__=='__main__':
    main()
