import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from flygo.data.corpus import audit_game, opening_family, publish_game
from flygo.data.katago import query
from flygo.data.label import parse_label
from flygo.go import Game
from flygo.go import GameConfig,_json
from flygo import _native
from flygo.runtime import allocate_cpus


class DataContracts(unittest.TestCase):
    def test_pin_includes_already_created_threads(self):
        code='''import os,threading,pathlib
from flygo.runtime import pin
event=threading.Event()
thread=threading.Thread(target=event.wait);thread.start()
selected={max(os.sched_getaffinity(0))}
try:
 pin(selected)
 assert all(os.sched_getaffinity(int(p.name))<=selected for p in pathlib.Path('/proc/self/task').iterdir())
finally:
 event.set();thread.join()
'''
        subprocess.run([sys.executable,'-B','-c',code],check=True,timeout=10)

    def test_physical_isolation_and_numa(self):
        rows = '\n'.join(f'{cpu},{cpu % 120},{(cpu % 120) // 60},{(cpu % 120) // 60}' for cpu in range(240))
        profile = allocate_cpus(rows, set(range(240)), 64, 8)
        self.assertEqual(len(profile['workers']), 8)
        self.assertEqual(profile['workers'][4], list(range(60, 68)))
        self.assertEqual(len(profile['research_cpus']), 56)
        self.assertFalse(set(profile['research_logical_cpus']) & set(profile['generation_siblings']))
        for worker in profile['workers']:
            self.assertEqual(len({cpu // 60 for cpu in worker}), 1)

    def test_causal_coordinates(self):
        self.assertEqual(query([0, 81, 80], visits=1)['moves'], [['B', 'A9'], ['W', 'pass'], ['B', 'J1']])

    def test_perspective_and_root_edge_counts(self):
        response = dict(policy=[0.5, -1, -1, -1, 0.5], rootInfo=dict(currentPlayer='W', rawWinrate=0.8, winrate=0.6),
                        moveInfos=[dict(move='A2', order=0, edgeVisits=3, visits=100), dict(move='pass', order=1, edgeVisits=1, visits=5)])
        label = parse_label(response, [0, 4], 2, size=2)
        self.assertAlmostEqual(float(label['raw_value']), 0.6)
        self.assertAlmostEqual(float(label['search_value']), 0.2)
        np.testing.assert_array_equal(label['search_policy'], [0.75, 0, 0, 0, 0.25])
        response['rootInfo']['currentPlayer'] = 'B'
        self.assertAlmostEqual(float(parse_label(response, [0, 4], 1, size=2)['raw_value']), -0.6)
        with self.assertRaises(ValueError):
            parse_label(response, [0, 1, 4], 1, size=2)

    def test_publication_replay_and_retry(self):
        game = Game()
        rows = []
        for ply in range(2):
            legal = np.zeros(82, bool)
            legal[game.legal()] = True
            rows.append(dict(stones=game.state().stones.copy(), legal=legal,
                             raw_policy=(legal / legal.sum()).astype(np.float32), raw_value=np.float32(0),
                             teacher_visits=np.int32(16 if ply == 0 else 1), search_policy_valid=False))
            game.play(1 + ply, 81)
        metadata = dict(game_id='test-game', contract_id='test-contract', rows=2, expert_color=1, visits=16,
                        terminal=True, white_score=game.state().white_score)
        with tempfile.TemporaryDirectory() as temporary:
            path = publish_game(Path(temporary), metadata, rows, [81, 81])
            before = path.read_bytes()
            self.assertEqual(audit_game(path)['rows'], 2)
            self.assertEqual(publish_game(Path(temporary), metadata, rows, [81, 81]), path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(len(list(Path(temporary).iterdir())), 1)
            rows[0]['stones'][0, 0] = 1
            with self.assertRaises(ValueError):
                publish_game(Path(temporary), {**metadata, 'game_id': 'corrupt-game'}, rows, [81, 81])

    def test_symmetric_opening_family(self):
        self.assertEqual(opening_family([0, 10, 20, 81]), opening_family([8, 16, 24, 81]))

    def test_native_feature_replay_and_augmentation(self):
        from flygo.data.loader import Sampler
        actions=np.array([40,41,31,81,0],np.int32)
        features=_native.replay_features(_json(GameConfig()),actions,np.array([0,5],np.int64)).reshape(5,9,9,12)
        game=Game()
        for ply,action in enumerate(actions):
            np.testing.assert_array_equal(features[ply],game.features())
            request=game.start()
            np.testing.assert_array_equal(features[ply],request.features)
            request=game.evaluate(request,np.zeros(82,np.float32),0.)
            self.assertIsNone(request.identity);game.finish()
            game.play(1+ply%2,int(action))
        legal=np.concatenate([features[:,:,:,-1].reshape(5,81).astype(bool),np.ones((5,1),bool)],axis=1)
        policy=legal.astype(np.float32);policy/=policy.sum(axis=1,keepdims=True)
        arrays=dict(features=features,legal=legal,raw_policy=policy,raw_value=np.zeros(5,np.float32))
        sampler=Sampler(arrays,dict(train=np.arange(5)))
        saved=sampler.state();batch=sampler.batch(32);sampler.restore(saved)
        for a,b in zip(batch,sampler.batch(32)):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(batch[0][:,:,:,-1].reshape(32,81).astype(bool),batch[1][:,:81])
        self.assertTrue(np.all(batch[2][~batch[1]]==0))

    def test_frozen_cache_preserves_history_splits_and_sampling(self):
        from flygo.data.corpus import identity,split_for_family,atomic_json
        from flygo.data.loader import load_release,Sampler
        from flygo.qualify import sha256
        rng=np.random.default_rng(519)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);records=[];needed={'train','validation','test'}
            while needed:
                actions=rng.choice(81,size=8,replace=False).tolist()+[81,81]
                split=split_for_family(opening_family(actions))
                if split not in needed:continue
                game=Game();rows=[]
                for ply,action in enumerate(actions):
                    legal=np.zeros(82,bool);legal[game.legal()]=True
                    rows.append(dict(stones=game.state().stones.copy(),legal=legal,
                        raw_policy=(legal/legal.sum()).astype(np.float32),raw_value=np.float32(.25),
                        teacher_visits=np.int32(16 if ply%2==0 else 1),search_policy_valid=False))
                    game.play(1+ply%2,action)
                metadata=dict(game_id=identity(actions),contract_id='fixture',expert_color=1,
                              opponent_index=0,visits=16,terminal=True,white_score=game.state().white_score)
                path=publish_game(root/'games',metadata,rows,actions)
                metadata=audit_game(path)
                records.append(dict(path=str(path.relative_to(root)),sha256=sha256(path),**metadata))
                needed.remove(split)
            manifest=dict(positions=sum(r['rows'] for r in records),games=len(records),records=records)
            manifest['dataset_id']=identity(manifest)
            path=root/'releases/fixture/manifest.json';atomic_json(path,manifest)
            atomic_json(path.parent/'replication.json',dict(status='passed',dataset_id=manifest['dataset_id']))
            _,direct,indices=load_release(root,path)
            with patch('flygo.data.cache.StorageBudget'):
                _,cached,cached_indices=load_release(root,path,cache=True)
                _,reopened,reopened_indices=load_release(root,path,cache=True)
            for name in direct:
                np.testing.assert_array_equal(direct[name],cached[name])
                np.testing.assert_array_equal(direct[name],reopened[name])
                self.assertFalse(cached[name].flags.writeable)
            for name in indices:
                np.testing.assert_array_equal(indices[name],reopened_indices[name])
            a=Sampler(direct,indices,seed=7);b=Sampler(cached,cached_indices,seed=7)
            for x,y in zip(a.batch(64),b.batch(64)):np.testing.assert_array_equal(x,y)
            self.assertFalse(set(indices['train'])&set(indices['validation']))
            self.assertFalse(set(indices['train'])&set(indices['test']))
            with self.assertRaisesRegex(ValueError,'bounded loader'):
                load_release(root,path,max_positions=1)
            cache_file=next((root/'cache').rglob('raw_value.npy'))
            with cache_file.open('ab') as stream:stream.write(b'bad')
            with self.assertRaisesRegex(ValueError,'cache hash mismatch'):
                load_release(root,path,cache=True)
            game_path=root/records[0]['path']
            with np.load(game_path,allow_pickle=False) as game:
                columns={key:game[key] for key in game.files}
            metadata=json.loads(columns['metadata'].tobytes())
            metadata['split']='validation' if metadata['split']=='train' else 'train'
            columns['metadata']=np.frombuffer(json.dumps(metadata).encode(),np.uint8)
            with game_path.open('wb') as stream:np.savez_compressed(stream,**columns)
            with self.assertRaisesRegex(ValueError,'family split assignment'):audit_game(game_path)


if __name__ == '__main__':
    unittest.main()
