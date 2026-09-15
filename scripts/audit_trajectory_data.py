#!/usr/bin/env python3
"""Check causal replay and color-frame conversion on a fixed training-only sample."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from flygo import _native
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release
from flygo.go import replay_observations
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB, StorageBudget


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); root = args.root; pin([56, 57, 58, 59])
    args.output.resolve().relative_to(root.resolve())
    if args.output.exists(): raise ValueError('Trajectory audit records are immutable')
    with StorageBudget(root).reserve(files=4 << 20, heap=2 * GIB, purpose='training trajectory and color-frame audit'):
        manifest, arrays, indexes = load_release(root, root / 'releases/v0-1m/manifest.json', cache=True)
        families = defaultdict(list); offset = 0
        for game, record in enumerate(manifest['records']):
            if record['split'] == 'train': families[record['opening_family']].append((game, offset, record))
            offset += record['rows']
        rng = np.random.default_rng(918457); names = sorted(families)
        if len(names) < 128: raise ValueError('Expected 128 distinct training opening families')
        selected = []
        for i in rng.choice(len(names), 128, replace=False):
            choices = families[names[i]]
            selected.append(choices[int(rng.integers(len(choices)))])
        rows = []; pass_pairs = 0; occupied_pass_pairs = 0
        for game, offset, record in selected:
            path = root / record['path']
            if sha256(path) != record['sha256']: raise ValueError('Frozen training replay changed')
            with np.load(path, allow_pickle=False) as source:
                metadata = json.loads(source['metadata'].tobytes())
                actions = source['actions']; stored_stones = source['stones']; stored_legal = source['legal']
            if any(metadata[k] != record[k] for k in ('game_id', 'rows', 'opening_family', 'split')):
                raise ValueError('Training replay metadata differs')
            length = len(actions); indices = np.arange(offset, offset + length, dtype=np.int64)
            if length != record['rows'] or not np.isin(indices, indexes['train']).all():
                raise ValueError('Trajectory escaped its training game')
            np.testing.assert_array_equal(arrays['ply'][indices], np.arange(length))
            np.testing.assert_array_equal(arrays['game_index'][indices], np.full(length, game))
            observed = replay_observations(actions, [0, length])
            np.testing.assert_array_equal(observed.stones, stored_stones.reshape(length, 9, 9))
            np.testing.assert_array_equal(observed.legal, stored_legal)
            np.testing.assert_array_equal(observed.legal, arrays['legal'][indices])
            if observed.outcomes[0]['terminal'] != record['terminal']:
                raise ValueError('Replay termination differs')
            x = arrays['features'][indices]
            black = np.arange(length) % 2 == 0
            np.testing.assert_array_equal(x[..., 8], np.broadcast_to(black[:, None, None], (length, 9, 9)))
            absolute_black = np.where(black[:, None, None], x[..., 0], x[..., 1])
            absolute_white = np.where(black[:, None, None], x[..., 1], x[..., 0])
            np.testing.assert_array_equal(absolute_black, observed.stones == 1)
            np.testing.assert_array_equal(absolute_white, observed.stones == 2)
            if observed.stones[0].any(): raise ValueError('Episode does not begin on the declared empty board')
            passes = np.flatnonzero(actions[:-1] == 81)
            if len(passes):
                np.testing.assert_array_equal(observed.stones[passes], observed.stones[passes + 1])
                relative = x[..., 0] - x[..., 1]
                np.testing.assert_array_equal(relative[passes], -relative[passes + 1])
                absolute = absolute_black - absolute_white
                np.testing.assert_array_equal(absolute[passes], absolute[passes + 1])
                occupied_pass_pairs += int(np.any(observed.stones[passes] != 0, axis=(1, 2)).sum())
            pass_pairs += len(passes)
            rows.append(dict(game_index=game, game_id=record['game_id'], opening_family=record['opening_family'],
                             offset=offset, rows=length, terminal=record['terminal'], sha256=record['sha256'],
                             nonterminal_pass_transitions=len(passes)))
        lengths = [r['rows'] for _, _, r in sum(families.values(), [])]
        result = dict(status='passed', created=time.time(), dataset_id=manifest['dataset_id'], seed=918457,
            selection_rule='One uniformly sampled game per sampled training opening family; no target-based selection.',
            checked_games=len(rows), checked_positions=sum(r['rows'] for r in rows), checked_families=len(rows),
            records=rows, nonterminal_pass_transitions=pass_pairs, occupied_pass_transitions=occupied_pass_pairs,
            training_games=len(lengths), training_positions=sum(lengths),
            training_game_length_quantiles=np.quantile(lengths, [0, .1, .5, .9, 1]).tolist(),
            source_sha256=sha256(Path(__file__)), native_sha256=sha256(Path(_native.__file__)),
            scope='Training-only replay/schema audit. Stored and cached pre-action observations, legality, game offsets and ply order match full-prefix replay on this fixed sample. Absolute black/white planes derive exactly from current own/opponent planes and turn. Passes preserve absolute imagery while occupied relative imagery flips. No teacher policy/value array is indexed, no final-test example is selected, and no neural inference/update, new dataset, scientific feature change or TPU use occurs. This does not qualify a sequence sampler or demonstrate a memory benefit.')
        atomic_json(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'scope')}), flush=True)


if __name__ == '__main__': main()
