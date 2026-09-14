"""Mine causal near-branch pairs from a frozen release, with family-held-out probes."""
import hashlib
import numpy as np


VERSION = 'bounded-prefix-branches-v2'


def common_prefix(a, b):
    length = min(len(a),len(b))
    different = np.flatnonzero(np.asarray(a[:length]) != np.asarray(b[:length]))
    return int(different[0]) if len(different) else length


def probe_families(families):
    """Reserve an exact quarter of eligible families before training; freeze this set."""
    names=sorted(set(families),key=lambda f:hashlib.sha256(('embedding-probes-v1:'+f).encode()).digest())
    return set(names[:len(names)//4])


def mine_pairs(games, *, min_prefix=20, max_divergence=6, min_gap=.2, seed=918420):
    """Compare up to 256 game pairs per exact shared-prefix bucket, before labels.

    games contain actions, raw_value, opening_family, split and teacher_sha256.
    Compare pre-action states at the same ply; both targets already refer to the
    same current player. Ordering the records by value does not reorder history.
    """
    if min_prefix < 8 or not 1 <= max_divergence <= 6 or not np.isfinite(min_gap) or not 0 < min_gap <= 2:
        raise ValueError('Require a full split prefix, 1–6-ply divergence and a positive teacher gap')
    for g in games:
        if g['split'] != 'train': raise ValueError('Only original training games enter this pilot')
        q = np.asarray(g['raw_value'])
        if len(q) != len(g['actions']) or not np.isfinite(q).all() or np.any(np.abs(q)>1.00001):
            raise ValueError('Invalid pre-action teacher targets')
    buckets = {}
    for i,g in enumerate(games):
        if len(g['actions'])>min_prefix:
            buckets.setdefault(tuple(g['actions'][:min_prefix]),[]).append(i)
    rng=np.random.default_rng(seed);candidates=[]
    for prefix in sorted(buckets):
        indices=buckets[prefix]
        left,right=np.triu_indices(len(indices),1)
        chosen=rng.choice(len(left),min(256,len(left)),replace=False)
        candidates.extend((indices[left[i]],indices[right[i]]) for i in chosen)
    pairs = []; eligible_branches = set()
    for a,b in candidates:
        ga,gb = games[a],games[b]
        if ga['teacher_sha256'] != gb['teacher_sha256']: continue
        if ga['opening_family'] != gb['opening_family']: continue
        prefix = common_prefix(ga['actions'],gb['actions'])
        if prefix < min_prefix: continue
        if prefix >= min(len(ga['actions']),len(gb['actions'])): continue
        eligible_branches.add(tuple(ga['actions'][:prefix]))
        for distance in range(1,max_divergence+1):
            ply = prefix+distance
            if ply >= min(len(ga['actions']),len(gb['actions'])): break
            qa,qb = float(ga['raw_value'][ply]),float(gb['raw_value'][ply])
            gap = abs(qa-qb)
            if gap < min_gap: continue
            good,bad = (a,b) if qa>qb else (b,a)
            pairs.append(dict(good=good,bad=bad,ply=ply,shared_prefix=prefix,
                divergence=distance,quality_gap=gap,to_play=1+ply%2,
                opening_family=ga['opening_family']))
    return pairs, dict(games=len(games),candidate_branches=len(eligible_branches),
        game_comparisons=len(candidates),pairs=len(pairs),min_prefix=min_prefix,
        max_divergence=max_divergence,min_gap=min_gap,seed=seed,max_game_pairs_per_prefix=256)


def family_sample(pairs, count, rng):
    """Uniform opening family, then uniform pair within it; avoids large-family bias."""
    by_family = {}
    for i,p in enumerate(pairs): by_family.setdefault(p['opening_family'],[]).append(i)
    if not by_family or count < 1: raise ValueError('Need pairs and a positive sample count')
    families = sorted(by_family)
    return np.asarray([rng.choice(by_family[families[int(rng.integers(len(families)))]])
                       for _ in range(count)],np.int64)


def load_bank(path):
    import json
    from ..qualify import sha256
    from ..vision import SphericalRenderer
    report=json.loads((path/'result.json').read_text())
    if report['status']!='complete':raise ValueError('Pair bank is not complete')
    for name,digest in report['files'].items():
        if sha256(path/name)!=digest:raise ValueError('Embedding bank hash mismatch: '+name)
    with np.load(path/'pairs.npz',allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in data.files}
    with np.load(path/'renderer.npz',allow_pickle=False) as data:
        geometry={k:data[k].copy() for k in data.files}
    renderer=SphericalRenderer(**{k:geometry[k] for k in ('index','weight','unit','side','patch')})
    ports={k:geometry[k] for k in ('input_index','output_group','output_scale')}
    pairs=json.loads((path/'pairs.json').read_text())
    if len(pairs)!=len(arrays['features']):raise ValueError('Pair metadata length mismatch')
    return report,arrays,pairs,renderer,ports,geometry


def quartet_views(renderer, features, rng, *, augment=True):
    """One shared D4 per branch pair; positive views only vary global contrast."""
    x=np.asarray(features,np.float32).copy()
    if x.ndim!=5 or x.shape[1:]!=(2,9,9,12):raise ValueError('Expected pairs of four-board observations')
    if augment:
        for i in range(len(x)):
            transform=int(rng.integers(8))
            view=np.flip(x[i],axis=2) if transform//4 else x[i]
            x[i]=np.rot90(view,transform%4,axes=(1,2))
    x=np.repeat(x,2,axis=1).reshape(-1,9,9,12)
    return renderer.render(x,contrast=rng.uniform(.9,1.1,len(x)).astype(np.float32))
