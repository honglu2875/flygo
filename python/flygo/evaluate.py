"""Batched prior/search matches against fixed KataGo with fresh paired openings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .data.corpus import atomic_json
from .data.katago import AnalysisClient,query,model_path
from .data.label import parse_label
from .go import Game
from .play import load_player,choose_moves,search_config
from .qualify import sha256
from .runtime import cpu_profile,pin
from .storage import StorageBudget,GIB


def matches(model,opponent,*,games=16,seed=41031,max_moves=324,simulations=0,search='puct'):
    active=[];completed=[]
    for index in range(games):
        rng=np.random.default_rng(seed+index//2)
        game=Game(search_config(simulations,search));actions=[]
        for ply in range(4):
            candidates=[int(a) for a in game.legal() if a<81 and 1<=a//9<=7 and 1<=a%9<=7]
            a=int(rng.choice(candidates));game.play(1+ply%2,a);actions.append(a)
        active.append(dict(index=index,game=game,actions=actions,fly_color=1+index%2,fly_neural_evaluations=0))
    while active:
        pending={}
        fly=[]
        for row in active:
            color=row['game'].state().to_play
            if color==row['fly_color']:fly.append(row)
            else:pending[row['index']]=opponent.submit(query(row['actions'],visits=16))
        if fly:
            if simulations:
                for row,result in zip(fly,choose_moves([row['game'] for row in fly],model)):
                    row['next_action']=result.action
                    row['fly_neural_evaluations']+=result.neural_evaluations
            else:
                output=model.infer(np.stack([row['game'].features() for row in fly]))
                for b,row in enumerate(fly):
                    legal=row['game'].legal();row['next_action']=int(legal[np.argmax(output['logits'][b,legal])])
                    row['fly_neural_evaluations']+=1
        remaining=[]
        for row in active:
            game=row['game'];color=game.state().to_play
            if row['index'] in pending:
                label=parse_label(pending[row['index']].result(timeout=180),game.legal(),color)
                action=label['best_action']
            else:action=row.pop('next_action')
            game.play(color,action);row['actions'].append(action)
            state=game.state()
            if state.terminal or len(row['actions'])>=max_moves:
                completed.append({k:v for k,v in row.items() if k!='game'} | dict(terminal=state.terminal,white_score=state.white_score))
            else:remaining.append(row)
        active=remaining
    return completed


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--opponent',type=int,default=0);p.add_argument('--games',type=int,default=16)
    p.add_argument('--seed',type=int,default=41031);p.add_argument('--threads',type=int,default=16)
    p.add_argument('--cpus');p.add_argument('--output',required=True,type=Path)
    p.add_argument('--simulations',type=int,default=0,help='Zero uses the prior; positive values use native search')
    p.add_argument('--search',choices=['puct','gumbel'],default='puct')
    args=p.parse_args(argv)
    if not 2<=args.games<=32 or args.games%2:p.error('Choose an even panel of 2..32 games')
    if args.simulations<0:p.error('Simulations must be nonnegative')
    cpus=list(map(int,args.cpus.split(','))) if args.cpus else cpu_profile()['research_cpus'][-args.threads:]
    pin(cpus)
    with StorageBudget(args.root).reserve(files=32*1024**2,heap=12*GIB,purpose='FlyGo prior strength panel'):
        graph=Path(json.loads((args.root/'runs/m4/graph.json').read_text())['path'])
        model,metadata=load_player(args.checkpoint,graph,threads=args.threads)
        checkpoint_hash=sha256(args.checkpoint)
        contract=json.loads((args.root/'runs/expert-v1/config.json').read_text())['contract']
        opponent_record=contract['opponents'][args.opponent]
        args.output.mkdir(parents=True,exist_ok=False)
        start=time.time()
        with AnalysisClient(args.root/'artifacts'/contract['engine_sha256']/'katago',model_path(args.root,opponent_record),
             args.output/'katago',cpus[:4],concurrency=8,eigen_threads=4,max_pending=32) as opponent:
            records=matches(model,opponent,games=args.games,seed=args.seed,simulations=args.simulations,search=args.search)
        completed=[g for g in records if g['terminal']]
        wins=sum((g['white_score']>0)==(g['fly_color']==2) for g in completed)
        result=dict(status='passed',kind='prior_only_9x9' if not args.simulations else args.search+'_9x9',
                    simulations=args.simulations,fly_neural_evaluations=sum(g['fly_neural_evaluations'] for g in records),
                    checkpoint_sha256=checkpoint_hash,
                    graph_id=metadata['graph_id'],dataset_id=metadata['dataset_id'],model_config=metadata['model_config'],
                    opponent=opponent_record,opponent_visits=16,seed=args.seed,games=records,
                    completed=len(completed),truncated=len(records)-len(completed),fly_wins=wins,
                    cpus=cpus,seconds=time.time()-start,
                    scope='Fresh shared four-move openings with swapped colors; small screening panel, no Elo claim')
        atomic_json(args.output/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='games'},indent=2),flush=True)


if __name__=='__main__':main()
