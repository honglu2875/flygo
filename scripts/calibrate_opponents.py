#!/usr/bin/env python3
"""Small adjacent-checkpoint 9x9 panels, with shared openings and swapped colors."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import ExitStack
import argparse
import json
from pathlib import Path
import time

import numpy as np

from flygo.data.corpus import atomic_json
from flygo.data.katago import AnalysisClient,query,model_path
from flygo.data.label import parse_label
from flygo.go import Game
from flygo.storage import StorageBudget,GIB
from flygo.runtime import pin


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs',required=True,help='Comma-separated lower checkpoint indices')
    parser.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    parser.add_argument('--cpus',default='32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47')
    args=parser.parse_args()
    cpus=list(map(int,args.cpus.split(',')))
    pin(cpus)
    pairs=list(map(int,args.pairs.split(',')))
    root=args.root;contract=json.loads((root/'runs/expert-v1/config.json').read_text())['contract']
    binary=root/'artifacts'/contract['engine_sha256']/'katago'
    out=root/'runs/m2/ladder';out.mkdir(parents=True,exist_ok=True)
    openings=[]
    for seed in range(8):
        rng=np.random.default_rng(90210+seed);game=Game();actions=[]
        for ply in range(4):
            legal=[int(a) for a in game.legal() if a<81 and 1<=a//9<=7 and 1<=a%9<=7]
            a=int(rng.choice(legal));game.play(1+ply%2,a);actions.append(a)
        openings.append(actions)

    def panel(item):
        job,lower=item
        assigned=cpus[job*8:(job+1)*8]
        records=contract['opponents'][lower:lower+2]
        panel_path=out/f'pair-{lower}-{lower+1}.json'
        start=time.time()
        with StorageBudget(root).reserve(files=16*1024**2,heap=8*GIB,purpose=f'9x9 ladder panel {lower}'):
            with ExitStack() as stack:
                clients=[stack.enter_context(AnalysisClient(binary,model_path(root,r),out/f'pair-{lower}/model-{i}',
                         assigned,concurrency=4,eigen_threads=3,max_pending=8)) for i,r in enumerate(records)]
                def play(index):
                    lower_color=1+index%2;actions=openings[index//2].copy();game=Game()
                    for ply,a in enumerate(actions):game.play(1+ply%2,a)
                    for ply in range(len(actions),324):
                        color=game.state().to_play
                        client=clients[0 if color==lower_color else 1]
                        label=parse_label(client.request(query(actions,visits=16),timeout=180),game.legal(),color)
                        action=label['best_action'];game.play(color,action);actions.append(action)
                        if game.state().terminal:break
                    state=game.state()
                    return dict(index=index,lower_color=lower_color,opening=actions[:4],actions=actions,
                                terminal=state.terminal,white_score=state.white_score)
                games=[]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    for future in as_completed([pool.submit(play,i) for i in range(16)]):
                        games.append(future.result())
                        atomic_json(panel_path,dict(status='running',lower=lower,games=games))
                complete=[g for g in games if g['terminal']]
                wins=sum((g['white_score']>0)==(g['lower_color']==2) for g in complete)
                result=dict(status='passed',lower=lower,models=records,lower_wins=wins,completed=len(complete),
                            games=games,visits=16,cpus=assigned,seconds=time.time()-start,
                            scope='Adjacent-checkpoint panel at fixed visits; small-sample local 9x9 calibration, not human rank or global Elo')
                atomic_json(panel_path,result)
                print(json.dumps({k:v for k,v in result.items() if k not in ('games','models')}),flush=True)
    with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
        list(pool.map(panel,enumerate(pairs)))


if __name__=='__main__':
    main()
