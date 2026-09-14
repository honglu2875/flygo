"""GTP adapter for a trained fixed-connectome prior and the native search layer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from .checkpoint import load_checkpoint
from .fly import FlyConfig,RustFly,load_graph,MODEL_VERSION
from .go import Game,GameConfig,GumbelConfig
from .gtp import action_to_vertex,vertex_to_action
from .runtime import cpu_profile,pin
from .storage import StorageBudget,GIB


def load_player(checkpoint:Path,graph_path:Path,*,threads=16):
    with np.load(checkpoint,allow_pickle=False) as data:
        metadata=json.loads(data['metadata'].tobytes())
        ports={key:data['port/'+key].copy() for key in ('input_index','output_group','output_scale')
               if 'port/'+key in data}
    if metadata.get('model_version',MODEL_VERSION)=='residual-cnn-v1':
        os.environ['JAX_PLATFORMS']='cpu'
        from .jax.cnn import CNNConfig,JaxCNN
        config=CNNConfig(**{**metadata['model_config'],'threads':threads})
        model=JaxCNN(config)
    else:
        config=FlyConfig(**{**metadata['model_config'],'threads':threads})
        model=RustFly(load_graph(graph_path),config,ports=ports)
    if (config.features,config.actions)!=(972,82):
        raise ValueError('This GTP profile requires a trained 9x9 Go model')
    load_checkpoint(checkpoint,model)
    return model,metadata


def choose_move(game:Game,model:RustFly):
    return choose_moves([game],model)[0]


def choose_moves(games,model):
    """Batch pending leaves across independent roots; Rust owns each search/history."""
    requests=[game.start(network=0) for game in games]
    while True:
        pending=[i for i,request in enumerate(requests) if request.identity is not None]
        if not pending:break
        output=model.infer(np.stack([requests[i].features for i in pending]))
        for b,i in enumerate(pending):
            requests[i]=games[i].evaluate(requests[i],output['logits'][b],float(output['value'][b]))
    return [game.finish() for game in games]


def search_config(simulations=0,search='puct'):
    gumbel=GumbelConfig() if search=='gumbel' and simulations else None
    return GameConfig(simulations=simulations,cpuct=0.0 if gumbel else 1.5,gumbel=gumbel)


def serve(model,*,simulations=0,search='puct',stdin=None,stdout=None):
    stdin=sys.stdin if stdin is None else stdin
    stdout=sys.stdout if stdout is None else stdout
    config=search_config(simulations,search)
    game=Game(config);history=[]
    commands=('protocol_version','name','version','known_command','list_commands','quit',
              'boardsize','clear_board','komi','play','genmove','undo','showboard','final_score')
    for line in stdin:
        fields=line.split('#',1)[0].split()
        if not fields:continue
        identifier=fields.pop(0) if fields[0].isdigit() else ''
        if not fields:continue
        command,*args=fields
        ok=True;response=''
        try:
            if command=='protocol_version':response='2'
            elif command=='name':response='FlyGo'
            elif command=='version':response='0.1.0'
            elif command=='known_command':response=str(args[0] in commands).lower()
            elif command=='list_commands':response='\n'.join(commands)
            elif command=='quit':pass
            elif command=='boardsize':
                if args!=['9']:raise ValueError('This checkpoint supports 9x9')
                game=Game(config);history=[]
            elif command=='clear_board':game=Game(config);history=[]
            elif command=='komi':
                if float(args[0])!=7.5:raise ValueError('This trained profile uses komi 7.5')
            elif command=='play':
                color={'b':1,'black':1,'w':2,'white':2}[args[0].lower()]
                action=vertex_to_action(args[1],9);game.play(color,action);history.append((color,action))
            elif command=='genmove':
                color={'b':1,'black':1,'w':2,'white':2}[args[0].lower()]
                if game.state().to_play!=color or game.state().terminal:raise ValueError('Wrong player or game is terminal')
                result=choose_move(game,model);game.play(color,result.action);history.append((color,result.action))
                response=action_to_vertex(result.action,9)
            elif command=='undo':
                if not history:raise ValueError('No move to undo')
                history.pop();game=Game(config)
                for color,action in history:game.play(color,action)
            elif command=='showboard':
                response='\n'+'\n'.join(f'{9-row} '+''.join('.XO'[int(x)] for x in stones)
                                         for row,stones in enumerate(game.state().stones))
            elif command=='final_score':
                state=game.state()
                if not state.terminal:raise ValueError('Final score requires a completed game')
                score=state.white_score
                response='0' if score==0 else ('W+' if score>0 else 'B+')+f'{abs(score):g}'
            else:raise ValueError('Unknown command')
        except (ValueError,KeyError,IndexError) as error:
            ok=False;response=str(error).replace('\n',' ')
        stdout.write(('=' if ok else '?')+identifier+(' '+response if response else '')+'\n\n');stdout.flush()
        if command=='quit' and ok:break


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--graph',type=Path)
    parser.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    parser.add_argument('--threads',type=int,default=16)
    parser.add_argument('--cpus')
    parser.add_argument('--simulations',type=int,default=0)
    parser.add_argument('--search',choices=['puct','gumbel'],default='puct')
    args=parser.parse_args(argv)
    cpus=list(map(int,args.cpus.split(','))) if args.cpus else cpu_profile()['research_cpus'][:args.threads]
    if args.threads>len(cpus):parser.error('Threads exceed CPU allocation')
    pin(cpus)
    graph=args.graph or Path(json.loads((args.root/'runs/m4/graph.json').read_text())['path'])
    with StorageBudget(args.root).reserve(files=0,heap=8*GIB,purpose='FlyGo GTP inference'):
        model,_=load_player(args.checkpoint,graph,threads=args.threads)
        serve(model,simulations=args.simulations,search=args.search)


if __name__=='__main__':main()
