#!/usr/bin/env python3
"""Compare resident game concurrency within one fixed eight-physical-core allocation."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import time

from flygo.data.corpus import atomic_json,identity
from flygo.data.generate import play_game
from flygo.data.katago import AnalysisClient,model_path
from flygo.runtime import pin
from flygo.storage import StorageBudget,GIB


def main():
    root=Path('/dev/shm/flygo');cpus=list(range(92,100));pin(cpus)
    contract=json.loads((root/'runs/expert-v1/config.json').read_text())['contract']
    out=root/'runs/m3/generation-benchmark';out.mkdir(parents=True,exist_ok=True)
    binary=root/'artifacts'/contract['engine_sha256']/'katago'
    results=[]
    with StorageBudget(root).reserve(files=64*1024**2,heap=12*GIB,purpose='generation concurrency benchmark'):
        for concurrency in (4,8,16):
            with ExitStack() as stack:
                teacher=stack.enter_context(AnalysisClient(binary,model_path(root,contract['teacher']),out/f'c{concurrency}/teacher',
                    cpus,concurrency=concurrency,eigen_threads=4,max_pending=32))
                opponent=stack.enter_context(AnalysisClient(binary,model_path(root,contract['opponents'][3]),out/f'c{concurrency}/opponent',
                    cpus,concurrency=concurrency,eigen_threads=2,max_pending=32))
                teacher.request(dict(action='query_models'));opponent.request(dict(action='query_models'))
                start=time.perf_counter()
                def game(i):return play_game(teacher,opponent,contract,identity(['concurrency-benchmark',i]),1+i%2)
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    games=list(pool.map(game,range(16)))
                elapsed=time.perf_counter()-start;positions=sum(len(g[1]) for g in games)
                record=dict(concurrent_games=concurrency,cpus=cpus,games=len(games),positions=positions,
                            seconds=elapsed,positions_per_second=positions/elapsed,
                            caveat='One intermediate opponent; short profile, not a guaranteed cluster speedup')
                results.append(record);atomic_json(out/'result.json',results);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
