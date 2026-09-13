#!/usr/bin/env python3
"""Real-model compatibility, independent raw-NN values and a small 9x9 teacher panel."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import argparse
import json
from pathlib import Path
import time
import uuid

from flygo.data.corpus import atomic_json, identity
from flygo.data.generate import play_game
from flygo.data.katago import AnalysisClient, query, RULES, model_path
from flygo.data.label import parse_label
from flygo.go import Game
from flygo.gtp import GTPClient, action_to_vertex
from flygo.runtime import cpu_profile
from flygo.storage import GIB, StorageBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    parser.add_argument('--games', type=int, default=16)
    args = parser.parse_args()
    root = args.root
    registry = json.loads((root / 'runs/m2/candidates.json').read_text())
    out = root / 'runs/m2/data-qualification'
    out.mkdir(parents=True, exist_ok=True)
    cpus = cpu_profile()['research_cpus']
    binary = root / 'artifacts' / registry['engine']['sha256'] / 'katago'
    model = lambda r: model_path(root, r)

    def compatible(item):
        index, record = item
        path = out / ('compatibility-' + record['sha256'][:12])
        with AnalysisClient(binary, model(record), path, cpus[(index % 4)*8:(index % 4+1)*8]) as client:
            started = time.time()
            models = client.request(dict(action='query_models'))
            labels = []
            game = Game()
            for actions in ([], [40]):
                if actions:
                    game.play(1, 40)
                response = client.request(query(actions, visits=1))
                label = parse_label(response, game.legal(), game.state().to_play)
                labels.append(dict(actions=actions, raw_value=float(label['raw_value']), root=response['rootInfo']))
            return dict(model=record, models=models, labels=labels, seconds=time.time()-started, status='passed')

    with StorageBudget(root).reserve(files=64*1024**2, heap=32*GIB, purpose='teacher qualification panel'):
        candidates = registry['teacher_candidates'] + registry['opponents']
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(compatible, enumerate(candidates)))
        atomic_json(out / 'compatibility.json', results)
        print('All model compatibility queries passed', flush=True)
        # An independent GTP endpoint verifies the interpretation of rawWinrate.
        expert = registry['teacher_candidates'][0]
        config = out / 'raw-gtp.cfg'
        config.write_text((Path(__file__).resolve().parents[1] / 'configs/katago-qualification.cfg').read_text()
                          + '\nnnRandomize = false\n')
        raw_checks = []
        with GTPClient(['taskset', '-c', ','.join(map(str, cpus[:8])), str(binary), 'gtp',
                        '-model', str(model(expert)), '-config', str(config)], out / ('raw-gtp-' + uuid.uuid4().hex[:8])) as gtp:
            gtp.command('boardsize 9'); gtp.command('komi 7.5')
            gtp.command('kata-set-rules ' + json.dumps(RULES))
            for i, label in enumerate(results[0]['labels']):
                if i:
                    gtp.command('play B ' + action_to_vertex(40, 9))
                text = gtp.command('kata-raw-nn ' + ('B' if i == 0 else 'W') + ' 0', timeout=120)
                tokens = text.split()
                win = float(tokens[tokens.index('whiteWin') + 1])
                loss = float(tokens[tokens.index('whiteLoss') + 1])
                expected = (1 if i else -1) * (win - loss)
                delta = abs(expected - label['raw_value'])
                raw_checks.append(dict(to_play='B' if i == 0 else 'W', analysis=label['raw_value'],
                                       gtp_raw=expected, absolute_error=delta))
                if delta > 2e-5:
                    raise ValueError(f'Raw value perspective qualification failed: {raw_checks}')
        atomic_json(out / 'raw-values.json', dict(status='passed', checks=raw_checks))
        contract = dict(visits=16, max_moves=324, opening_moves=8)
        with ExitStack() as stack:
            clients = [stack.enter_context(AnalysisClient(binary, model(r), out / f'panel-{i}',
                        cpus[i*16:(i+1)*16], concurrency=8, eigen_threads=8, max_pending=24))
                       for i, r in enumerate(registry['teacher_candidates'])]
            panel = []
            def play(index):
                color = 1 + index % 2
                rows, actions, outcome = play_game(clients[0], clients[1], contract,
                                                   identity(['teacher-panel-v1', index // 2]), color)
                return dict(index=index, specialist_color=color, actions=actions, **outcome)
            with ThreadPoolExecutor(max_workers=8) as pool:
                for future in as_completed([pool.submit(play, i) for i in range(args.games)]):
                    record = future.result(); panel.append(record)
                    atomic_json(out / 'teacher-panel.json', dict(status='running', games=panel))
                    print(json.dumps(record), flush=True)
            specialist_wins = sum(g['terminal'] and ((g['white_score'] > 0) == (g['specialist_color'] == 2)) for g in panel)
            completed = sum(g['terminal'] for g in panel)
            # Small panels cannot establish a global ranking. Retain the specialist
            # unless the general network has a clear observed advantage.
            chosen = 1 if completed == args.games and specialist_wins <= args.games // 4 else 0
            selection = dict(status='passed', games=panel, completed=completed, specialist_wins=specialist_wins,
                             selected_teacher=registry['teacher_candidates'][chosen],
                             criterion='Retain 9x9 specialist unless current general wins at least 75% of completed panel',
                             limitation='Small 16-visit panel; no global strongest-model or calibrated Elo claim')
            atomic_json(out / 'teacher-panel.json', selection)
            print(json.dumps({k:v for k,v in selection.items() if k != 'games'}), flush=True)


if __name__ == '__main__':
    main()
