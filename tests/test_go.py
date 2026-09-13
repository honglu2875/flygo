from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from flygo.go import ActorConfig, Actors, Game, GameConfig, GumbelConfig, replay_observations
from flygo.gtp import action_to_vertex, vertex_to_action


FIXTURE = Path(__file__).parent / "fixtures/pass_alive_area.json"


def finish_move(actors, network=0):
    batch = actors.start(network)
    while batch.active_count:
        batch = actors.evaluate(batch, np.zeros((actors.config.games, actors.config.actions), np.float32),
                                np.zeros(actors.config.games, np.float32))
    return actors.commit()


def small_actors(**changes):
    config = ActorConfig(size=3, komi=0.5, history=2, games=2, workers=2,
                         simulations=4, max_search_edges=10000, max_game_moves=48,
                         dirichlet_fraction=0.25, temperature_moves=8, seed=27)
    return replace(config, **changes)


class GameTests(unittest.TestCase):
    def test_batched_search_matches_independent_roots(self):
        from flygo.play import choose_moves,search_config

        class Evaluator:
            def infer(self, x):
                occupied=x[:,:,:,0].reshape(len(x),9)
                logits=np.concatenate([occupied*.3+np.linspace(-.4,.4,9,dtype=np.float32)[None],
                                       np.full((len(x),1),-.2,np.float32)],axis=1)
                return dict(logits=logits,value=np.tanh(occupied.sum(axis=1)/9).astype(np.float32))

        evaluator=Evaluator()
        for gumbel in (None,GumbelConfig(max_considered_actions=8)):
            config=replace(search_config(16,'gumbel' if gumbel else 'puct'),size=3,history=2,komi=.5,gumbel=gumbel)
            prefixes=([],[0],[0,1,4])
            def game(prefix):
                result=Game(config)
                for ply,action in enumerate(prefix):result.play(1+ply%2,action)
                return result
            batched=[game(prefix) for prefix in prefixes]
            before=[g.features().copy() for g in batched]
            results=choose_moves(batched,evaluator)
            for prefix,actual,original,unchanged in zip(prefixes,results,batched,before):
                independent=game(prefix)
                request=independent.start()
                while request.identity is not None:
                    prediction=evaluator.infer(request.features[None])
                    request=independent.evaluate(request,prediction['logits'][0],float(prediction['value'][0]))
                expected=independent.finish()
                self.assertEqual(actual.action,expected.action)
                self.assertEqual(actual.value,expected.value)
                self.assertEqual(actual.neural_evaluations,expected.neural_evaluations)
                np.testing.assert_array_equal(actual.policy,expected.policy)
                np.testing.assert_array_equal(original.features(),unchanged)

    def test_coordinates_include_pass_skip_i_and_reject_outside_board(self):
        for size in (9, 19):
            for action in range(size**2 + 1):
                self.assertEqual(vertex_to_action(action_to_vertex(action, size), size), action)
        self.assertEqual(vertex_to_action("A9", 9), 0)
        self.assertEqual(vertex_to_action("J1", 9), 80)
        for text in ("I1", "A0", "A10", "resign", "A1\nquit"):
            with self.assertRaises(ValueError):
                vertex_to_action(text, 9)

    def test_fixture_replay_matches_each_pre_action_board_mask_and_terminal_score(self):
        fixture = json.loads(FIXTURE.read_text())
        actions = [vertex_to_action(v, 9) for v in fixture["action_vertices"]]
        for scoring, field in (("raw_area", "raw_white_minus_black"),
                               ("pass_alive_area", "katago_adjudicated_white_minus_black")):
            with self.subTest(scoring=scoring):
                replay = replay_observations(actions, [0, len(actions)], scoring=scoring)
                game = Game(GameConfig(scoring=scoring))
                for ply, action in enumerate(actions):
                    np.testing.assert_array_equal(replay.stones[ply], game.state().stones)
                    np.testing.assert_array_equal(np.flatnonzero(replay.legal[ply]), game.legal())
                    self.assertIsNone(game.state().white_score)
                    game.play(1 + ply % 2, action)
                state = game.state()
                self.assertTrue(state.terminal)
                self.assertEqual(state.white_score, fixture[field])
                self.assertEqual(replay.outcomes, [dict(terminal=True, white_score=fixture[field])])
                self.assertEqual("\n".join("".join(".XO"[int(c)] for c in row) for row in state.stones),
                                 fixture["board"])
                with self.assertRaises(ValueError):
                    game.play(1, 81)
                self.assertFalse(replay.stones.flags.writeable)

    def test_suffix_replays_prefix_legality_and_keeps_truncation_explicit(self):
        actions = [0, 1, 3, 4, 8, 6, 0, 9, 9]
        full = replay_observations(actions, [0, 9], size=3)
        part = replay_observations(actions, [0, 9], size=3, starts=[4])
        np.testing.assert_array_equal(part.stones, full.stones[4:])
        np.testing.assert_array_equal(part.legal, full.legal[4:])
        self.assertEqual(part.outcomes, full.outcomes)
        self.assertEqual(replay_observations([0], [0, 1], size=3).outcomes,
                         [dict(terminal=False, white_score=None)])
        for moves in ([0, 0, 1], [9, 9, 1]):
            with self.assertRaises(ValueError):
                replay_observations(moves, [0, 3], size=3, starts=[2])

    def test_histories_reject_lossy_float_and_overflow_conversions(self):
        for actions, offsets in (([0.5], [0, 1]), ([2**32 + 81], [0, 1]),
                                 ([0], [0.0, 1.0]), ([0], [0, 2**64 - 1])):
            with self.assertRaises(ValueError):
                replay_observations(actions, offsets)

    def test_leaf_history_is_causal_and_stale_evaluations_are_rejected(self):
        for gumbel in (None, GumbelConfig()):
            config = GameConfig(size=3, komi=0.5, history=2, simulations=32,
                                cpuct=1.5 if gumbel is None else 0, gumbel=gumbel)
            game = Game(config)
            game.play(1, 0)
            game.play(2, 1)
            before = game.state().stones.copy()
            request = game.start(27)
            old = request
            deepest = 0
            count = 0
            while request.identity is not None:
                history = game.request_history(request)
                self.assertEqual(history[:2].tolist(), [0, 1])
                deepest = max(deepest, len(history) - 2)
                replay = Game(replace(config, simulations=0, gumbel=None))
                for ply, action in enumerate(history):
                    replay.play(1 + ply % 2, int(action))
                np.testing.assert_array_equal(request.features, replay.start(0).features)
                if count:
                    with self.assertRaises(ValueError):
                        game.evaluate(old, np.zeros(10, np.float32), 0)
                request = game.evaluate(request, np.zeros(10, np.float32), 0)
                count += 1
            self.assertGreaterEqual(deepest, 2)
            inspection = game.inspect_search()
            result = game.finish()
            self.assertEqual(result.simulations, 32)
            self.assertEqual(result.action, inspection["chosen"])
            np.testing.assert_array_equal(game.state().stones, before)

    def test_gumbel_target_is_completed_policy_and_not_visit_proportions(self):
        game = Game(GameConfig(size=3, komi=0.5, history=2, simulations=4,
                               cpuct=0, gumbel=GumbelConfig()))
        request = game.start(1)
        while request.identity is not None:
            request = game.evaluate(request, np.zeros(10, np.float32), 0)
        inspection = game.inspect_search()
        result = game.finish()
        self.assertEqual((result.action, result.simulations, result.neural_evaluations), (0, 4, 5))
        np.testing.assert_allclose(result.policy, np.full(10, 0.1), atol=1e-7, rtol=0)
        self.assertFalse(np.allclose(result.policy, np.asarray(inspection["visits"]) / 4))


class ActorTests(unittest.TestCase):
    def test_bad_batches_do_not_consume_pending_request(self):
        with Actors(small_actors()) as actors:
            batch = actors.start(11)
            self.assertEqual(batch.features.shape, (2, 3, 3, 8))
            self.assertFalse(batch.features.flags.writeable)
            for logits in (np.zeros((1, 10), np.float32), np.full((2, 10), np.nan, np.float32)):
                with self.assertRaises(ValueError):
                    actors.evaluate(batch, logits, np.zeros(2, np.float32))
            with self.assertRaises(ValueError):
                actors.evaluate(replace(batch, network=12), np.zeros((2, 10), np.float32), np.zeros(2))
            with self.assertRaises(RuntimeError):
                actors.checkpoint()
            while batch.active_count:
                batch = actors.evaluate(batch, np.zeros((2, 10), np.float32), np.zeros(2, np.float32))
            actors.commit()
            self.assertIsInstance(actors.checkpoint(), str)

    def test_worker_count_independent_restore_and_current_player_targets(self):
        completed = 0
        config = small_actors(max_game_moves=128)
        with Actors(config) as original:
            for network in range(4):
                finish_move(original, network)
            checkpoint = original.checkpoint()
            with Actors(replace(config, workers=1), checkpoint=checkpoint) as restored:
                for network in range(4, 48):
                    a, b = finish_move(original, network), finish_move(restored, network)
                    self.assertEqual(a.games, b.games)
                    for field in ("features", "policies", "outcomes", "root_values", "metadata", "ownership"):
                        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
                    for game in a.games:
                        if game["truncated"]:
                            continue
                        completed += 1
                        take = a.metadata[:, 0] == game["game_id"]
                        signs = 1 - 2 * a.features[take, 0, 0, -4]
                        np.testing.assert_array_equal(a.outcomes[take], signs * np.sign(game["white_score"]))
                        np.testing.assert_array_equal(a.ownership[take].reshape(-1, 9),
                                                      signs[:, None] * np.asarray(game["ownership"]))
        self.assertGreater(completed, 0)
        with self.assertRaises(ValueError):
            Actors(replace(config, seed=99), checkpoint=checkpoint)

    def test_move_cap_has_no_outcome_rows(self):
        with Actors(small_actors(max_game_moves=1)) as actors:
            rows = finish_move(actors)
        self.assertEqual(len(rows.games), 2)
        self.assertEqual(rows.features.shape, (0, 3, 3, 8))
        for game in rows.games:
            self.assertTrue(game["truncated"])
            self.assertIsNone(game["white_score"])
            self.assertEqual(game["ownership"], [])

    @unittest.skipUnless(os.environ.get("FLYGO_RUST_PARITY"), "Build and set FLYGO_RUST_PARITY for cross-language qualification")
    def test_direct_rust_and_python_callers_agree_bit_for_bit(self):
        config = small_actors()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(asdict(config)))
            expected = json.loads(subprocess.check_output([os.environ["FLYGO_RUST_PARITY"], str(path)], timeout=60))
        requests, commits = [], []
        def bits(array):
            return array.ravel().view(np.uint32).tolist()
        with Actors(config) as actors:
            for network in range(32):
                batch = actors.start(network)
                while True:
                    requests.append(dict(round=batch.round, network=batch.network,
                                         features=bits(batch.features), active=batch.active.tolist()))
                    if not batch.active_count:
                        break
                    batch = actors.evaluate(batch, np.zeros((2, 10), np.float32), np.zeros(2, np.float32))
                rows = actors.commit()
                commits.append(dict(games=rows.games, rows=[dict(
                    features=bits(rows.features[i]), policy=bits(rows.policies[i]),
                    action=int(rows.metadata[i, 2]), network=int(rows.metadata[i, 1]),
                    root_value=int(rows.root_values.view(np.uint32)[i]),
                    black_to_play=bool(rows.features[i, 0, 0, -4])) for i in range(len(rows.outcomes))]))
        self.assertEqual(requests, expected["requests"])
        self.assertEqual(commits, expected["commits"])


if __name__ == "__main__":
    unittest.main()
