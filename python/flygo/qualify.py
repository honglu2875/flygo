"""Actual CPU KataGo self-play checked through FlyGo's native game interface."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

from . import _native
from .go import Game, GameConfig, replay_observations
from .gtp import GTPClient, board_cells, vertex_to_action
from .storage import GIB, StorageBudget


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage_artifact(source: Path, target: Path, expected: str) -> Path:
    if target.exists():
        if sha256(target) != expected:
            raise ValueError(f"Cached artifact hash mismatch: {target}")
        return target
    if sha256(source) != expected:
        raise ValueError(f"Source artifact hash mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        temporary.chmod(source.stat().st_mode & 0o777)
        if sha256(temporary) != expected:
            raise ValueError("Artifact changed while copying")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def qualify_katago(*, reference_root: Path, registry: Path, config: Path,
                   storage_root: Path, output: Path, cpus: list[int], timeout: float) -> dict:
    """Qualification only: no claim about a trained FlyGo network's strength."""
    budget = StorageBudget(storage_root)
    with budget.reserve(files=GIB // 2, heap=4 * GIB, purpose="KataGo qualification"):
        return _qualify_game(reference_root=reference_root, registry=registry, config=config,
                             storage_root=storage_root, output=output, cpus=cpus,
                             timeout=timeout, budget=budget)


def _qualify_game(*, reference_root, registry, config, storage_root, output, cpus, timeout, budget):
    if not cpus or not set(cpus) <= os.sched_getaffinity(0) or timeout <= 0:
        raise ValueError("Choose available CPUs and a positive timeout")
    record = json.loads(registry.read_text())
    engine = record["engine"]
    model = record["model"]
    binary = stage_artifact(reference_root / engine["source_relative_path"],
                            storage_root / "artifacts" / engine["sha256"] / "katago", engine["sha256"])
    weights = stage_artifact(reference_root / model["source_relative_path"],
                             storage_root / "artifacts" / model["sha256"] / "model.bin.gz", model["sha256"])
    output.mkdir(parents=True, exist_ok=False)
    resolved_config = output / "katago.cfg"
    shutil.copyfile(config, resolved_config)
    result = dict(schema_version=1, kind="katago_rules_qualification", status="running",
                  claims_go_strength=False, size=9, komi=7.5, scoring="pass_alive_area",
                  engine_sha256=engine["sha256"], model_sha256=model["sha256"],
                  native_sha256=sha256(Path(_native.__file__)), config_sha256=sha256(resolved_config),
                  cpus=cpus, moves=[], started_unix=time.time())
    start = time.monotonic()
    deadline = start + timeout
    game = Game(GameConfig())
    argv = ["taskset", "-c", ",".join(map(str, cpus)), str(binary), "gtp",
            "-model", str(weights), "-config", str(resolved_config)]

    def command(client, text):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("KataGo game deadline expired")
        return client.command(text, timeout=min(remaining, 120))

    try:
        with GTPClient(argv, output / "katago") as kata:
            result["version"] = command(kata, "version")
            for text in ("boardsize 9", "clear_board", "komi 7.5"):
                command(kata, text)
            rules = json.loads(command(kata, "kata-get-rules"))
            result["katago_rules"] = rules
            required = {"ko": "POSITIONAL", "scoring": "AREA", "suicide": True,
                        "tax": "NONE", "hasButton": False}
            if any(rules.get(key) != value for key, value in required.items()):
                raise ValueError(f"KataGo rules differ from the declared profile: {rules}")
            actions = []
            for ply in range(324):
                color = 1 + ply % 2
                vertex = command(kata, "genmove " + ("B" if color == 1 else "W")).strip()
                action = vertex_to_action(vertex, 9)
                if action not in game.legal():
                    raise AssertionError(f"KataGo move illegal at ply {ply}: {vertex}")
                game.play(color, action)
                actions.append(action)
                state = game.state()
                actual = "\n".join("".join(".XO"[int(cell)] for cell in row) for row in state.stones)
                if actual != board_cells(command(kata, "showboard"), 9):
                    raise AssertionError(f"Board mismatch at ply {ply}")
                result["moves"].append(dict(color=color, vertex=vertex, action=action))
                if ply % 10 == 9:
                    budget.check()
                    print(json.dumps(dict(phase="katago_game", plies=ply + 1,
                                          seconds=time.monotonic() - start)), flush=True)
                if state.terminal:
                    score = command(kata, "final_score")
                    expected = "0" if state.white_score == 0 else (
                        ("W+" if state.white_score > 0 else "B+") + f"{abs(state.white_score):g}")
                    if score != expected:
                        raise AssertionError(f"Terminal score differs: native={expected}, KataGo={score}")
                    replay = replay_observations(actions, [0, len(actions)])
                    if replay.outcomes != [dict(terminal=True, white_score=state.white_score)]:
                        raise AssertionError("Packed replay terminal result differs")
                    np.testing.assert_array_equal(replay.stones[-1], state.stones)
                    result.update(status="passed", score=score, white_score=state.white_score,
                                  replay_rows=len(actions), board=actual)
                    break
            else:
                result.update(status="truncated", white_score=None)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        sgf = "(;GM[1]FF[4]CA[UTF-8]SZ[9]KM[7.5]RU[Tromp-Taylor]AP[flygo:qualification]"
        if result["status"] == "passed":
            sgf += "RE[" + result["score"] + "]"
        for move in result["moves"]:
            action = move["action"]
            vertex = "" if action == 81 else chr(97 + action % 9) + chr(97 + action // 9)
            sgf += ";" + ("B" if move["color"] == 1 else "W") + "[" + vertex + "]"
        (output / "game.sgf").write_text(sgf + ")\n")
    return result
