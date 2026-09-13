"""Implemented commands only; later milestones add data and learning commands."""
import argparse
import json
import os
from pathlib import Path
import time
import sys


def main() -> int:
    if len(sys.argv)>1 and sys.argv[1] in ('train','gtp','eval'):
        if sys.argv[1]=='train':
            from .train import main as command
        elif sys.argv[1]=='gtp':
            from .play import main as command
        else:
            from .evaluate import main as command
        command(sys.argv[2:])
        return 0
    parser = argparse.ArgumentParser(prog="flygo")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser('train',help='Train the fixed fly graph on a frozen expert corpus')
    commands.add_parser('gtp',help='Play 9x9 Go from a trained checkpoint over GTP')
    commands.add_parser('eval',help='Evaluate a trained prior against a fixed KataGo checkpoint')
    qualify = commands.add_parser("qualify", help="Check a real CPU KataGo game against native Go")
    qualify.add_argument("--reference-root", type=Path, default=Path.home() / "go")
    qualify.add_argument("--registry", type=Path, default=Path("configs/katago-bootstrap.json"))
    qualify.add_argument("--config", type=Path, default=Path("configs/katago-qualification.cfg"))
    qualify.add_argument("--storage-root", type=Path, default=Path("/dev/shm/flygo"))
    qualify.add_argument("--output", type=Path)
    qualify.add_argument("--cpus", default=",".join(map(str, sorted(os.sched_getaffinity(0))[:8])))
    qualify.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    from .qualify import qualify_katago
    output = args.output or args.storage_root / "runs" / ("qualify-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    result = qualify_katago(reference_root=args.reference_root, registry=args.registry,
                           config=args.config, storage_root=args.storage_root, output=output,
                           cpus=[int(x) for x in args.cpus.split(",")], timeout=args.timeout)
    print(json.dumps({k: v for k, v in result.items() if k not in ("moves", "board")}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
