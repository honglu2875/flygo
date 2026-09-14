#!/usr/bin/env python3
"""Build/check this repo using an isolated RAM environment; no accelerator use."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from flygo.storage import GIB, StorageBudget
from flygo.runtime import cpu_profile, pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "check"])
    parser.add_argument("--storage-root", type=Path, default=Path("/dev/shm/flygo"))
    parser.add_argument("--reference-root", type=Path, default=Path.home() / "go")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--extended", action="store_true", help="Also run the million-transition oracle test")
    parser.add_argument("--cpus", help="Explicit physical CPUs; by default use the research allocation")
    parser.add_argument("--jax", action="store_true", help="Install the pinned optional JAX CPU reference")
    parser.add_argument("--data", action="store_true", help="Install PyArrow for graph source-table preparation")
    parser.add_argument("--native", action="store_true", help="Compile for this CPU's instruction set; qualify before copying to other CPU types")
    args = parser.parse_args()
    pin([int(value) for value in args.cpus.split(',')] if args.cpus else cpu_profile()['research_cpus'])
    if not 1 <= args.jobs <= len(os.sched_getaffinity(0)):
        parser.error("--jobs must fit the current CPU allocation")
    runtime = args.storage_root.resolve()
    reference = args.reference_root.resolve()
    python = shutil.which("python3.12") or str(reference / ".venv/bin/python")
    cargo = shutil.which("cargo") or str(reference / ".gozero/rustup/toolchains/1.98.1-x86_64-unknown-linux-gnu/bin/cargo")
    if not Path(python).is_file() or not Path(cargo).is_file():
        parser.error("Install Python 3.12 and Rust 1.98.1, or provide the existing --reference-root")
    uv = shutil.which("uv")
    if uv is None:
        parser.error("Install uv to build the isolated Python environment")
    venv = runtime / "venv"
    build = runtime / "cache/build"
    wheels = runtime / "cache/wheels"
    environment = {**os.environ, "CARGO_HOME": str(runtime / "cache/cargo"),
                   "CARGO_TARGET_DIR": str(build), "TMPDIR": str(runtime / "tmp"),
                   "UV_CACHE_DIR": str(runtime / "cache/uv"), "UV_PROJECT_ENVIRONMENT": str(venv),
                   "PYO3_PYTHON": str(venv / "bin/python"), "PYTHONDONTWRITEBYTECODE": "1",
                   "PATH": str(Path(cargo).parent) + os.pathsep + os.environ["PATH"]}
    if args.native:
        environment['RUSTFLAGS'] = (environment.get('RUSTFLAGS', '') + ' -C target-cpu=native').strip()

    def run(argv):
        print("Running:", " ".join(map(str, argv)), flush=True)
        subprocess.run([str(x) for x in argv], cwd=ROOT, env=environment, check=True)

    budget = StorageBudget(runtime)
    with budget.reserve(files=4 * GIB, heap=4 * GIB, purpose="development build/check"):
        for path in (runtime / "tmp", runtime / "cache/cargo", wheels):
            path.mkdir(parents=True, exist_ok=True)
        cached = reference / ".gozero/cargo/registry"
        if cached.is_dir() and not (runtime / "cache/cargo/registry").exists():
            shutil.copytree(cached, runtime / "cache/cargo/registry")
        extras=[value for name in ('jax','data') if getattr(args,name) for value in ('--extra',name)]
        # Preserve the last installed project while compiling its replacement.
        # Operational jobs use immutable snapshots; this also keeps local CLI
        # imports available during the dependency/build phase.
        run([uv, "sync", "--locked", "--inexact", "--no-install-project", "--python", python] + extras)
        run([cargo, "build", "--release", "--workspace", "--all-targets", "--locked", "-j", args.jobs])
        budget.check()
        run([venv / "bin/maturin", "build", "--release", "--locked", "--out", wheels])
        wheel = max(wheels.glob("flygo-*.whl"), key=lambda p: p.stat().st_mtime_ns)
        run([uv, "pip", "install", "--python", venv / "bin/python", "--reinstall", "--no-deps", wheel])
        (venv / 'build.json').write_text(json.dumps(dict(
            created=time.time(), rustflags=environment.get('RUSTFLAGS', ''),
            wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
            cpu=subprocess.check_output(['lscpu', '-J'],text=True),
            cpus=sorted(os.sched_getaffinity(0))), indent=2) + '\n')
        if args.action == "check":
            run([cargo, "test", "--workspace", "--locked", "-j", args.jobs])
            environment["FLYGO_RUST_PARITY"] = str(build / "release/examples/go-parity")
            run([venv / "bin/python", "-m", "unittest", "discover", "-s", "tests", "-v"])
            if args.extended:
                run([cargo, "test", "--release", "--locked", "-p", "go-core",
                     "million_transition_qualification", "--", "--ignored", "--nocapture"])
    print("Ready:", venv / "bin/flygo")


if __name__ == "__main__":
    main()
