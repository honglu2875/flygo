# FlyGo

Learn 9×9 Go with one fixed fly connectome: **165,122 neurons and 15,270,273
directed edges**. Neuron and edge identities remain fixed. Strengths, local
dynamics, sensory attachment, readout and internal passes are research variables.

The repository now runs expert generation on four CPU hosts, complete Rust
inference/backpropagation/Adam, offline teacher distillation, portable
checkpoints, and prior/PUCT/Gumbel play. JAX supplies an independent CPU
reference and a qualified four-host, 16-device TPU learner with exact sparse
forward/backward kernels. A separate residual CNN provides a compute-matched control.
The early trained models are weak; working training is not a strength claim.

- [Interactive model guide](docs/model.html): illustrated graph, Go adapters,
  recurrent passes, retinal controls, measured studies and proposed extensions. Open directly in
  a browser; all images, data and controls work offline.
- [PROGRESS.md](PROGRESS.md): current jobs, results and remaining gates.
- [DESIGN.md](DESIGN.md): equations, module boundaries and extension rules.
- [DATASET.md](DATASET.md): teacher, targets, splits and RAM storage policy.
- [MILESTONES.md](MILESTONES.md): acceptance criteria and research sequence.
- [RESEARCH.md](RESEARCH.md): comparison budgets, optimizer and biology hypotheses.
- [Controlled results](docs/research-results.md): completed comparisons and their limits.
- [Neuron-group study](docs/group-study.md): next-phase milestones, with the
  [measured density and regrowth proposal](docs/structural-plasticity.md).

**Everyday commands**

Run these from the repository on the first host. Runtime artifacts live under
`/dev/shm/flygo`; development builds do not replace running jobs' immutable
environments.

```bash
# Build and check Rust/Python, including optional independent JAX CPU parity.
python3 -B scripts/dev.py check --jax
# Optional build for this CPU's instruction set; qualified on these EPYC hosts.
python3 -B scripts/dev.py check --jax --native

# Read-only CPU kernel/profile check on spare cores, using frozen real Go data.
/dev/shm/flygo/venv/bin/python scripts/profile_cpu.py \
  --output /dev/shm/flygo/runs/my-cpu-profile --cpus 117,118,119

# Fresh-process prediction latency and peak RSS; no optimizer updates.
/dev/shm/flygo/venv/bin/python scripts/benchmark_prediction.py \
  --output /dev/shm/flygo/runs/my-prediction.json --batch-size 32

# Concise fleet status; --json is compact, --json --details includes full records.
python3 -B scripts/cluster.py status --run-id expert-v1

# Gracefully drain generation, or restart it with a measured runtime setting.
python3 -B scripts/cluster.py stop --run-id expert-v1
python3 -B scripts/cluster.py restart --run-id expert-v1 --concurrent-games 16

# Freeze once all opponent/color strata can supply the requested size.
/dev/shm/flygo/venv/bin/python scripts/freeze_corpus.py \
  --release v0-1m --positions 1000000 --wait

# Clone all currently complete games into a new research release.
# Original opening-family splits are preserved; the opponent mix is recorded.
/dev/shm/flygo/venv/bin/python scripts/freeze_corpus.py \
  --release research-snapshot-001 --selection all-complete

# One new offline run. Select currently available physical cores with --cpus.
/dev/shm/flygo/venv/bin/flygo train \
  --release screen-v1 --run-id my-k8-seed1 \
  --passes 8 --seed 1 --steps 1000 --batch-size 32 --threads 24

# Execute an explicit, disjoint four-host experiment plan.
# A plan name is immutable; use a new name for another experiment.
/dev/shm/flygo/venv/bin/python scripts/deploy_research.py \
  --config configs/screen-v1.json

# Qualified four-host TPU learner: one immutable SPMD cohort per run.
/dev/shm/flygo/venv/bin/python scripts/tpu.py launch \
  --run-id my-tpu-trial --mode train --train-plan configs/tpu-v0-b2048-v1.json
/dev/shm/flygo/venv/bin/python scripts/tpu.py status --run-id my-tpu-trial

# Interactive GTP; --simulations 0 uses the prior, positive budgets use search.
/dev/shm/flygo/venv/bin/flygo gtp \
  --checkpoint /dev/shm/flygo/runs/my-k8-seed1/checkpoints/step-00001000.npz \
  --simulations 16 --search puct

# Fresh, paired-color games against a calibrated KataGo checkpoint.
/dev/shm/flygo/venv/bin/flygo eval \
  --checkpoint /dev/shm/flygo/runs/my-k8-seed1/checkpoints/step-00001000.npz \
  --opponent 0 --games 16 --output /dev/shm/flygo/runs/my-match
```

All commands expose `--help`. Training uses a fixed release and uniform
position sampling with deterministic D4 augmentation. To continue a run,
pass the same release/model settings, `--resume <checkpoint>` and a larger
total `--steps`. Creating `<run>/stop` requests a checkpoint at the next safe
boundary. `status.json`, `metrics.jsonl`, `latest.json` and the last two
checkpoints are the run's small operational interface.

Fly trials accept `--ports <qualified.npz>` and `--rate-scales '{"bias":0.01}'`.
`--epsilon` controls Adam's denominator outside the square root (default 1e-8).
It is saved in the training contract; resume rejects an unintended change.
Rate multipliers affect the final Adam step after common global clipping and
moment estimation. The defaults preserve the baseline. `--model cnn` selects
the separate control; it uses the same data, loss, sampler and JAX optimizer.
TPU launch performs source/runtime replication and collective setup; do not
start unrelated cohorts on the same devices concurrently.

Optional `--warmup-steps 500` ramps to the peak rate. `--decay-until 4000
--final-rate-ratio 0.1` reaches the cosine floor at that absolute update;
the endpoint stays fixed across restarts. Resume validates the stored schedule.
`--diagnostic-batch-size 32` bounds extra activity/gradient measurements even
when the training batch is large. Constant-rate defaults preserve the baseline.

`--rate-softness 0.01` selects the experimental smooth rate
`s*softplus(v/s)` in recurrence and readout. The default zero keeps hard ReLU
and existing checkpoints. This factor has full CPU numerical qualification;
actual TPU qualification must precede its first TPU training run.
The optional `readout_mean_scale` experiment is retained for numerical
investigation: it fails strict full-model parameter-update parity, and its
learning trials are deferred. The normal value 1 preserves the qualified model.
`scripts/queue_cpu.py --plan <plan.json> --source <frozen-environment> --run-id
<queue-id>` freezes a one-case-per-host plan and waits for every declared lane
dependency before launch. `finish_study.py` supplies matching validation gates.

The active snapshot study is specified in
[configs/prototype-v1.json](configs/prototype-v1.json): four depth/readout
configurations, two paired seeds, 10,000 updates each. Periodic validation uses
fixed random 2,048-position slices shared by all variants. Complete validation
and fresh KataGo panels remain separate from these frequent progress metrics.
If deployment is interrupted, rerun the identical plan with `--resume-launch`;
it validates and adopts matching live trials, then starts the missing ones.

The cluster launcher pins generation to **64 physical cores per host**:
`0–31,60–91`, eight workers with eight cores each. Research uses
`32–59,92–119`; generation's SMT siblings are excluded from research profiles.
The four hosts have **480 physical / 960 logical CPUs in total**. Affinity is
applied only to FlyGo jobs. Explicit experiment plans check for overlapping
allocations within the plan; the operator must still account for existing jobs.

**Small programming interfaces**

```python
from pathlib import Path
from flygo.go import Game
from flygo.fly import FlyConfig, RustFly, load_graph

game = Game()                     # 9×9, komi 7.5, exact full rules history
game.play(1, 0)                    # Black at the top-left intersection

graph = load_graph(Path("/dev/shm/flygo/graphs/<graph-id>"))
model = RustFly(graph, FlyConfig(steps=8, threads=24))
prediction = model.infer(game.features()[None])
# model.loss_and_grad(features, legal, teacher_policy, teacher_value)
# model.train_step(features, legal, teacher_policy, teacher_value)
```

Actions are row-major, with `size²` for pass. Observations are
`[batch, size, size, 2*history+4]`; the 9×9 baseline has 972 features. Policy
logits have 82 actions and value is in the player-to-move perspective.
`Actors` in [go.py](python/flygo/go.py) supplies coarse batched search requests
for high-throughput callers. `choose_moves` in [play.py](python/flygo/play.py)
batches independent native searches around any compatible evaluator.

Native callers use `go_core::Board`, `go_actors::game::Game`, `go_actors::Pool`
and `fly_core::{Graph, Model, Params}` with `fly_core::optim::Adam`.
The fly numerical crate does not depend on Go, Python or JAX.

**Where to make a change**

| Area | Read/edit |
|---|---|
| Go rules, search, actors | `crates/go-core`, `go-search`, `go-actors`; imported unchanged with source hashes |
| Sparse forward, transpose, edge gradients | `crates/fly-core/src/sparse.rs` |
| Neuron equation and its backward rule | `crates/fly-core/src/recurrent.rs`, `python/flygo/jax/model.py`, `tests/test_fly.py` |
| Hard/smooth scalar rate and derivative | `crates/fly-core/src/rate.rs`, `python/flygo/jax/numerics.py`, `tests/test_dynamics.py` |
| Sensory/readout maps and initialization | `python/flygo/fly.py`; generic Rust composition in `fly-core/src/model.rs` |
| Audited spatial ports and shuffled controls | `python/flygo/ports.py`, `scripts/retinotopy.py` |
| Sensory path lengths and annotated visual types | `scripts/audit_circuit.py` |
| Loss and CPU Adam | `fly-core/src/model.rs`, `optim.rs`; matching JAX functions |
| Absolute-update learning-rate schedules | `python/flygo/schedule.py` |
| TPU sparse kernels and common learner | `python/flygo/jax/{sparse,numerics,learner}.py` |
| Independent residual-CNN control and FLOP ledger | `python/flygo/jax/cnn.py`, `python/flygo/cost.py` |
| Frozen data and learner | `python/flygo/data/loader.py`, `train.py` |
| Shared immutable feature cache | `python/flygo/data/cache.py` |
| Teacher protocol and production | `python/flygo/data/{katago,label,generate,corpus,service}.py` |
| Checkpoint portability and peer copies | `python/flygo/{checkpoint,replication}.py` |
| Placement, shared RAM and launch | `runtime.py`, `storage.py`, `scripts/{cluster,deploy_research,queue_cpu,tpu}.py` |
| Validation, dependent panels and paired analysis | `scripts/{compare_checkpoints,finish_study,summarize_study}.py` |

The initial equation is
`v[k+1] = (1-a)*v[k] + a*(W*ReLU(v[k]) + bias + sensory_input)`.
The sparse multiplication costs O(E×batch) per pass. Parameters are shared
across passes; activity resets for each board evaluation. Training differentiates
the K internal passes, while Go retains the real game's complete rules history.
Python `infer()` and Rust `Model::predict()` retain only the current recurrent
state, using O(N×batch) state memory independently of K. Training and explicit
`trace=True` retain O(K×N×batch) states for differentiation or inspection.
Both paths call the same propagation and readout code; graph/parameter storage
is separate from these state-memory counts.
Edge strengths use signed softplus magnitudes; type-shared leak lies in
`(0.01,0.99)`. The [design](DESIGN.md) specifies initialization and derivatives.

**Reproducibility and storage**

Graph preparation preserves original synapse counts and annotations separately
from learned parameters, and verifies every retained edge against the source
tables. The current source is the pinned MaleCNS v1.0 `Traced` selection with
at least two retained synapses per pair. It is not interchangeable with a
different fly release or threshold. Attribution is in [THIRD_PARTY.md](THIRD_PARTY.md).

Teacher raw policy/value label both players' turns. Played moves, searched
teacher labels and terminal outcomes remain separate fields. Frozen manifests
fix complete-game/opening-family splits and hashes. Validation includes D4
input novelty; final test labels remain outside architecture selection.

An `all-complete` release stores independent, checksum-verified copies under
its own `games/` directory, so ongoing generation cannot change its membership.
Its manifest records the cutoff, actual opponent mix and inherited splits.
The bounded loader accepts up to two million positions and preallocates typed
arrays once. Trials share a read-only memory-mapped cache under
`/dev/shm/flygo/cache/features/`, verified against its dataset identity and file
hashes. Cache construction has shared RAM admission and atomic publication;
reader leases protect live arrays, including NumPy views and readers in other
processes. Only registered, unused v2 caches may be reclaimed under pressure.
`scripts/cache.py` exposes inspection and reclamation. Corpus and checkpoint
files are not cache-eviction candidates.

Checkpoints contain canonical parameters, Adam moments/step, port assignments,
model configuration, graph/dataset IDs and sampler state. Local publication
precedes peer copying; a failed copy leaves a recoverable checkpoint and an
explicit retry status. The four-host launcher starts a replica coordinator on
host 0, which already has trusted SSH to the peers.

Each host enforces a shared **100 GiB FlyGo file cap**, **64 GiB free `/dev/shm`
floor** and **96 GiB available RAM floor**, including reservations and staging.
Pressure pauses new work. Frozen releases and checkpoints have verified peer
copies; RAM remains volatile across reboot or common cleanup. Owned persistent
SSH sessions are part of the current job lifetime arrangement.

The runtime package requires NumPy, Python 3.12 and the Rust extension. Optional
`jax` adds the CPU reference/CNN, `tpu` adds the pinned accelerator runtime and
`data` adds PyArrow for graph preparation.
Build tooling can reuse Python/Rust and verified KataGo artifacts from `~/go`;
the installed engine does not import that framework. The TPU path replicates
graph/parameters and shards the batch. Degree buckets, bounded gather tiles and
an explicit transpose/edge VJP avoid a full edge-by-batch message array. Actual
device ownership, all-group updates and fresh-process recovery are qualified;
new precision or dynamics still require their own checks.
