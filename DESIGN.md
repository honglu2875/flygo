FlyGo design, 2026-09-13. The CPU baseline is implemented, using the existing
`~/go` rules/search source and the pinned MaleCNS data. This document separates
the current numerical contract from later extensions; acceptance evidence is
recorded in [PROGRESS.md](PROGRESS.md).

The research question is how well a fixed fly connectome can learn Go when we
optimize its numerical parameters, its interface to the board, its internal
computation, and the search used to improve its predictions. The project will
use all 165,122 selected neurons and all 15,270,273 selected directed edges.
Smaller boards reduce task complexity; they do not reduce the fly graph.

The user has confirmed that topology is fixed and strengths may learn. The
canonical node IDs and edge list are immutable. Numerical weights may become
weak or temporarily gated, but edges are not deleted, added, or rewired. Index
permutations, degree buckets, and transposed layouts are implementation details
and must map back to the same canonical edges. Additional neuron state is
allowed. Learned communication between different fly neurons must use existing
edges; a dense recurrent controller would break this contract. Input and output
adapters are explicit task interfaces, with their parameter counts reported.

The first learning stage is offline distillation from a large 9×9 KataGo corpus.
One fixed strong expert plays sampled historical checkpoints, with random
expert color. The strong teacher labels both players' states. Architecture
comparisons use identical frozen data, validation losses and actual KataGo
matches. This data pipeline starts immediately after Go-engine qualification,
while the fly engine is developed. [DATASET.md](DATASET.md) specifies the corpus.

The later improvement loop remains network evaluation, search, self-play and
distillation. The fly supplies policy/value predictions; exact Go search
improves action selection and supplies new targets. Offline teacher data and
online replay feed the same learner through explicitly typed target records.

**Ownership and repository boundaries.**

| Component | Responsibility | Provenance |
|---|---|---|
| `crates/go-core` | Board, legality, superko, apply/undo, terminal scoring | Copy existing crate and its tests |
| `crates/go-search` | PUCT, Gumbel, search requests, backup, improvement targets | Copy existing crate and its tests |
| `crates/go-actors` | Persistent games, batched requests, move sampling, completed targets | Copy existing crate and its tests |
| `crates/fly-core` | Graph arrays, sparse kernels, recurrence, generic ports/heads, full derivatives and CPU optimizer | New; independent of Go, Python and JAX |
| `crates/flygo-python` | Thin PyO3 batch bindings for Go and fly computation | Adapt the relevant `go-bridge` bindings |
| `python/flygo` | Configuration, replay scheduling, training loop, evaluation, checkpoints and launch | One small Python package |
| `python/flygo/data` | KataGo generation, labeling and immutable corpus loading | Bootstrap from useful `~/go` infrastructure; small permanent adapter |
| `python/flygo/jax` | Independent implementation of supported model equations, gradients and updates | JAX CPU reference and future TPU execution |

The three reusable Go crates total about 288 KiB of source directories in the
inspected checkout. Their public API, tests, and existing KataGo/Mctx license
notices are worth preserving. Keep their initial behavior intact and record
source-file hashes. The new binding exposes the needed board, actor and search
operations; it need not expose every historical trace experiment.

The existing rules engine has reversible incremental chains, exact
collision-checked positional superko, allowed multi-stone suicide, and two
explicit scoring profiles. Start with `pass_alive_area` for training and
matching the existing qualified KataGo evaluation setup. Pin size, komi, rules,
and scoring in every run. Keep the independent scoring/oracle tests. Preserve
the rule that move-cap truncations are recorded and produce no fabricated
terminal targets.

The existing native observations contain current-player/opponent stone history,
side to move, signed komi divided by area, consecutive passes, and point
legality: `[batch, size, size, 2 * history + 4]`. Full superko history remains in
Rust independently of this finite observation window. Use these features
unchanged initially, including their perspective convention. Policy has
`size² + 1` actions, including pass; value and ownership use the current player's
perspective. The existing separate JAX Go implementation is not needed for the
first version: both network backends can use the same authoritative Rust Go
engine.

There is one offline learner loop; an online batch source will feed the same
numerical training API. Models are named, versioned implementations, not
copied training recipes. Explicit experiment plans and hardware profiles
describe placement and resources without changing scientific settings.

```text
Cargo.toml, Cargo.lock, rust-toolchain.toml
pyproject.toml, uv.lock
crates/
  go-core/                    Copied rules/scoring implementation and tests
  go-search/                  Copied PUCT/Gumbel implementation and tests
  go-actors/                  Copied game workers and target generation
  fly-core/src/
    lib.rs                    Public Rust numerical API
    graph.rs                  Validated CSR and canonical transpose edge IDs
    sparse.rs                 Forward, transpose and edge-gradient kernels
    recurrent.rs              Core parameters, rate equation, K-step tape and VJP
    model.rs                  Generic ports, pooling, heads, losses and derivatives
    optim.rs                  Complete CPU Adam update
  flygo-python/src/
    lib.rs, game.rs, observations.rs, fly.rs   Thin NumPy/PyO3 boundaries
python/flygo/
  go.py                       Typed Game/Actors/replay API
  fly.py                      Model config, graph loading, ports and initialization
  train.py                    One offline learner loop
  play.py, evaluate.py         GTP and paired prior/search evaluation
  checkpoint.py               Portable arrays, sampler state and contract checks
  replication.py              Verified immutable peer transfers and bundles
  runtime.py, storage.py       Physical-core placement and shared RAM admission
  cli.py, gtp.py, qualify.py   Public commands and Go/KataGo protocol qualification
  data/
    katago.py, label.py        Resident teacher protocol and target interpretation
    generate.py, service.py    Concurrent games and one bounded host supervisor
    corpus.py, loader.py       Immutable game records, frozen releases and sampling
    cache.py                  Bounded shared feature arrays and verified read-only maps
  jax/model.py                Independent recurrence, loss, gradients and Adam
configs/                      Frozen teacher contract and explicit experiment plans
scripts/
  dev.py, inventory.py        Build/check and hardware inventory
  prepare_graph.py            Canonical source-table verification and graph artifact
  cluster.py                  Generation start/status/drain/restart
  freeze_corpus.py             Balanced releases or current-data clones, with peer verification
  deploy_research.py           Immutable environment/data and disjoint CPU trials
  replicate_checkpoints.py    Star-topology checkpoint coordinator
  compare_checkpoints.py      Common validation slices and game-level uncertainty
  fit_subset.py               Small real-training-data optimization diagnostic
  qualify_*.py, verify_restore.py, diagnostic_fly.py  Focused acceptance helpers
tests/                        Rules, numerical parity, targets, restore and budgets
README.md, DESIGN.md, DATASET.md, MILESTONES.md, PROGRESS.md
```

A separate dynamics module will be introduced with the first additional neuron
model. Online replay, a TPU-specific sparse kernel and auxiliary heads are
added only when their milestones require them.

Source and configuration stay in this repository. Graph artifacts, datasets,
checkpoints, logs, caches and temporary files resolve under the configurable
runtime root, initially `/dev/shm/flygo` on each host. See [DATASET.md](DATASET.md)
for its layout and host-wide budget; no runtime dataset directory is required
inside the source tree.

Create modules as their milestone needs them; the tree is not a request to
prebuild a plugin system, general computation DSL, or distributed control plane.
A person changing neuron decay should read the dynamics specification, its
Rust implementation, its JAX counterpart, and their focused tests. They should
not need to understand SSH launch or MCTS.

The public boundaries use these few records. B is batch size, N neuron count,
E edge count, L board side length, and H observation history depth.

| Record | Contents and lifetime |
|---|---|
| `Graph` | Immutable N/E arrays, metadata and graph identity; shared by models |
| `ModelSpec` | Versioned equations, ports, K and parameter/state declarations |
| `Parameters` | Named arrays in canonical order; shared across positions and K passes |
| `NeuralState` | Named N×B state channels for one evaluation; never optimizer state |
| `ObservationBatch` | FP32 `[B,L,L,2H+4]`, legal action mask, active rows and request identities |
| `PredictionBatch` | Logits `[B,L²+1]`, value `[B]`, optional ownership `[B,L,L]` and score `[B]` |
| `SearchResult` | Chosen move, policy target, value estimates, version and measured work |
| `TrainingBatch` | Observations, typed policy/value/auxiliary targets, separate validity/weight masks and provenance |
| `TrainState` | Parameters, optimizer arrays and update/RNG state; no actor or search ownership |

`infer` reads model parameters and returns predictions. `loss_and_grad` returns
loss terms and named gradients without updating parameters. `train_step`
applies the specified optimizer and returns metrics plus updated training
state. Rust may update owned buffers internally; JAX may return new arrays.
The observable semantics are identical. Checkpoint orchestration combines
`TrainState` with separately owned actors/replay. Scratch buffers and derivative
tapes stay internal to the backend. This keeps the neural Rust API useful
without Python and prevents search or deployment concerns entering dynamics.

**The shared mathematical contract.**

Use canonical edge order `(destination, source)` with a separate biological
body-ID array. A graph artifact contains the selected IDs, `src`, `dst`, synapse
counts, initial weight prior, type/NT/side annotations and missingness masks.
The manifest records dataset release, raw-file checksums, `Traced` selection,
minimum synapse count 2, and hashes of the canonical arrays. These choices
define the fixed graph; changing a threshold is not an ordinary experiment.
Forward CSR, transposed CSR with canonical edge IDs, and TPU layouts are
rebuildable caches, not different datasets.

Use `manifest.json` plus uncompressed `.npy` numeric arrays for graph artifacts,
with explicit shapes/dtypes and small JSON lookup tables for annotation labels.
This permits memory-mapped CPU access and ordinary NumPy loading. Feather/Arrow
is a preparation dependency; it need not enter the Rust execution engine.

For neuron state `v` and a batch of board observations `x`, the first model is
a leaky rate network:

\[
I=P_{\mathrm{in},\theta}(x),\qquad v^0=v_{\mathrm{rest}},
\]
\[
r^k=\operatorname{ReLU}(v^k),\qquad
v^{k+1}=(1-\alpha)\odot v^k+
\alpha\odot\left(W_\theta r^k+b+I\right),\quad k=0,\ldots,K-1.
\]
\[
r^K=\operatorname{ReLU}(v^K),\qquad
(\ell_\pi,z_v,\ell_o,z_s)=H_\theta(P_{\mathrm{out},\theta}(r^K)),
\qquad v_{\mathrm{Go}}=\tanh(z_v).
\]

`W[destination, source]` has exactly the permitted edge slots. One trainable
scalar magnitude per edge is the principal baseline. Initialize from the
previously reconstructed signed, incoming-normalized `log1p(synapse_count)`
prior. A sign-preserving softplus parameterization can reproduce its initial
magnitudes; a later signed-weight experiment can relax the sign assumption
without changing topology. NT-derived signs are modeling priors, not measured
physiological truth. Keep counts and initial weights distinct from parameters.

Learn biases and bounded leak coefficients initially by cell type, with an
explicit fallback group for missing type labels. Initialize a nonzero resting
activity and inspect response/gradient coverage: a zero-state ReLU network
driven through inhibitory sensory outputs can otherwise be ineffective.
Bounding the leak alone does not guarantee recurrent stability. Monitor
activity growth, saturation, gradient norms and sensitivity to perturbations;
document any gain constraints or regularization. Initial computational time
constants are in internal-step units, with no claim of biological milliseconds.

The implemented `leaky-rate-v1` contract uses
`W[e]=sign[src[e]]*softplus(edge[e])` and
`alpha[type]=0.01+0.98*sigmoid(leak[type])`. Initial values are `v_rest=0.01`,
`bias=0.02`, `leak=0` (alpha=0.5), sensory feature gains 0.2 and readout gains 1.
The nine named parameter arrays live in `fly.py`; the Rust model owns them.
`adam-fp32-v1` uses FP32 moments, beta1=0.9, beta2=0.999, epsilon=1e-8, and a
global gradient norm accumulated in FP64, clipped before moment updates.
The first trials use learning rate 0.003 and clip norm 1. Checkpoints name the
contracts; early schema-1 files with no explicit names have these same first
contracts. New equations or transforms require new contract names.

Parameters are shared across the K passes. Start with K=8 as a benchmark
candidate, then compare 2, 4, 8 and 16. A board-dependent current enters the
first update, so information crossing d graph edges requires at least d+1
updates under this particular initialization convention. Greater K changes
processing depth and cost; it does not guarantee stronger predictions.

Reset all neural activity for every independent board evaluation initially.
Train through K internal steps, not through an entire Go game. Offline targets
come from the strong teacher; later online game credit can come from outcomes
and search. This makes inference
a deterministic function of observation and model version and lets the
existing search interface work unchanged. Rust still retains complete rules
history. If caching search results is added, a board-only hash is insufficient
for rules-sensitive state; observation and full rules-state identities have
different roles.

**Input and output are research variables with explicit budgets.**

The first implementation uses a deterministic, balanced assignment of the
972 board features to all 15,912 annotated sensory cells, with one learned gain
per feature. Komi and other global features are already in the observation.
Type-conditioned projections remain an input-adapter experiment. Broadcast the resulting
sensory current on each internal pass. Assignment seeds, coverage and parameter
counts are part of the experiment. Changing board size changes the adapter,
not the graph.

A design-time audit of the cached graph found 4,114 `ol_sensory` cells and
15,912 cells whose superclass includes `sensory`. None of these candidate
sensory cells has both optic-column coordinates in the annotation table.
Thus direct retinotopic coordinates are not presently available for these
ports. A geometry-based alternative could infer them from connections to
column-annotated neurons, but it must be labeled as inferred and tested against
the balanced assignment.

Boolean reachability, ignoring signs and dynamics, was measured by repeatedly
applying the binary destination-by-source adjacency to the reached set:

| Input set | Initially | Within 2 graph hops | Within 4 graph hops | Eventually |
|---|---:|---:|---:|---:|
| Visual sensory | 4,114 | 94,572 | 163,908 | 164,133 |
| All annotated sensory | 15,912 | 152,928 | 164,650 | 164,650 |

Counts include the inputs. Structural reachability is not evidence of useful
signal propagation or learning. The unreachable neurons remain in the model.
Measure activity and gradient coverage as well as these static paths.

The first readout assigns all 149,210 nonsensory neurons to 656 balanced pools.
Each neuron's contribution is multiplied by its learned gain and the reciprocal
square root of its pool size. A small dense 656→82 head produces policy logits,
and a 656→1 tanh head produces value. Pool assignments stay fixed within a run.
This is 220,625 stored adapter/head parameters, including unused gain slots for
the sensory neurons, and 15,514,402 parameters in total. The initial runtime
keeps this generic composition in `fly-core/model.rs`; a separate integration
crate is unnecessary while the model remains this small.

Shared pointwise heads, type/anatomy-aware pools, inferred spatial assignments,
factorized readouts and auxiliary ownership/score heads remain controlled
variants. The current pooling map is a task adapter, not a claim that a fly
neuron represents a Go intersection. Input and readout are disjoint, so there
is no direct board-to-head bypass.

Avoid a first implementation with a dense `neurons × board-actions` readout:
at 19×19 that alone is about 60 million coefficients. Set an initial combined
adapter/head budget below one million parameters and report its compute. All
learned predictions derive from fly activity; board features do not bypass the
core into a separate predictor. Input and readout sets are disjoint initially,
so board information must cross a fly connection. Apply exact legality outside
the learned policy, and use the same mask in the policy loss. D4 augmentation
transforms board features, policy and ownership together; the biological graph
is not assumed to have square-board symmetry.

**Rust CPU computation and JAX have one contract, two implementations.**

The Rust engine supports forward evaluation, backward propagation, losses and
optimizer updates. Python schedules batches but contains no loop over neurons,
edges, search nodes, or individual Go transitions. The Python binding exposes
coarse `infer`, `loss_and_grad`, and `train_step` calls. A Rust caller can use
the same model and Go actor APIs without embedding Python. A later fused Rust
actor/evaluator loop can remove batch crossings if profiling justifies it.

`fly-core` has explicit graph, sparse-kernel, dynamics, and recurrence modules.
A dynamics implementation owns its parameter/state declarations, one-step
forward rule and vector-Jacobian product. Local nonlinear functions and sparse
communication stay separate. Add named variants such as `rate_v1` and
`adaptive_rate_v1`, with corresponding JAX functions and fixture tests. Do not
hide unimplemented features behind silently ignored flags. A capability check
rejects a model unsupported by the selected backend.

For the sparse primitive `Y = W H`, use the identities

\[
\bar H=W^T\bar Y,\qquad
\bar w_e=\sum_b \bar Y_{\mathrm{dst}(e),b}H_{\mathrm{src}(e),b}.
\]

Implement forward accumulation by destination rows, state gradients using a
transposed layout, and edge gradients as batch dot products. Accumulate into
canonical edge order across recurrent steps. This avoids dense N×N gradients
and an E×B message tensor. Local VJPs also differentiate leak, biases, input
mapping and readout parameters. Backpropagation through time and optimizer
state must cover all trainable arrays, not merely the policy head.

Start with safe Rust, contiguous FP32 arrays, tiled batches and bounded Rayon
thread pools. Prefer parallel row/edge ownership over floating-point atomic
scatter. Define a deterministic reduction mode for qualification. Benchmark
single-socket execution and NUMA placement before adding more threads; this
graph can be constrained by memory bandwidth. SIMD, high-degree row splitting,
checkpointed recurrence and alternate layouts follow measured bottlenecks.

JAX initially uses an easy-to-audit CPU implementation and autodiff as an
independent check of the Rust derivatives. Its model and named parameter arrays
match Rust. Optimizer equations, epsilon placement, reductions, clipping,
regularization, constraints and update order are specified identically. Use
one optimizer initially, Adam with explicit parameter groups; optimizer choice
is a later experiment. Export canonical arrays, not backend-specific object
pickles. CPU-only Rust execution has no JAX dependency; JAX parity processes
set `JAX_PLATFORMS=cpu` before importing it.

For scale: FP32 weights plus int32 CSR occupy about 123 MB. One neuron-state
array at batch 128 occupies 84.5 MB; storing K+1 such arrays at K=8 is about
761 MB before other buffers. One per-edge, per-example array would occupy
7.82 GB. CPU training is therefore a plausible engineering target, but raw
memory capacity is not a throughput result. Measure complete forward/backward
and optimizer updates on the actual graph before choosing production batches.

**Parity is a milestone, not just a final-output comparison.**

| Scope | Required check |
|---|---|
| Go engine | Existing oracle/rules tests, apply/undo, exact history, scoring and batch ordering |
| Sparse operations | Hand-computed tiny directed graphs; self-loops, zero-degree and high-degree rows; canonical edge mapping |
| Dynamics | Every state channel at every internal step, including resets and bounded parameter transforms |
| Gradients | Rust versus JAX autodiff for all parameter groups and selected inputs; finite differences away from ReLU kinks on tiny fixtures |
| Full graph | Sampled neuron/edge gradients, head outputs and loss components on identical small batches |
| Training | Same optimizer moments and parameters after one and several identical minibatch updates; loss reduction on a fixed diagnostic task |
| Integration | Teacher/raw/search value distinctions, replay perspective, masks, search requests, dataset splits, checkpoint restore and bounded play |
| TPU later | All relevant numerical checks again, then memory, latency, throughput and multi-host behavior |

Use FP64 tiny fixtures when useful and FP32 as the deployment reference.
Record maximum absolute/relative error and norm error, with justified
scale-aware tolerances. Cross-backend floating-point equality is not promised.
Specify the derivative at ReLU zero. For stochastic variants, parity fixtures
supply identical noise tensors rather than assuming Rust/JAX RNG equivalence;
runtime reproducibility has its own seed/state contract. Matching arithmetic
does not guarantee identical long self-play games near tied actions, so
separate numerical parity from controlled deterministic search tests.

**Search and learning remain independent of neuron physics.**

Keep the existing request/response identities and freeze a network version for
each complete root search. The evaluator returns raw logits and a value; it
does not own search. PUCT supplies its visit target; Gumbel supplies its improved
policy target and selected action. They use the same exact rules and terminal
values. The copied search already has a useful Gumbel reference qualification
against the authors' [Mctx implementation](https://github.com/google-deepmind/mctx).

Use a search result record containing action, improved policy, root value,
prior value, model version, completed simulations, actual neural evaluations,
and elapsed time. Preserve the distinction between simulations and the separate
root inference. Put tree reuse, neural caching and speculative evaluation
behind later search experiments; they are not requirements for the first
learning loop.

The initial offline loss is teacher-policy cross entropy plus MSE to the
strong teacher's raw neural value, with optional separately weighted teacher
ownership/score targets. Keep raw policy, search policy, raw value, search
value and actual game outcomes distinct. The chosen teacher labels both sides;
an opponent's played move is not automatically an expert target. Valid capped
game prefixes may have teacher labels while their terminal-label masks stay
false. [DATASET.md](DATASET.md) defines the labels and perspective conversion.

Architecture studies start on frozen 9×9 data, with identical sampling and
target contracts. Compare validation policy KL/value error and actual matches
against several KataGo strengths, including held-out opponents and openings.
Value-stratified and endgame metrics help expose misleading aggregate losses.
Use equal tuning budgets and repeated seeds. Test both matched-data and
matched-wall-time performance; validation loss alone does not establish play.

After offline selection, use the learned fly prior with native PUCT/Gumbel.
New targets can be relabeled by the teacher or generated through fly self-play.
An online value objective may use completed-game outcomes and declared
bootstrapped targets. Introduce this as an explicit training stage; do not
silently mix behavior-game outcomes into the offline teacher-value task.
Ownership/score auxiliaries have a Go-learning precedent in
[KataGo's paper](https://arxiv.org/abs/1902.10565). Keep win value and score
utility separate.

K, fly search budget, expert game-generation budget and teacher-labeling budget
are four separate controls. A searched fly move can require approximately
K(S+1) graph steps, whereas an offline learner uses no search inside an update.
Precomputing labels amortizes expensive expert computation across architecture
trials. Measure generation, labeling, learning and evaluation costs separately.

Small-board fixtures can qualify Go behavior, but the specialist teacher and
first corpus are 9×9. Do not apply a 9×9-only teacher to 5×5 tests. Keep the full
fly graph throughout. Start with K=8, four history states, FP32 and measured
CPU batches; these are initial hypotheses, not tuned settings. Fly evaluation
can begin with 8/16 simulations alongside prior-only play. A future 19×19
study needs a separately qualified teacher, corpus and task evaluation.

**CPU resources now, optional TPU resources later.**

The current host reports 120 physical CPU cores / 240 logical CPUs and two
NUMA nodes. Read-only inspection on 2026-09-13 found 240 available logical CPUs,
about 400 GiB RAM and about 200 GiB of `/dev/shm` capacity on each of the four
hosts. Recheck available capacity and NUMA placement at launch, preserving the
resources needed by existing workloads. Profiles assign physical cores, memory
and bounded thread pools; they do not leave whole hosts idle around a
coordinator. Large RAM supports resident models and independent learner trials.

Use `/dev/shm/flygo` for all runtime storage, including authoritative corpus
shards and checkpoints. The mount is host-local tmpfs: its contents consume
the same physical RAM as learner tapes and teacher processes. It is shared
with other tasks, and the four mounts are separate filesystems. Initial
per-host admission limits are 100 GiB for all FlyGo storage combined, 64 GiB
minimum free on the mount, and 96 GiB minimum available system RAM. Account
for in-flight reservations, replicas and peak staging/compute needs; pause
new work when shared headroom is insufficient. The exact storage, retention
and two-host recovery contract is in [DATASET.md](DATASET.md).

Persistent disk is not a prerequisite. Accept RAM volatility explicitly and
qualify recovery from surviving peer copies; simultaneous loss of all copies
requires regeneration. Reuse cached source data after checking hashes. New
artifacts must be independently identifiable rather than permanently tied to
a temporary `~/go` path.

| Runtime | Initial placement |
|---|---|
| `cpu-local` | Rust inference/training and small data qualification jobs on explicit CPU sets |
| `cpu-cluster` | Generation/labeling across all hosts first; then independent learner trials, continued data work and evaluation |
| `jax-cpu` | Small qualification jobs using the JAX backend on CPU |
| `tpu-v4` | Future JAX learner and optionally batched JAX inference; Rust remains the Go/search engine |

Qualify each workload locally, then fan out the data producer immediately
after Go qualification. Keep teacher/model processes warm, group work by
checkpoint and NUMA node, and transfer coarse immutable game/label shards.
Use stable IDs, bounded queues, retries and atomic publication. A small SSH
launcher with owned process groups, logs and deadlines is sufficient. Detailed
phase-by-phase allocation and corpus throughput checks are in
[DATASET.md](DATASET.md).

For initial CPU architecture research, independent Rust trials on different
hosts or NUMA groups give useful parallelism without gradient collectives.
Give each trial declared resources and data/compute budgets. When shared-model
learning is a measured bottleneck, qualify distributed learning separately.
For later online training, local actor/evaluator replicas exchange model
versions and game shards; fly state is never sent across SSH every neural step.

The inspected four-host v4-32 resource is one multi-host slice with 16 chips.
For TPU, replicate the modest graph/parameters and shard batches of positions
or games. Synchronize gradients at optimizer steps, not neuron state at every
internal pass. Initialize distributed JAX before device enumeration and use
actual process indices, not hostname suffixes, for data ownership. Keep search
inference local to each host's device group; the existing `local_inference`
work provides a useful pattern for avoiding cross-host barriers inside search.

The initial TPU candidate is an exact degree-bucketed or tiled sparse kernel
with a matching backward pass and canonical edge IDs. JAX's generic sparse
module is a reference rather than a performance foundation; its
[documentation](https://docs.jax.dev/en/latest/jax.experimental.sparse.html)
explicitly discourages performance-critical use. The earlier full-graph
one-chip benchmark measured 54.18 ms for a batch-128 bucketed forward step;
it did not measure full recurrent training or Go throughput. TPU performance
therefore remains a qualification task. CPU parity cannot certify TPU kernel
lowering or BF16 behavior. Recheck those when the device is available.

**Richer dynamics are introduced in a controlled order.**

| Stage | Model extension | Computational consequence and interpretation |
|---|---|---|
| 1 | Learned edge strengths, type-shared leak and bias | Baseline; one sparse multiply per internal step |
| 2 | Neuron-local adaptation or slow synaptic filtering | A small number of extra N×B states and local derivatives |
| 3 | Local pre/post gates and stochastic activity/release | Existing edges only; state scope and noise are explicit |
| 4 | A few compartments or delay classes | More state and possibly multiple sparse operations; measure cost |
| 5 | Fast plasticity and slow consolidation | Separate global learned parameters from episode-local state; avoid unbounded E×B traces |

Type-shared time constants/biases and trainable strengths have a direct
precedent in the [flyvis study](https://www.nature.com/articles/s41586-024-07939-3).
Compartment-specific modulation is motivated by
[Cohn et al.](https://pubmed.ncbi.nlm.nih.gov/26687359/), and richer local
nonlinear computation by [Groschner et al.](https://www.nature.com/articles/s41586-022-04428-3).
These results motivate experiments; they do not specify missing physiology for
every MaleCNS neuron or establish a Go advantage.

A cheap gating form is `g_post ⊙ W(g_pre ⊙ r)`, where each gate depends on the
neuron's own state or signals received through existing edges. A global learned
readout broadcast back into the core would introduce additional communication
and is outside the strict initial contract. Structural rerouting is excluded;
functional routing changes transmission on existing edges. Age/decay can affect
strengths or consolidation without deleting anatomical edges. Annotated
developmental birth class is not a measured synapse-age trajectory.

Default dynamic state lives only within one board evaluation. Carrying activity
or plasticity across actual moves is a later stateful model: define resets,
training truncation, checkpoint state and search branch cloning together.
Sibling branches must not share mutable neural state, hypothetical search
experience must not update the real game's state, and cached predictions must
include the relevant history/model state. Persistent opponent-modeling memory
is a distinct research question from extra computation on the current board.

Each extension lands with equations, parameter/state shapes and lifetimes,
Rust forward/VJP, JAX implementation, parity fixtures, a memory/time measurement
and one controlled learning comparison. Full-graph sparse kernels should remain
reusable when only the local neuron model changes.

**Runs should be easy to start and explain.**

The proposed everyday interface is:

```bash
flygo data generate configs/data-9x9.toml --runtime cpu-cluster
flygo data label configs/data-9x9.toml --runtime cpu-cluster
flygo data freeze configs/data-9x9.toml --version v1
flygo train configs/offline-9x9.toml --runtime cpu-local
flygo sweep configs/ablation-9x9.toml --runtime cpu-cluster
flygo bench configs/offline-9x9.toml --runtime cpu-local
flygo eval /dev/shm/flygo/runs/<run>/checkpoints/<checkpoint> --opponent <opponent-config>
```

Preparation, parity checking and GTP export can be additional focused commands.
The Rust/Python evaluator contract consists of `infer`, `loss_and_grad`,
`train_step` and canonical state import/export; the shared trainer should not
branch on backend internals. The same scientific config selects topology ID,
dynamics version, adapters, K, trainable parameterization, search algorithm,
losses, optimizer, dataset version, target kinds, sampler and seeds. Runtime
config selects backend, batching resources, CPU sets, hosts, device mesh,
`storage_root` and shared storage/memory limits.
Keep global batch/replay semantics explicit
when hardware changes. Reject unknown fields and incompatible combinations.

Each run writes one resolved configuration, source/build and graph identities,
JSONL metrics, game/replay shards, and atomic checkpoints under
`/dev/shm/flygo/runs/<run>`. Checkpoints contain
parameters, optimizer moments and step, RNG states, dataset/sampler position
and identities, and counters. Online runs additionally save replay state and
unfinished actor games/history/targets. Publish only complete checkpoints,
with all required host receipts for a cluster run. Verify a complete checkpoint
copy on a second host before declaring it recoverable from single-host loss.
Atomic rename protects publication from process interruption; it does not make
tmpfs durable across reboot. Retain the last two recovery checkpoints and
explicitly selected/best checkpoints within the shared budget. Do not remove
the last verified pair while publishing its replacement.
Canonical parameter ordering allows Rust/JAX interchange. Exact continuation
is qualified within a declared deterministic runtime; backend or topology
changes are recorded as continuations with different numerical execution.
Preserve data lineage without reproducing the old snapshot/recipe hierarchy.

Use bounded replay and a declared checkpoint retention policy. Measure compile
time, inference latency and throughput, backward/update throughput, active
batch fraction, native Go/search time, checkpoint time and memory separately.
Log neuron activity and gradient coverage plus adapter/core parameter counts.
The principal research comparisons vary one of ports, readout, K, dynamics or
search while retaining the exact fly graph. For K/search comparisons, report
both fixed-data and fixed-wall-time outcomes rather than equating visits with
compute.

**Implementation order and acceptance.**

Detailed deliverables and acceptance criteria are in
[MILESTONES.md](MILESTONES.md); actual status and revision reasons live in
[PROGRESS.md](PROGRESS.md). Data generation and fly computation form two
independent work streams after the Go interface is qualified.

| Milestone | Deliverable |
|---|---|
| M1 | Standalone Go engine/bindings and inherited correctness qualification |
| M2 | Qualified expert/opponent registry, pilot generator and labeler, four-host resource/storage plan |
| M3 | Four-host corpus production and frozen 1M/10M-position releases, with a measured path to 100M+ |
| M4 | Immutable full graph, Rust fly forward pass and JAX CPU parity; proceeds alongside M2/M3 |
| M5 | Complete Rust backward/optimizer, offline learner and portable checkpoints |
| M6 | Frozen-data 9×9 architecture studies, repeated seeds and KataGo strength panels |
| M7 | Optional JAX TPU qualification after CPU parity and hardware availability |
| M8 | Selected fly prior, PUCT/Gumbel improvement, fresh-state labeling and online self-play refinement |

The first learning result should come from a repeatable offline experiment
with valid teacher targets and a stable evaluation interface. Strong Go,
beneficial biological detail, and a TPU speed advantage remain empirical
results to establish.
