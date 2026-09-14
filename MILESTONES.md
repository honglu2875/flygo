FlyGo implementation milestones, revised for offline expert data first.
These are the implementation deliverables and acceptance criteria. M0 covers
design and source inspection; actual completed work and remaining gates are
recorded in [PROGRESS.md](PROGRESS.md).

[DESIGN.md](DESIGN.md) specifies the software/model boundaries;
[DATASET.md](DATASET.md) specifies expert selection, labeling, splits, scale
and architecture comparisons. [PROGRESS.md](PROGRESS.md) records actual status,
next steps and plan revisions. Each milestone produces a usable artifact and
an acceptance record. CPU work and dataset production do not wait for TPU.

After the first M6 batch, the active research sequence is
[B0–B5: biological interfaces](docs/biological-interfaces.md), within the
[G1–G5 neuron-group study](docs/group-study.md). B0's bounded atlas/response
pilot is complete. The coherent spherical input and bounded
[motor-embedding pilot](docs/embedding-study.md) are now CPU-qualified. That
pilot barely changes circuit weights and shows no benefit over a frozen
representation. Signal and optimizer conditioning precede substantial
pretraining; functional output groups remain downstream of trained responses.

| Milestone | Deliverable | Depends on |
|---|---|---|
| M0 | Revised design and data protocol | Complete in this design pass |
| M1 | Standalone Rust Go stack and Python binding | M0 |
| M2 | Teacher/opponent qualification and small corpus pilot | M1 |
| M3 | Four-host production and frozen corpus releases | M2 |
| M4 | Fixed graph, batched Rust fly inference and JAX CPU reference | M1; runs alongside M2/M3 |
| M5 | Complete Rust training and first offline 9×9 learner | M4 and usable M2/M3 data |
| M6 | Controlled offline architecture studies and KataGo matches | M5 and a frozen corpus |
| M7 | Optional JAX TPU inference/training qualification | M5 and TPU availability |
| M8 | Search improvement and online refinement of selected models | M6 |

The critical early sequence is Go qualification → expert pilot → corpus
production. The fly backend is a separate work stream, so expert games can
accumulate while numerical inference/training is implemented. M6 starts on
V0/V1; it does not wait for the largest intended corpus.

**M1 — extract and expose the Go engine.**

Work in the copied `go-core`, `go-search`, `go-actors`, the Go portion of
`flygo-python`, and the workspace/package build files.

- Copy the three crates with tests and attribution notices. Record source
  hashes and compatible pinned Rust/Python dependencies.
- Adapt the existing coarse board, actor and single-game bindings. Preserve
  exact history, observations, legality, score and PUCT/Gumbel requests.
- Define a small public Rust API and Python array contracts. Release the GIL
  during native operations and avoid per-search-node Python callbacks.
- Retain the useful native oracle/scoring and external match qualification
  helpers. FlyGo's runtime does not import the old recipe framework.
- Identify the bootstrap integration with the existing official KataGo CPU
  build and model receipts. Use a small bridge to existing tools for the pilot.

Acceptance: new extension builds; inherited rules/search/actor tests pass;
Rust and Python callers complete equivalent game batches; capture, superko,
passes, perspective and scoring fixtures pass. Run a real 9×9 KataGo game
through the new Go interface, checking moves/boards and final score. Record
native throughput with explicit CPU resources. Once this gate passes, start
M2 immediately; do not wait for fly inference or learning.

**M2 — qualify teacher data and measure its production cost.**

Work in `python/flygo/data/{katago,generate,label,corpus}.py`, the data config,
small data tests and the minimal CPU launcher. Bootstrap from `~/go`'s GTP,
match scheduling, model records, history replay and dataset audits.

- Refresh the four-host inventory: available physical cores, NUMA layout,
  memory, `/dev/shm` capacity and current workloads. Put all runtime artifacts
  under `/dev/shm/flygo`; initial host-wide limits are 100 GiB of FlyGo files,
  64 GiB free on the mount and 96 GiB available system RAM.
- Implement shared admission/reservations in `storage.py`, accounting for peak
  writes, replicas, caches and process heaps together. Check pressure handling
  with small configured limits and concurrent writers. Qualify session/logout
  survival with a small owned sentinel under the intended job supervisor;
  restrict cleanup to FlyGo's own expendable files.
- Qualify the cached 9×9 specialist and a small current-strong shortlist at
  the intended rules/budgets; freeze the chosen teacher and 8–16 historical
  opponents in calibrated strength bands. Do not infer 9×9 rank from global
  network ratings or chronological checkpoint order alone.
- Specify teacher color randomization, per-game opponent sampling, opening
  exploration, resignation/cap behavior, seeds and budgets. Check both colors.
- Generate complete histories. Independently label all valid pre-action
  positions with the strongest selected teacher, including opponent turns.
- Preserve played actions, raw/search policies, raw/search values and behavior
  outcomes as distinct fields. Qualify White-to-current-player conversion,
  pass/masks, root-edge count mapping and missing-label semantics.
- Define stable game/position IDs, immutable shard publication, label identities
  and train/validation/test assignment by game/opening family.
- Benchmark long-lived CPU teacher processes, concurrent queries, model queues,
  NUMA placement, neural batching and thread counts. Measure generation and
  each label kind separately. Scale the qualified pilot across all four hosts.

Acceptance: about 10k labeled positions with valid game histories, no silent
rule substitutions, correct teacher identity on both turns, normalized legal
policy targets, finite values and explicit truncation masks. Compare raw value
conversion against the raw neural interface. Verify label order and causal
prefixes, plus a restart/retry test with no duplicate published rows. Verify
that storage admission, reservation release and pressure handling preserve
the configured headroom, and record session/logout survival results.

Deliver a short teacher/opponent selection record, label-schema fixtures,
per-host resource profiles, actual bytes per row, labeled positions/s and a
production-time/storage estimate. A small teacher selection panel identifies
our qualified choice; it does not prove a global strongest-model ranking.

**M3 — produce and freeze the shared expert corpus.**

Work in the producer/labeler queues, corpus manifests/cache loader and runtime
profiles. No fly model is required.

- Fan out generation and labeling across all available CPU allocations on all
  four hosts. Keep model processes resident and match production to labeling
  throughput. The coordinator occupies a small allocation, not an idle host.
- Implement bounded queues, worker leases, stable shard IDs, atomic publication,
  failure logs and deduplicated retries. Keep teacher/model/query hashes.
- Freeze an initial V0 of roughly 1M positions, then V1 around 10M. Pursue
  100M+ only with measured throughput, storage capacity and useful coverage.
  Count complete games and unique states separately from position exposures.
- Store compact histories, labels and manifests in `/dev/shm/flygo`, with two
  verified copies of frozen shards on distinct hosts. Build bounded typed
  training caches; reuse labels for identical qualified requests. Include
  copies and peak staging in the shared budget. Mark replication status.
- Cover opponent bands, expert colors, phases, near-even positions and endgames.
  Track easy won-state saturation and duplicate openings.
- Freeze game-family splits and audit actual student-input overlap. Provide
  natural-distribution, balanced and novel-state validation indexes; keep the
  headline test set closed during architecture selection.
- Preserve a cheap raw-policy/value corpus and a clearly identified search-
  policy subset/release. Every selected search-policy training row needs its
  specified target; do not substitute raw labels silently.

Acceptance: all published shard hashes verify; full histories replay under
native rules; sampling is reproducible; color/strength/phase coverage and label
completeness are reported. A worker restart preserves published data without
repeated rows. Demonstrate recovery from a surviving peer using only owned
fixture files, and pressure-driven cache eviction without deleting active
corpus data or another task's files. Freeze manifests so a growing producer
cannot change an active experiment. Record sustained accepted positions/s,
CPU core-hours, process RAM, `/dev/shm` footprint/headroom and the largest
measured bottleneck. A usable V0 enables M5 while V1 expands.

**M4 — fixed graph and Rust/JAX forward execution.**

Work in `fly-core/src/{graph,sparse,recurrent,model}.rs`, Python port
initialization, matching JAX functions and the thin bindings. This work runs
alongside corpus production using explicitly allocated resources.

- Reuse the pinned MaleCNS inputs after checksum checks. Publish 165,122 node
  IDs and 15,270,273 canonical edges, counts, weight prior, type/NT annotations
  and missingness. Use a JSON manifest plus typed `.npy` arrays.
- Construct forward/transpose layouts with canonical edge mappings. Optimizer
  parameters never include topology. Reordering or smaller Go boards does not
  remove neurons or connections.
- Define named parameters, transforms, initialization, state shapes and scopes.
  Implement type-shared leak/bias and independently learnable edge magnitudes.
- Implement destination-row sparse multiplication, bounded thread pools,
  contiguous batched state, sensory injection, K-step rate dynamics, sparse
  readout and policy/value heads.
- Implement the same forward rules on JAX CPU, loading identical arrays.
  Provide per-step traces for focused diagnostics and a common `infer` API.

Acceptance: graph identities/layouts reproduce; tiny directed fixtures and
full-graph forward outputs agree across Rust/JAX; batch/single and padding
semantics agree. Verify all K states, reset behavior, disjoint ports and
signal coverage through the actual graph. Benchmark batches 1/8/32/128 and
K=2/8 with bounded CPU/NUMA allocations, separating setup from warm execution.
Do not use the 9×9 specialist's labels on smaller-board numerical fixtures.

**M5 — Rust backward/optimizer and a real offline learner.**

Work in sparse/dynamics derivatives, recurrent tapes, full-model backward,
loss/optimizer, `jax/model.py`, `train.py`, corpus loading and checkpoints.

- Implement transposed state gradients and per-edge batch dot products without
  E×B message materialization or an N×N gradient. Differentiate K passes,
  constrained parameters, input mapping, pooling and every trained head.
- Implement masked teacher-policy cross entropy and MSE to raw teacher value,
  followed by optional teacher auxiliaries. Labels expose their own masks and
  kinds. Game-outcome targets are not silently mixed into value training.
- Implement one optimizer with identical Rust/JAX equations, parameter groups,
  clipping and update order. Declare any recurrence stability constraints.
- Provide `loss_and_grad` and `train_step` with named arrays. Rust training has
  no JAX runtime dependency; JAX serves as an independent CPU derivative check.
- Implement frozen-corpus sampling, D4 transforms, validation and portable state
  containing parameters, optimizer, dataset ID, sampler/RNG state and counters.
  Publish checkpoints within the shared RAM budget, verify peer copies and
  retain the last two recovery checkpoints plus explicitly selected/best ones.
- Add recurrence checkpointing or optimized tapes only if the measured memory
  profile warrants them, and compare gradients before changing the default.

Acceptance: finite differences on suitable tiny cases, all-group Rust/JAX
checks, selected full-graph gradients and matching one/several optimizer
updates. A full-graph diagnostic task learns. Then train on actual 9×9 V0
labels, measure held-out losses, restore in a fresh process and reproduce the
next deterministic update, including restore from a verified peer copy. Export
a model that plays legal 9×9 games via GTP.
Deliver forward/backward/update memory and throughput, including loader costs.

**M6 — controlled architecture ablations and strength evaluation.**

Work in scientific configs, `scripts/deploy_research.py`, named dynamics/port
variants, focused parity tests, validation reports and KataGo match panels.
Use all available CPU allocations for independent trials, continuing data work
and evaluation; share immutable graph/data caches where practical. Bounded
preliminary screens can use the 10k pilot and the intermediate 100k release;
they do not replace the V0 confirmation gate or select a final architecture.

- Establish raw-policy/value and search-policy/raw-value controls as distinct
  target contracts. Freeze dataset/split, sampling, augmentation and budgets
  within each comparison.
- Screen K=2/4/8/16; sensory attachment; pooling/readout; parameter sharing and
  sign constraints; then one adaptation, filtering or gating extension.
- Give variants equal small tuning budgets. Track core/adapter parameters,
  neuron/gradient coverage, memory and training/inference throughput.
- Match the conventional neural control on prediction FLOPs and training
  exposures. Declare how input-dependent zero skipping is counted, report
  achieved mismatch, and keep padding/latency and total project compute separate.
- Compare held-out policy KL and value MSE by phase, value band and opponent
  band, plus pass/endgame/tactical slices and the novel-state subset.
- Run periodic fresh KataGo panels for screened candidates. Test prior-only
  play and common PUCT/Gumbel budgets, followed by fixed move-time comparisons.
  Include held-out opponent checkpoints and paired fresh opening colors.
- Replicate promising results with at least three seeds and larger panels.
  Keep final test positions and openings separate from tuning decisions.

Acceptance: every claimed improvement has matched data/target provenance,
measured resource use and uncertainty. Primary algorithmic-efficiency claims
require matched prediction arithmetic and exposure horizons; report fixed-time
results separately. Lower validation loss alone is not a playing-strength result. Freeze
a selected architecture/checkpoint before online refinement, while continuing
useful larger-data confirmations.

Every new neuron variant first specifies equations, parameter/state shapes,
initialization, state lifetime and derivative rules. It then gets Rust
forward/VJP, a matching JAX step, focused parity, a full-graph cost measurement
and a controlled learning comparison. Do not bundle several biological changes
into one uninterpretable experiment or copy the whole trainer for each variant.

**M7 — optional TPU qualification.**

Work in JAX sparse kernels, recurrence/training, TPU resource profiles and the
existing parity suite. CPU corpus production and experiments continue while
the TPU is occupied.

- Start on an available chip with FP32 and the canonical graph. Qualify sparse
  forward, transpose, edge gradients and the complete K-step model/update.
- Inspect actual compiled temporary memory and full update throughput. Optimize
  exact buckets/tiling and tape storage without changing graph semantics.
- Qualify BF16 separately, with FP32 accumulation/state where needed. Replicate
  the modest graph/parameters and shard examples across the 16-chip slice.
- Verify actual process-rank ownership, padding, gradients and checkpoints;
  qualify local inference before multi-host learner collectives.
- Prefer the qualified fly learner as the initial TPU workload. If expert
  labeling dominates, a separate teacher port must load the selected exact
  checkpoint and match full features/policy/value before producing labels.
  The existing policy-only JAX work is not sufficient evidence of that parity.

Acceptance: complete numerical qualification and actual end-to-end memory/time
measurements. CPU parity does not qualify TPU lowering. Record backend changes
as numerical-runtime changes, without promising bitwise game trajectories.
Choose TPU workloads by measured benefit to labeled data or experimental
progress, not device occupancy alone.

**M8 — search improvement and online refinement.**

Work in online replay sourcing and actor scheduling around the same learner,
not a second research framework.

- Start from the selected offline fly prior and use native PUCT/Gumbel to
  improve moves and generate policy targets. Pin one model per root search.
- Collect states reached by the student, including its errors. First consider
  relabeling them with the same teacher to study distribution shift explicitly.
- Add self-play outcomes, ownership and declared bootstrapped targets as a
  separate online objective. Specify any replay/teacher mixing or annealing.
- Preserve full actor history/RNG/unfinished targets and replay alongside
  learner state in atomic, complete checkpoints.
- Measure online gains against the frozen offline checkpoint at matched search
  and time budgets, with teacher/online compute costs reported separately.

Acceptance: correct policy/value perspective at all leaves, explicit target
lineage, restart qualification and a fresh strength comparison. Capped games
have no fabricated outcomes. A stronger result must survive the predeclared
panel rather than just lower training loss.

Memory across real moves, fast plasticity, compartments and delays remain
explicit model extensions. Before persistent state is used with search, define
branch-local cloning, causal updates, sequence replay and training truncation.
Before per-edge fast plasticity, budget E×B state. Every extension retains the
full fixed topology and the Rust/JAX qualification discipline.
