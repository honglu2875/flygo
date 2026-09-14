# Qualification history through 2026-09-13

Archived progress ledger from commit bf251ce. This is historical evidence;
see [current progress](../PROGRESS.md) for active jobs and decisions.

# Progress

Research status last checked **2026-09-13 20:07 UTC**. Eight new baseline/variant trials are training
on a cloned 1.48M-position snapshot. Generation continues on all four hosts.
The earlier nine 100k-data trials and their evaluation panels are complete.
See [README.md](../README.md) for commands and [MILESTONES.md](../MILESTONES.md) for gates.

**Model guide added 2026-09-13 21:32 UTC.** The standalone
[interactive HTML guide](model.html) explains the fixed graph, sensory
attachment, pooling/heads, recurrent equation, current 2×2 study, proposed
extensions, offline distillation/search, and CPU/TPU costs. It embeds an actual
64×64 connectivity summary and a validation-position prediction from the
K=4/G=656/seed-1 step-2,000 checkpoint, with source IDs and hashes. Illustrative
circuits and proposed mechanisms are labeled separately. Chromium checks
passed for all controls, offline operation, reduced motion and widths from
320 to 1,440 pixels; figures were visually reviewed. Evidence is in
`/dev/shm/flygo/runs/model-guide/render-checks.json`; a
[preview image](model-preview.png) is also available.

| Milestone | Status | Evidence / remaining gate |
|---|---|---|
| M0 — design | Complete | [DESIGN.md](../DESIGN.md), [DATASET.md](../DATASET.md), explicit fixed-topology contract |
| M1 — Go stack | Complete | Source hashes, native oracle, Rust/Python parity and actual KataGo game |
| M2 — expert pilot | Complete | Audited pilot, raw-value/prefix checks, 8-model calibration, storage/session and live-restart qualification |
| M3 — corpus production | Running | Four-host production, balanced 100k and cloned 1.48M research releases, shared feature cache and peer recovery; balanced 1M V0 and cache-eviction qualification remain |
| M4 — fly forward | CPU baseline complete | Full graph identity and Rust/JAX parity; batching/reset/coverage and measured costs |
| M5 — CPU learning | Engine qualified; data-scale gate pending | Full backward/Adam, synthetic learning, bitwise peer restore, real offline training/GTP; V0 confirmation remains |
| M6 — architecture studies | Larger snapshot screen running | Eight K/readout trials on 1.48M positions; earlier screens complete. Additional seeds, matched-time confirmation and architecture selection remain |
| M7 — optional TPU | Waiting | TPU untouched; CPU parity does not qualify TPU kernels or memory |
| M8 — online refinement | Not started | Offline prior/search evaluation works; online target generation/distillation remains later |

**Live work and resources**

- `expert-v1`: 32 resident workers, 16 concurrent games each. Generation uses
  64 pinned physical cores/host (`0–31,60–91`), 256 of 480 physical cores total.
  Research excludes their SMT siblings. Every live KataGo thread passed affinity
  checks after restart; no worker failures in the current session.
- `prototype-grid-v1`: **eight live trials**, two per host, 24 physical cores
  each. This uses 48 research cores/host (`32–55,92–115`) alongside the 64
  generation cores; eight physical cores/host remain outside those allocations.
  All 200 training threads passed affinity checks. Every trial has performed
  optimizer updates, and initial checkpoints have verified peer copies.
- [configs/prototype-v1.json](../configs/prototype-v1.json) crosses K=4/8 with
  656/2,624 readout groups, seeds 1/2, 10,000 updates × batch 32. All trials use
  the same immutable CPU-native environment, dataset, optimizer and sampling
  contract. Initial parameters/optimizer/ports were bitwise identical within
  each depth pair; width variants preserve sensory maps and subdivide pools.
- The 16-position real-training-data fitting diagnostic **passed at 200
  updates** and released its three cores. KL fell from 2.1021 to 0.09996 and
  value MSE from 1.1512 to 0.00532, meeting the predeclared KL<0.1/MSE<0.05 gate.
  15,269,858 of 15,270,273 edge parameters changed. This demonstrates fitting
  actual targets; generalization and playing strength remain separate checks.
- The nine earlier final checkpoints have verified copies on two hosts;
  their finished study coordinators are stopped. The new study has its own
  active checkpoint coordinator on host 0.
- At 20:06 UTC: **18,753 complete games / 1,776,357 labeled positions** in the
  growing corpus; all 32 workers healthy. A measured 22.7-minute interval produced
  175.96 accepted positions/s, approximately 404 allocated core-hours per raw
  million positions. Opponent 7 limits the balanced release.
- The `v0-1m` freezer remains running, pinned to four spare cores. At 20:05 it
  could select 871,097 balanced positions; it waits for enough complete games
  in every opponent/color stratum before audit, publication and peer copying.
  Raw production above one million is not a frozen million-position release.
- Last resource check with the new trials: 179–191 GiB free shared memory and 346–360 GiB available
  RAM/host. Admission retains a 100 GiB own-file cap, 64 GiB free-shm floor and
  96 GiB available-RAM floor across all FlyGo jobs and staging.
- `python3 -B scripts/cluster.py status --run-id expert-v1` gives concise status;
  add `--json` for full records. `... stop` drains generation. The freezer status
  is `/dev/shm/flygo/runs/freeze-v0-1m/status.json`; its sibling `stop` file
  requests a stop. Owned SSH keepalive sessions remain open for RAM lifetime.

**Frozen data**

| Release | Complete games | Positions | Train / validation / test | Audited compressed size |
|---|---:|---:|---|---:|
| `pilot-v1` | 112 | 11,242 | 10,249 / 811 / 182 | 2.66 MB |
| `screen-v1` | 1,024 | 100,417 | 89,633 / 6,274 / 4,510 | 24.15 MB |
| `prototype-20260913` | 15,632 | 1,481,078 | 1,322,078 / 98,256 / 60,744 | 361.25 MB |

The first two releases balance complete games across opponents and expert
colors. The prototype release independently clones all completed games written
before 19:38:37 UTC, preserving original opening-family splits. Its actual
production mix is unequal (969 games against opponent 7, 3,137 against opponent
6); it does not replace balanced V0. Every release has zero truncated games and
checksum-verified peer copies. The prototype and its **6.40 GB read-only feature
cache** are deployed to all four hosts. **87,777 validation positions** have no
D4-equivalent full input in training. About 77.5% of prototype values are
saturated (`abs(value)>0.98`), so phase/value-band metrics matter. Final test
labels remain outside architecture selection.

**Qualified results**

- All 21 copied Go-crate files retain their source hashes. The extended native
  oracle checked 2,970,115 legal transitions / 8,308,582 proposals. A real
  KataGo game matched every board and final score. Rules-only throughput was
  7.68M moves/s on eight cores; neural/search costs are separate.
- **44 regular Rust tests and 33 Python tests passed** in the new native build.
  Cache/direct loading and augmented batches agree exactly; corrupted caches
  and reassigned family splits are rejected. Existing-thread affinity,
  interrupted peer receipts and batched PUCT/Gumbel have regression coverage.
  Finite differences, all parameter-group JAX gradients and three Adam updates
  are covered. Changed Rust crates pass strict Clippy without modifying imports.
- All 165,122 nodes and 15,270,273 edges reproduce the original source tables.
  Full-graph Rust/JAX maximum state error was `8.35e-7`, edge-gradient error
  `1.92e-10`. Batch/single and reset checks are bitwise. All 972 inputs attach.
- The full-graph synthetic diagnostic passed at 200 updates (CE 0.116, MSE
  0.0358). Its first 100-update attempt missed the original MSE<0.1 gate;
  that failed evidence is retained.
- Fresh local and peer checkpoint restore reproduced the next batch, update
  and all 28 numerical state arrays bitwise. A deleted disposable corpus
  fixture was recovered from peer RAM and fully replay-audited.
- The 4/8/16-game generation benchmark reached 3.97/4.68/5.50 positions/s on
  the same eight cores. The drained fleet restart preserved all 4,441 games
  present at its start byte-for-byte. The benchmark used one intermediate
  opponent; fleet measurements and the slowest stratum determine production ETA.
- Cached inference preserved full-graph outputs bitwise. A separate CPU-native
  build preserved three full B=32/K=8 updates bitwise and improved their time
  by about 10%. On 24 cores, native K=8/B=32 inference reached 82.26 positions/s
  and training took 2.26–2.35 s/update. The mixed B=1..128 benchmark peaked at
  1.72 GiB RSS, excluding the 100k loader (about 0.4 GiB).
- The K=4/8 controls used the generic build; the wide-readout trials used the
  qualified native build. Wider K=4 forward and three full B=32 updates were
  also bitwise identical across builds. Observed study medians were 1.29–1.32
  s/update for K=4, 2.42–2.57 for K=8, and 1.19–1.21 for native wider K=4.
  These are matched-example studies with measured cost, not equal-time curves.
- Corpus replay audit now decompresses each bounded game record once. The same
  100-game audit improved from 7.23 to 0.76 s; all 100,417 frozen positions
  passed the optimized full audit in 7.82 s without weakening checks.

Pilot, 500 updates, one matched seed, **all 811 validation positions / 9 games**:

| Internal passes | Policy KL | Value MSE | Prior-only wins vs weakest KataGo |
|---:|---:|---:|---:|
| 2 | 1.8832 | 0.7889 | 0/16 |
| 4 | 1.8613 | 0.6707 | 0/16 |
| 8 | 1.8484 | 0.6163 | 0/16 |
| 16 | 1.8540 | 0.5490 | 0/16 |

These tiny validation/game panels do not establish an architecture winner.
K=8 also scored 0/16 with 16-simulation PUCT and 0/16 with Gumbel.
Earlier training logs used a fixed 512-position validation slice and therefore
have different numbers. The full comparison includes phase, value/opponent
bands, D4-novel inputs and complete-game bootstrap intervals.

100k screen, **three matched seeds, 1,000 updates / 32,000 examples per trial**,
all **6,274 validation positions / 71 games** (5,727 D4-novel positions):

| Model | Policy KL, mean ± seed SD | Value MSE, mean ± seed SD | Prior-only wins |
|---|---:|---:|---:|
| K=4, 656 readout groups | 1.6918 ± 0.0034 | 0.6110 ± 0.0488 | 0/48 |
| K=8, 656 readout groups | 1.6919 ± 0.0026 | 0.6140 ± 0.0616 | 0/48 |
| K=4, 2,624 readout groups | 1.6769 ± 0.0060 | 0.6040 ± 0.0160 | 0/48 |

SD is sample standard deviation across seeds, not a confidence interval. All
144 fresh prior games completed against the weakest ladder checkpoint at
16 KataGo visits. Wider pooling improved policy KL in all three paired seeds
(0.88% mean), with no demonstrated strength gain. K=8 offered no clear benefit
at this budget. The wider seed-1 model also lost all 16 PUCT and all 16 Gumbel
games at 16 simulations; these panels took 169 and 232 seconds respectively.
No architecture has been selected, and final test labels remain unused.

New launch evidence is in `/dev/shm/flygo/runs/prototype-v1/launch-qualification.json`;
the plan, job records, logs and replica status are under `runs/prototype-grid-v1`.
The initial launcher failures were fixed, and three live trials were adopted
without restarting; original failure logs are retained. Earlier evidence is
under `/dev/shm/flygo/runs/{m1,m2,m3,m4,m5,m6,screen-100k,readout-2624}` and
`/dev/shm/flygo/releases/`. These files are volatile. Persistent SSH sessions
qualified control-session disconnections; final logout/reboot/common cleanup
can still lose RAM files, including replicas.

**Next gates**

Inspect the longer prototype learning curves following the successful fitting
diagnostic. Periodic reports use the same fixed random 2,048-position train,
validation and D4-novel slices; full validation and fresh KataGo panels follow
the training screen. These runs each expose 320k sampled examples, so further
training may still be necessary. Promising results need a third seed, matched
wall-time comparisons and balanced-data confirmation. The balanced V0 freezer
continues independently. TPU and online distillation remain separate gates.

**Plan revisions.**

| Date | Change and reason | Effect |
|---|---|---|
| 2026-09-13 | Record the agreed baseline: fixed topology, learnable strengths, Rust CPU learning with JAX parity, expert corpus before architecture studies | M2/M3 data production can proceed alongside M4; online refinement remains M8 |
| 2026-09-13 | Use `/dev/shm` for data, checkpoints and temporary files, as requested | Removes the persistent-disk prerequisite; M2 qualifies shared budgets and session survival, M3/M5 qualify peer recovery |
| 2026-09-13 | Add this progress ledger, as requested | Milestone status and plan changes stay visible without reading implementation details |
| 2026-09-13 | Begin implementation as authorized; add shared-budget foundations alongside the M1 build helper | Protects initial RAM use and starts M2; topology, dataset protocol and acceptance criteria are unchanged |
| 2026-09-13 | User authorized 64 physical cores/host and autonomous work while away | Pin 32 workers to 256 physical cores total; reserve the remaining 224 physical cores for research, without changing other jobs |
| 2026-09-13 | Scale the qualified label pipeline across four hosts while finishing ladder calibration | Pilot freezes balance opponent/color strata; chronological/global ratings are not treated as calibrated 9×9 strength; no change to target semantics |
| 2026-09-13 | Retain one immutable compressed record per complete game during the pilot | Simplifies audit/retry and measures actual size; larger typed training caches and release manifests can be added without changing existing records |
| 2026-09-13 | Direct worker-to-peer SSH trust is absent; use verified checkpoint pulls from host 0 | Resume K=2/4/16 from intact step-100 files without losing updates; no global SSH or credential changes |
| 2026-09-13 | Tighten novelty auditing to include D4-equivalent full model inputs, matching training augmentation | Training/splits/targets unchanged; initial K=8 `novel` metrics used exact orientation, so compare runs using natural validation and recompute a common D4 novelty slice |
| 2026-09-13 | Eight-core concurrency benchmark: 4/8/16 games gave 3.97/4.68/5.50 positions/s; apply 16 through a drained restart | Same 64 physical cores/host, model/label contract and immutable game IDs; short intermediate-opponent benchmark requires fleet follow-up |
| 2026-09-13 | Add a frozen 100k screening release between the pilot and 1M V0 | Enables larger-data learning while the slowest opponent stratum accumulates; does not substitute for V0 acceptance |
| 2026-09-13 | Screen K=4/8 on the 100k release with three matched seeds and 1,000 updates | Both completed pilot runs improve held-out predictions at practical CPU cost; K=16 pilot still finishes separately. This bounded next experiment does not declare an architecture winner |
| 2026-09-13 | After the K=4 controls finish, compare 2,624 readout pools against 656 at K=4, three matched seeds and 1,000 updates | Tests pooling width without changing topology, targets or sampling; task adapters remain below one million parameters. First verify the wider configuration and CPU-build parity |
| 2026-09-13 | Queue automatic `v0-1m` publication once all opponent/color strata suffice | Total generated rows reach one million earlier than a balanced release; keep the original balance requirement and wait for the slowest stratum |
| 2026-09-13 | User requested cloning current data and starting baseline/variant research now | Freeze all completed games as a separate production-distribution release; preserve family splits and closed test labels. Start a bounded depth × readout screen without waiting for balanced V0; record its unequal opponent mix and require later confirmation |

**Revision discipline**

Record the reason, affected milestone and effects on existing data/checkpoints
before dependent work. Version scientific contracts; preserve earlier results.
Recheck only affected behavior, and mark a milestone complete only when its
acceptance criteria have evidence.

## Archived ledger — 2026-09-14 04:25 UTC

The following snapshot preserves detailed evidence while the active progress file is shortened. Relative links retain their original repository-root context.

# Progress

Updated **2026-09-14 04:25 UTC**. Autonomous work is authorized until about
09:02 UTC. The target is an efficient fixed-fly Go engine, compared with a
conventional neural engine at matched prediction FLOPs and training-position
exposure. No fly strength advantage has been established.

[README](../README.md) describes interfaces; [RESEARCH](../RESEARCH.md) records study
contracts and biological hypotheses; [MILESTONES](../MILESTONES.md) defines gates.
Earlier qualification detail remains in [history](qualification-history.md).

| Milestone | Status | Evidence / remaining work |
|---|---|---|
| M0–M2 — design, Go, expert pilot | Complete | Rust/Python rules/search parity, real KataGo games, teacher/history/storage qualification |
| M3 — corpus | V0 complete; production continues | Balanced million-position release and cache verified on all four hosts |
| M4 — fly forward | Complete | All 165,122 nodes / 15,270,273 edges; CPU and actual TPU parity |
| M5 — CPU learning | Complete | V0 training, bitwise fresh local/peer recovery and exported legal Go games |
| M6 — studies | Active | Three-seed matched CNN confirmation complete; optimizer and adapter refinements running |
| M7 — TPU | Qualified and in use | Four-host SPMD; full-model updates, matching copies and bitwise full-trainer replay |
| M8 — online refinement | Pending model selection | Offline prior/PUCT/Gumbel works; stronger prior and controlled comparison first |

## Live work

- **Generation:** 32 workers × 16 concurrent games, using 64 pinned physical
  cores per host (`0–31,60–91`). Production and frozen research releases are independent.
- **Prototype:** K=4/8 × G=656/2,624 × seeds 1/2, 10,000 updates at B=32,
  rate 0.003. All eight training runs, full validations and match panels are complete.
  Paired depth/readout reports retain every seed and count search evaluations.
- **CPU optimizer:** all six global rates and two bias-rate variants completed
  full V0 validation. `optimizer-confirm-v1.json` registers 4,000 updates at
  B=32, three seeds, comparing 0.003, 0.01, and 0.03 with bias multiplier 0.01.
  Wave 1 and its full validation finished; wave 2 runs on all four `32–55` lanes.
  A bounded coordinator waits for validation before launching wave 3.
  All waves reuse source `2185746d8cb5a7b93d1c`; original trials keep their binaries.
- **TPU:** the equal three-rate `matched-control-v1` screen completed. Its
  registered KL+MSE rule selects fly rate 0.01 and CNN rate 0.003.
  `matched-confirm-v1` completed three seeds per family at 512 updates × B=2,048
  (1,048,576 exposures each). Full validation and common 32-game
  prior/PUCT/Gumbel panels are complete. The registered `spatial-input-v1`
  adopted all three fly baselines and is training spatial/shuffled ports at
  identical settings/horizons; spatial seed 1 finished and seed 2 runs.
  All six full-validation/match followups are queued on `32–55` after optimizer
  validation releases those lanes. One cohort owns all 16 devices.
- **Schedule screen:** four seed-1 warmup/decay cases are registered at 128,000
  exposures. All four started after their matched panels left `92–115`.
  They use source `c149278eda30697520f9` and queued full validation.
- **Visual readout:** a separate input-fixed, visual-output spatial/shuffled
  screen is queued after schedule validation. Both variants pass full-state,
  all-gradient and three-update JAX CPU parity on real V0 inputs. They retain
  23,565 annotated output cells, exclude 155 outside the sensory bounds, and
  cover 60 board points. The full recurrent topology still executes.
  Source `c149278eda30697520f9`, B=32, 4,000 updates, seed 1; no learning result yet.

Runtime artifacts stay under `/dev/shm/flygo`. Floors remain **64 GiB free
shared memory**, **96 GiB available RAM**, and a **100 GiB own-file cap** with
reservations. At 04:19, free shared memory was 137–177 GiB and available RAM
308–351 GiB. All 32 generation workers are healthy; 73,296 games are published. Generation uses 256 physical cores across the fleet; CPU trials
and evaluations use up to another 192. TPU runtime/development uses spare cores;
waiting coordinators use core 116. Owned SSH keepalives remain active. RAM and
peer copies remain volatile across reboot or common cleanup.

## Frozen data and numerical gates

| Release | Complete games | Positions | Train / validation / test |
|---|---:|---:|---|
| pilot-v1 | 112 | 11,242 | 10,249 / 811 / 182 |
| screen-v1 | 1,024 | 100,417 | 89,633 / 6,274 / 4,510 |
| prototype-20260913 | 15,632 | 1,481,078 | 1,322,078 / 98,256 / 60,744 |
| v0-1m | 10,256 | 1,000,201 | 887,338 / 70,425 / 42,438 |

V0 contains 840,659 unique board/turn states, 63,354 D4-novel validation
inputs and 79.1% saturated teacher values. Its compressed release is 241.1 MB.
All releases exclude truncated games; final test labels remain outside tuning.
Registered unused v2 feature caches are evictable, while live mmap readers,
NumPy views, corpus, checkpoints and legacy caches remain protected.

- **Regression:** 47 Rust tests and all 57 current Python tests pass. Includes
  sparse VJPs, recurrence gradients, scaled Adam, port controls, CNN equations,
  storage, cache leases and portable checkpoints.
- **V0 CPU:** `v0-cpu-recovery-v1` reproduces the next update bitwise in fresh
  local and peer processes (all 28 model/optimizer arrays). A trained V0
  checkpoint completes two legal games against KataGo; this is an export gate.
- **TPU:** `tpu-real-v0-v2` passes full states, all gradients and three updates.
  `tpu-v0-restore-v1` passes fresh TPU continuation and Rust portability.
  `tpu-v0-trainer-restore-v1` replays 64 real learner updates, reproducing all
  31 parameter/moment/step/port arrays and the sampler **bitwise**.
- **Variants/control:** full spatial-port plus scaled-bias TPU parity passes.
  CNN parity passes three transitions from identical checkpoint states; two
  failed free-running attempts remain recorded because near-cancelled Adam
  updates can cross ReLU boundaries. This is not a bitwise cross-backend claim.
- **TPU execution:** explicit high-accuracy math fixes the original softplus
  discrepancy without relaxing its tolerance. Exact degree buckets and a
  custom transpose/edge VJP avoid the reference kernel's 52.78 GiB allocation.
  B=2,048 gives 754 inference / 577 training positions/s; B=16,384 gives
  3,038 / 1,491, with 2.10 / 15.00 GB compiled training temporaries per chip.
  Larger tiles were slower. End-to-end B=2,048 fly training delivered 517
  positions/s including validation/checkpoints (575 during steady updates).

## Scientific evidence so far

**K=4 prototype:** wider pools improve full-validation KL by 0.0251 and
0.0162 across paired seeds, but value-MSE changes disagree (-0.00875/+0.02056).
Prior panels lost 64/64; PUCT won 1/64; Gumbel won 0/63 completed games, with
one capped game excluded. Evidence: `prototype-followup-v1/k4-summary.json`.

**Optimizer screen, full validation (70,425 positions / 800 games):**

| Global rate | Bias multiplier | Policy KL | Value MSE |
|---:|---:|---:|---:|
| 0.0003 | 1 | 1.8938 | 0.8654 |
| 0.001 | 1 | 1.8354 | 0.6716 |
| 0.003 | 1 | 1.7150 | 0.6423 |
| 0.01 | 1 | 1.6553 | 0.6089 |
| 0.03 | 1 | 1.8352 | 0.6803 |
| 0.1 | 1 | 1.9168 | 0.8961 |
| 0.03 | 0.1 | 1.7358 | 0.6046 |
| 0.03 | 0.01 | 1.6495 | 0.5711 |

At rate 0.03, bias scaling 0.01 reduces KL by 0.1857 (paired game-bootstrap
95% interval -0.1912 to -0.1798) and MSE by 0.1092 (-0.1339 to -0.0823).
Its final diagnostic batch has nonzero gradients on 13.53M edges, versus
19,036 at unscaled 0.03 and zero at 0.1. These are one-seed results at 32,000
exposures; multi-seed confirmation is active. Evidence:
`runs/optimizer-v0-full-validation-v1/summary.json`.

**Matched-control rate screen:** best CNN reaches slice KL 0.8684 / MSE 0.4285
at 0.003; the selected fly reaches 1.6078 / 0.6970 at 0.01. Each used 524,288
exposures and approximately 126.7M nominal FLOPs/prediction. CNN end-to-end
training takes about 69 seconds versus 1,014 for the fly. Padding, memory
traffic and parameter counts differ. Longer three-seed confirmation and full
validation/matches follow; no playing-strength advantage is claimed.

**New CPU execution result:** contiguous transpose destinations, weights packed
once per backward call, and exact all-zero-row skipping preserve full outputs,
losses and every gradient byte-for-byte on initial/trained V0 fixtures at B=1/32.
All 47 Rust / 50 Python tests pass, including signed-zero and nonfinite behavior.
On three spare cores, trained B=32 inference improves from 0.69 to 0.44 s and
loss-plus-gradient time from 4.32 to 1.70 s; initial dense-state gradients
improve from 4.35 to 2.67 s. Extra storage is 122.2 MB of graph indices plus
61.1 MB during backward. On 24 cores, trained B=32 inference is 0.232 → 0.172 s and gradients
1.171 → 0.594 s; initial gradients are 1.166 → 0.786 s. All reference
bytes match. Evidence: `runs/cpu-lane-*-v1`. Active
studies retain their original binaries. Skipping is data-dependent execution,
with every node, edge and parameter retained; nominal model FLOPs are unchanged.

**CNN recovery:** replaying 64 actual TPU trainer updates reproduces all 157
model/optimizer arrays and sampler exactly. The 131,072 replayed exposures are
qualification work, separately recorded in `next-gates/cnn-trainer-recovery.json`.

**Retinal ports:** 23,720 neurons have column annotations, but no sensory cells
do. Existing synapse-count-weighted column votes infer 3,721 photoreceptors;
confidence filtering selects 3,490. The first affine overlay covers 57/81
board points while other baseline ports remain. Spatial and side/type/channel
shuffled artifacts exist for seeds 1/2/3. No spatial learning result yet.

The offline model guide now illustrates the retinal/shuffled attachment and
matched CNN control alongside measured screens. Browser checks pass all
interactions, offline loading and widths 320–1,440 px; desktop/mobile figures
were visually reviewed. Evidence: `runs/model-guide/20260914/render-checks.json`.

**Completed matched confirmation:** full-validation KL is 0.7401–0.7556 for
CNN versus 1.5037–1.5054 for fly. Mean paired differences favor CNN by 0.7559
KL and 0.1540 MSE. Prior wins are CNN 94/96 versus fly 0/96; PUCT 93/96 versus
3/96; Gumbel 92/95 versus 0/95, with one capped game per family. This is a
substantial current gap. [Controlled results](research-results.md) records
all seeds, nominal compute/exposure contracts and uncertainty; raw counted
panels are not Elo. Final test labels remain outside tuning.

**Schedules:** absolute-update warmup/cosine settings are stored in checkpoints
and validated on resume. Full V0 replay across warmup and decay boundaries
reproduces all 31 arrays, sampler and checkpoint bytes. Diagnostics use a
bounded batch prefix, independent of the large training batch. Evidence:
`next-gates/schedule-cpu-recovery.json` (288 qualification exposures).

**Circuit audit:** by pass 4, possible sensory paths reach 148,736/149,210
readout neurons (99.7%). The 23,720 annotated column cells span 15 repeated
visual types. These are structural paths, ignoring gating/cancellation. The
corrected `circuit-routes-v3` accounts for missing class labels and uses
interneuron soma side. Earlier route counts agree; annotation corrections and
the failed first readout-artifact attempt are retained.

**Readout qualification:** `visual-readout-cpu-parity-v2` passes both new port
maps with the original tolerances. V1 exposed a zero-based step passed to the
one-based Adam reference in the new harness; only the harness was corrected.

**Constant-state cache:** the first `W * relu(h0)` message is now reusable at
fixed parameters, costing another 660,488 bytes. Full initial/trained B=1/32
states, outputs, losses, all gradients and the next V0 update remain byte-identical.
Three-core warm inference improves from 0.138 to 0.111 s at trained B=1 and
0.445 to 0.302 s at B=32. Training timings remain workload-sensitive; original
studies keep their binaries. `inference-work-v1` audits executed arithmetic on
256 declared validation inputs separately from nominal architecture FLOPs.

## Next gates and deliberate revisions

1. Confirm the completed optimizer and compute-matched rate screens
   with three seeds and longer equal horizons, then full validation
   and fresh common KataGo panels. Retain every failed/unhelpful trial.
2. Finish baseline/spatial/shuffled sensory attachment. Independently compare
   warmup/decay and a controlled visual readout. Calibrate these ordinary
   gradient-based choices before a structured zeroth-order/plasticity hybrid.
3. Select a reproducible fly prior before online relabeling/distillation.

The TPU became available and its exact tiled learner passed qualification,
so it now accelerates bounded studies. The CNN control moved ahead of plasticity
while CPU optimization completed, making the grand-goal comparison measurable.
High-rate activity collapse motivated a separate bias-rate factor. Missing
sensory coordinates motivated an inferred, audited overlay with a matched
shuffle. Original prototype contracts and failed numerical evidence are retained.
Record further reasons and affected contracts before dependent work.

The longer constant-rate behavior motivated separate warmup and decay factors
before zeroth-order work. The measured strength gap, short sensory path lengths
and repeated visual columns motivated testing output aggregation. These
revisions retain original study contracts, topology and final test isolation.

Local commit `ee520ac` contains qualified schedules, visual readouts and analysis.
Upstream remains at `772db92`: automatic approval review rejected the next push
and requires explicit user confirmation of `git@github.com:honglu2875/flygo.git`.
Local implementation and experiments continue; no further push is attempted.

## Readout offset investigation — 2026-09-14

A read-only reconstruction of retained B32 fixtures found 99.923% common
pooled energy at initialization, versus 5.20% in the trained bias-scaled
checkpoint. The six-update smooth fixtures retained over 99% common energy
and saturated all value outputs. The audit checks reconstructed heads against
saved native outputs and adds no optimization exposures:
`next-gates/readout-offset-audit-v1.json`.

A fixed invertible mean-damping transform was implemented and passed independent
small-fixture derivatives, Rust/JAX updates, batch independence and portable
checkpoint checks. Source `a5c061aba55a5f50cf64` uses direct mean subtraction;
source `2d7c3ad5728e93e20125` uses a more stable deviation-based evaluation of
the same linear map and its symmetric transpose. Neither source met the strict
full-model parameter-trajectory gate. At unchanged rtol .003 / atol 5e-6:

| Attempt | Failure |
|---|---|
| `readout-mean-hard-cpu-parity-v1`, B1 | Third update: 38/53,792 policy weights; max error 7.42e-5 |
| `readout-mean-hard-cpu-parity-v2`, B1 | First update: one leak parameter; max error 8.15e-6 |
| `readout-mean-hard-b32-cpu-parity-v2`, B32 on w3 | First update: 5/53,792 policy weights; max error 2.13e-4 |

States, losses and gradients passed before these failures. Near-cancelled
gradients and Adam's epsilon sensitivity are the working explanation, not a
claim that every discrepancy has been fully resolved. The local B32 attempt
was refused by the existing RAM budget before allocation. The bounded w3
attempt used the released spatial-evaluation lane and exited before its next
owner. Evidence was pulled through the primary host's established SSH path;
no host-key policy was changed. Scientific mean-conditioning trials remain
unlaunched. The simpler head-rate screen advances on source
`6d9b14c70b154a5cb821`, which predates the mean transform.

The updated offline HTML guide passes all interactions, graph/data consistency,
reduced motion, no external requests and widths 320–1,440. Hard/smooth rate
selectors include the correct threshold derivative and negative-state response.
Desktop and mobile figures were visually reviewed. Evidence:
`runs/model-guide/20260914-smooth/render-checks.json`.

## Head-rate and CPU validation gates — 2026-09-14 06:40 UTC

All four combinations of hard/smooth .01 and value-only/all-output-weight
rate scaling pass the independent learner gate and actual V0 six-update
training plus fresh update 3→6 recovery. Every one of 28 checkpoint arrays and
the sampler matches; peer w1 recovery also passes. Each qualification consumes
288 separately counted position exposures. Evidence:
`next-gates/head-rate-{parity-v1.log,recovery-v1.json}`. The registered 32k
screen waits on the four schedule-confirmation validations and uses immutable
source `6d9b14c70b154a5cb821`, without mean conditioning.

Source `599a1030d1aaf214fd8c` removes the duplicate core-parameter finite scan
and parallelizes the remaining scan on the pinned model executor. Shape checks
remain, and invalid restoration of every parameter family is atomic. All
48 Rust and 62 Python tests pass. Initial and trained B1/B32 outputs, losses
and gradients match the previous source byte-for-byte; three actual fresh
scheduled continuation updates match all arrays and sampler. The completed
w3 24-core profiles also retain matching numerical artifact hashes. Runtime
reports/provenance were copied to w0; large reference arrays remain on w3.
Evidence: `next-gates/parallel-validation-{check-v1.log,recovery-v1.json}`,
`parallel-validation-{initial,trained}-v1`, `cpu-validation-lane-v1`.

The first shuffled seed 2 TPU attempt failed before any optimizer update when
a step-zero checkpoint reserve would cross the unchanged available-RAM floor.
All four workers exited; an overlapping bounded CPU qualification had ended
before recovery. `spatial-input-recovery-v2` retains the failure, adopts seven
successful cohorts and restarts only seed 2 under an attempt-specific ID before
the original unstarted seed 3. Every host passed a 58 GiB heap plus 4 GiB file
preflight. The retry has published its initial checkpoint and is training.
Dependent smooth-confirmation and small-CNN launchers use new attempt records;
scientific contracts and qualified source versions remain unchanged.

## Streaming prediction and Adam epsilon — 2026-09-14 08:15 UTC

Prediction now uses the same recurrent step and readout functions as the
differentiation path but retains only the current state. The public Python
inference interface is unchanged; explicit traces still retain all states.
Source `224914260953ab863b96` matches every initial/trained hard and smooth
.01/.05 output, loss and gradient byte against the preceding implementation.
The first profiling attempt stopped because an old smooth reference file was
absent; the new attempt independently generated and retained that reference.
Both attempts and all six completed cases remain under `cpu-streaming-lane-v*`.

At K32/B128, initial-model peak process RSS falls from 4,004,872,192 to
1,434,501,120 bytes. Warm times are approximately 6.3–6.4 s in both versions;
there is no resolved latency gain. This is a depth/memory implementation probe,
not a trained architecture result. [Raw repetitions and hashes](results/streaming-memory-v1.json).

`cpu-streaming-recovery-v1` and `v2` reproduce all 28 checkpoint-array hashes
and sampler after three actual continuation updates. Their extra exact
comparison of diagnostic FP64 gradient norms fails in the last bits, including
one attempt with the same three-thread count. For example, one step records
15.691457648564205 versus 15.691457648564212. These remain failed diagnostic
gate records; the reports do not claim that every logged float is bitwise
identical. No scientific numerical tolerance was relaxed.

Adam epsilon is now explicit in Rust, the Python API, JAX and checkpoint
training contracts. It is positive finite FP32 and lies outside the square
root of the bias-corrected second moment. Legacy checkpoints imply 1e-8;
resume rejects an unintended epsilon change. Source `c65c9808cdf39ee16587`
passes all four hard/smooth-.01 by epsilon-1e-6/1e-4 full-graph CPU reference
checks, including every gradient and three free-running updates. Each case
also completes six actual scheduled/scaled V0 updates and fresh 3→6 recovery
with all 28 arrays and sampler exact. [Complete gate evidence](results/epsilon-qualification-v1.json).
Actual TPU epsilon and smooth-rate qualification remain pending.

## Deployment recovery and smaller control — 2026-09-14 08:15 UTC

Immutable bundle replication now hashes existing peer files and archives only
missing entries. Equal-size content conflicts and escaping symlinks fail before
transfer; the extraction path still verifies racing existing files and publishes
atomically. Tests execute the real inventory/extraction protocol on two fixture
roots, including partial, idempotent and interrupted-receipt cases. Source
`4fe16e716e2ab90ec073` passes **48 Rust and 72 Python tests**. Large running
studies keep their original model sources.

The smooth-confirmation host-2 deployment stopped before launch because its
old bundle path tried to archive already present feature data. The hard-e6
screen stopped at initial checkpoint admission with only step-zero validation
logged, no checkpoint and no optimizer updates. Missing-only deployment
resumes the former with the identical plan/source and reruns the latter on
host 2 under an attempt-specific ID; original failures remain. File/RAM floors
and the own-file cap are unchanged.

The small-CNN's first qualification lookup failed before any worker launch:
the frozen launcher resolved its default worker relative to the wrong folder.
The launcher now resolves sibling scripts, and `tpu-small-cnn-parity-v2`
passes on all 16 TPU devices: full outputs, losses, gradients and three
checkpoint-aligned CPU/TPU transitions at the original pointwise tolerances,
maximum absolute reported error 3.54e-5. Free-running cross-backend trajectory
identity is not claimed. The completed screen selects .01 by the declared
final-slice KL+MSE rule.

Confirmation seed 1 completes. The first seed-2 attempt fails at initial
checkpoint admission while a development file reservation overlaps; all four
controllers stop before updates. The replacement adopts seed 1, repeats seed 2
from the same initialization and runs original seed 3. All hosts pass an
additional 4 GiB files / 58 GiB heap preflight, and no large build overlaps the
recovered cohorts. All three final validations and common 32-game panels are
now complete. Attempt IDs change, scientific settings and exposure budgets do not.

The proposed archive of 54 retired prototype checkpoint replicas stopped
before moving anything on storage admission. Automatic approval review then
rejected its retry because it would remove local copies after verification on
another host's volatile RAM. No checkpoint was relocated or removed; the
safer deployment optimization above permits progress within the existing limits.

## Source-driven sparse-kernel probe — 2026-09-14 08:45 UTC

Snapshot `0377c6696cbbccb9cb07` adds a profiling-only source traversal with
disjoint destination partitions. Exhaustive small-graph activity masks and
full-graph B1/B32 initial, trained hard and trained smooth fixtures all retain
reference output bits. The experimental source passes 49 Rust / 72 Python tests.
Its kernel takes about 2.1–2.5 times as long as the existing multiply on trained
hard fixtures, and also regresses on dense fixtures. Production selection was
never changed. The unused method/test are removed after preserving the frozen
source and exact working diff under `sparse-source-profile-v1`.

The profiler retains a separate parameter-validation timer: about 1.2–1.6 ms
in these cases. This alone does not justify a state-ownership refactor. The
[complete profiling artifact](results/sparse-source-profile-v1.json) distinguishes
kernel measurements from full prediction latency and retains all repetitions.

After removing the candidate, source `9058055f34adfea1b7bd` passes 48 Rust and
72 Python regressions and a direct retained-profiler smoke check. The existing
three-update actual TPU gate passes for this source's default epsilon,
following an all-host additional 4 GiB file / 60 GiB heap preflight. All four
controllers then reproduce the step-2-to-step-3 checkpoint arrays and sampler
exactly in fresh processes, with the portable Rust update checks also passing.
No pointwise tolerance changes. [Complete evidence](results/current-tpu-default-v1.json).
This does not qualify nondefault epsilon or smooth rates on the TPU.

## Final smooth confirmation — 2026-09-14 09:14 UTC

The recovered seed-2 learner completes all 4,000 updates and 128,000 exposures
on its original immutable source `6d9b14c70b154a5cb821`. Its existing validator
then completes the full held-out split. The new collected recovery study adopts
the already completed seeds 1/3 and all three hard controls; no validation is
rerun. All three paired comparisons verify the same dataset, seed, exposure
horizon and position/game indices. The [complete report](results/smooth-rate-confirm-v1.json)
retains checkpoint and metric hashes alongside conditional game-bootstrap
uncertainty and variation across seeds. Original pre-launch failures remain.

All currently registered scientific cases now have final reports, apart from
the explicitly deferred mean-conditioning proposal, which never launched
scientific training after its failed update gate. Generation and checkpoint
replication remain active. At 09:10 the fleet has 105,902 published games,
32 healthy workers and all original storage buffers intact.
