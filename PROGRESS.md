# Progress

Research status last checked **2026-09-13 20:07 UTC**. Eight new baseline/variant trials are training
on a cloned 1.48M-position snapshot. Generation continues on all four hosts.
The earlier nine 100k-data trials and their evaluation panels are complete.
See [README.md](README.md) for commands and [MILESTONES.md](MILESTONES.md) for gates.

**Model guide added 2026-09-13 21:32 UTC.** The standalone
[interactive HTML guide](docs/model.html) explains the fixed graph, sensory
attachment, pooling/heads, recurrent equation, current 2×2 study, proposed
extensions, offline distillation/search, and CPU/TPU costs. It embeds an actual
64×64 connectivity summary and a validation-position prediction from the
K=4/G=656/seed-1 step-2,000 checkpoint, with source IDs and hashes. Illustrative
circuits and proposed mechanisms are labeled separately. Chromium checks
passed for all controls, offline operation, reduced motion and widths from
320 to 1,440 pixels; figures were visually reviewed. Evidence is in
`/dev/shm/flygo/runs/model-guide/render-checks.json`; a
[preview image](docs/model-preview.png) is also available.

| Milestone | Status | Evidence / remaining gate |
|---|---|---|
| M0 — design | Complete | [DESIGN.md](DESIGN.md), [DATASET.md](DATASET.md), explicit fixed-topology contract |
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
- [configs/prototype-v1.json](configs/prototype-v1.json) crosses K=4/8 with
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
