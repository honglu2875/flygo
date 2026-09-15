# Progress

Updated **2026-09-15 13:25 UTC**. Fixed fly topology, learnable strengths.
Compare **prediction FLOPs and labeled-position exposures**; report training,
tuning, memory and latency costs separately. No advantage over the matched CNN
has been established. Detailed history stays in the linked study reports.

[Interfaces](README.md) · [Contracts](RESEARCH.md) ·
[Results](docs/research-results.md) · [Qualification history](docs/qualification-history.md)

## Current priority

[Clipping confirmation](docs/clipping-confirmation-study.md) is running on CPU:
global versus per-parameter-group clipping, fresh paired seeds **13/14/15**,
**4,000 B32 updates / 128,000 exposures per endpoint**. All six learners have
verified thread affinity and initial checkpoint replicas. Full validation,
motor probes and prediction counts are queued after each final checkpoint
and verified recovery copy. **Final results remain pending.**

Only clipping differs within a pair. Keep B's current-board spherical input,
dense 2,129-motor heads, K8 hard recurrence, reset state and optimization
settings fixed. All **12 numerical / six exact recovery** gates pass; **13
analysis checks** pass on root and worker 2. Qualification adds **3,264** V0
update exposures separately from **768,000** planned scientific exposures.
All 306 qualification evidence files have verified owner/peer archives.

| Worker | CPUs 32–55 | CPUs 92–115 |
|---|---|---|
| 1 | Global, seed 13 | Per-group, seed 15 |
| 2 | Per-group, seed 13 | Global, seed 14 |
| 3 | Global, seed 15 | Per-group, seed 14 |

The tested automatic collector waits for actual trainer exit and unlocked,
completed followups, audits each endpoint, relays verified evidence to worker
2 and runs the frozen paired analysis. Five collection checks pass. Final
analysis and the optimizer decision remain pending.

A separate [policy/value gradient audit](docs/loss-balance-study.md) is complete:
all nine cases recorded, **seven passing / two failed numerical gates**. Value
gradients dominate the shared circuit on the fixed 32-position training probe;
an independent bias adjoint closely traces the failures to FP32 summation.
All evidence has verified second-host copies. A value-to-circuit gradient
scale is a proposed next study, pending the optimizer decision and separate
qualification. Persistent state and propagation changes remain separate.

## Milestones

| Milestone | Status | Remaining work |
|---|---|---|
| M0–M2: design, Go engine, expert pilot | Complete | Optional throughput improvements |
| M3: corpus | V0 complete; generation stopped | Preserve stop markers; separately register later releases |
| M4–M5: fly execution and learning | Rust/Python/JAX paths qualified | Qualify each new numerical model independently |
| M6: controlled studies | Clipping confirmation running; loss-balance audit complete | Close fixed endpoints, then qualify a separate loss-routing study |
| M7: four-host TPU | Default path qualified; use paused | Nondefault spherical/epsilon/smooth paths need separate TPU gates |
| M8: online refinement | Pending a useful prior | Prior/PUCT/Gumbel interfaces and panels work |
| Biological groups and interfaces | Mapping, readout and signal studies complete | Improve learnability through separately controlled changes |
| Research notebook | Executed public Hub walkthrough complete | Available for independent researcher exploration |
| Persistent state | Rust core primitive and trajectory audit complete | Model/Python/JAX integration and scientific training pending |

## What the results support

- **Matched CNN:** small CNN policy KL **.8929–.9194**, versus fly
  **1.5037–1.5054**, at the registered 1,048,576 exposures and approximately
  8.3–8.9M prediction FLOPs. CNN wins 89/96 common-opening prior games;
  fly wins 0/96. These panels are not Elo. Same-lane B1 CPU medians are
  1.30 ms for small CNN and 23.73 ms for that fly configuration.
  [Comparison and cost accounting](docs/research-results.md).
- **Earlier smooth recurrence:** full-validation KL **1.4866–1.4930**,
  three-seed mean **1.4903**, at 128k exposures. Its input/readout contract
  differs from the current K8 spherical studies. It is not a new matched-CNN
  result. [Smooth-rate confirmation](docs/research-results.md#smooth-rate-confirmation-three-seeds).
- **Clipping discovery:** at 32k exposures, per-group improves mean KL
  **1.70966 → 1.63205** and value MSE **.64897 → .58767**, in every seed.
  Warm B1 prediction FLOPs rise **185.12M → 201.43M** (+8.81%). Entropy
  falls only .04047; the leading motor component still explains 96.95–99.71%
  of raw visual-response variance. A provisional optimizer candidate, with
  no CNN or playing-strength claim. [Complete study](docs/group-clipping-study.md).
- **Signal flow:** 48 recurrence and 75 independent VJP group checks pass.
  Initial motor visual variation is about 2,322 times smaller than total
  variation. Trained responses and gradients are strongly concentrated;
  most edge Adam denominators are epsilon-dominated. All motors are two to
  five hops from an eye. These measurements motivate isolating optimization;
  they do not establish a biological function. [Audit](docs/signal-flow-study.md).
- **Policy/value pressure:** on the three passing per-group discovery endpoints,
  shared value-gradient norms are 8.20–29.38 times policy norms and oppose
  them. This is one training probe and raw gradient geometry, not an Adam-step
  or validation result. Two other cases fail a native bias decomposition;
  all 30 independent head-boundary checks pass. No learner changed.
  [Audit, numerical failures and proposed follow-up](docs/loss-balance-study.md).
- **Readouts and retinal mapping:** larger retinal allocation and soma-side
  policy heads do not show a repeatable policy benefit. The fresh-seed soma
  confirmation worsens mean KL by .01874 despite better value MSE. Retain
  the B current-board montage and dense learned motor projections.
  [Retinal study](docs/retinal-allocation-study.md),
  [readout confirmation](docs/readout-confirmation.md).
- **Motor decoders:** all 27 convergence checks pass. Ridge .01 gated heads
  improve subset KL **1.68393 → 1.54872**, while weak regularization fits
  training KL .68778 and worsens validation to 2.07265. All 97.42M solver
  label exposures are counted. This identifies optimization and overfitting,
  not a motor-information ceiling. [Study](docs/motor-convergence-study.md).
- **Research notebook:** [quintic/go9x9 walkthrough](notebooks/flygo_research_walkthrough.ipynb)
  downloads 96 games from a pinned revision, selects 512 training / 256
  validation positions from disjoint opening families, explores weights,
  spherical eye mapping and recurrent propagation, then trains for 64 B32
  updates. Actual plots and outputs are saved: KL **2.03807 → 1.90137**
  in **387 s on four CPU cores**; value MSE **.88994 → .90108**.
  This is a different public sample, not a V0 benchmark.
  [Setup](notebooks/README.md), [execution record](docs/results/research-notebook-hf-v1.json).
- **Persistent state:** ten Rust core tests pass, including finite differences
  and split-trajectory composition. The public model/training interfaces do
  not yet carry state across Go moves. The replay audit shows current-player
  color reversal after passes; qualify an absolute-color reset control before
  attributing a future gain to persistent memory.
  [Design and audit](docs/temporal-vision-study.md).

## Resource and evidence rules

**TPU remains paused.** Current work uses CPU and does not reserve TPU devices.
Data-generation stop markers appeared on all four hosts around **2026-09-14
23:46 UTC**; processes exited. Preserve those markers. V0 remains frozen at
10,256 games / 1,000,201 positions: train **887,338**, validation **70,425**,
test **42,438**. Final-test labels remain outside tuning.

Keep the **100 GiB own-file cap**, **64 GiB free-filesystem floor** and
**96 GiB available-RAM floor**, including live reservations. Artifacts under
`/dev/shm/flygo` are volatile. Root is near its allowance: reserve log growth,
keep new checkpoint payloads and bulk response arrays on workers, and relay
between workers through root pipes. Worker 2 has the qualified analysis
runtime. Do not use the default bulk-to-root collection for confirmation.

Preserve immutable sources, failed attempts, registered endpoints and verified
recovery copies. Archive qualification and scientific evidence on a second
worker. Changes to scientific contracts need an explicit rationale and a new
registration; completed results are never silently revised.

The authorized publication destination is **master** at
`git@github.com:honglu2875/flygo.git`. Current status is kept here; detailed
completed studies, engineering attempts and pending biological ideas remain in
[qualification history](docs/qualification-history.md),
[biological interfaces](docs/biological-interfaces.md),
[neuron groups](docs/group-study.md) and [regrowth](docs/structural-plasticity.md).
