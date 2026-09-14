# Progress

Updated **2026-09-14 03:51 UTC**. Autonomous work is authorized until about
09:02 UTC. The target is an efficient fixed-fly Go engine, compared with a
conventional neural engine at matched prediction FLOPs and training-position
exposure. No fly strength advantage has been established.

[README](README.md) describes interfaces; [RESEARCH](RESEARCH.md) records study
contracts and biological hypotheses; [MILESTONES](MILESTONES.md) defines gates.
Earlier qualification detail remains in [history](docs/qualification-history.md).

| Milestone | Status | Evidence / remaining work |
|---|---|---|
| M0–M2 — design, Go, expert pilot | Complete | Rust/Python rules/search parity, real KataGo games, teacher/history/storage qualification |
| M3 — corpus | V0 complete; production continues | Balanced million-position release and cache verified on all four hosts |
| M4 — fly forward | Complete | All 165,122 nodes / 15,270,273 edges; CPU and actual TPU parity |
| M5 — CPU learning | Complete | V0 training, bitwise fresh local/peer recovery and exported legal Go games |
| M6 — studies | Active | Screens complete; longer three-seed optimizer and matched CNN confirmation running |
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
  prior/PUCT/Gumbel panels run on CPU. The registered `spatial-input-v1`
  adopted all three fly baselines and is training spatial/shuffled ports at
  identical settings/horizons. One cohort owns all 16 devices.
- **Schedule screen:** four seed-1 warmup/decay cases are registered at 128,000
  exposures. Each starts after its matched panels leave `92–115`; two have
  started. All use source `c149278eda30697520f9` and queued full validation.
- **Visual readout:** a separate input-fixed, visual-output spatial/shuffled
  screen is queued after schedule validation. Both variants pass full-state,
  all-gradient and three-update JAX CPU parity on real V0 inputs. They retain
  23,565 annotated output cells, exclude 155 outside the sensory bounds, and
  cover 60 board points. The full recurrent topology still executes.
  Source `c149278eda30697520f9`, B=32, 4,000 updates, seed 1; no learning result yet.

Runtime artifacts stay under `/dev/shm/flygo`. Floors remain **64 GiB free
shared memory**, **96 GiB available RAM**, and a **100 GiB own-file cap** with
reservations. At 03:41, free shared memory was 142–178 GiB and available RAM
315–353 GiB. All 32 generation workers are healthy; 69,153 games are published. Generation uses 256 physical cores across the fleet; CPU trials
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

- **Regression:** 47 Rust tests and all 56 current Python tests pass. Includes
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

**Current playing evidence:** matched CNN seeds 2/3 each win 32/32 prior games
against opponent 0 at 16 visits. Their PUCT panels win 31/32 each and Gumbel
31/32, 32/32. Fly seed 1 wins 0/32 prior and 1/32 PUCT. This is a substantial
current strength gap; the remaining panels are still running. These fixed
openings are for this registered screening comparison, not an Elo estimate.

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

Local commit `51d6110` contains the qualified CPU speedup and updated guide.
Upstream remains at `772db92`: automatic approval review rejected the next push
and requires explicit user confirmation of `git@github.com:honglu2875/flygo.git`.
Local implementation and experiments continue; no further push is attempted.
