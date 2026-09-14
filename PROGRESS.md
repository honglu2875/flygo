# Progress

Updated **2026-09-14 01:23 UTC**. Autonomous work is authorized until about
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
| M6 — studies | Active | K=4 prototype done; K=8 running; optimizer validation and matched CNN screen active |
| M7 — TPU | Qualified and in use | Four-host SPMD; full-model updates, matching copies and bitwise full-trainer replay |
| M8 — online refinement | Pending model selection | Offline prior/PUCT/Gumbel works; stronger prior and controlled comparison first |

## Live work

- **Generation:** 32 workers × 16 concurrent games, using 64 pinned physical
  cores per host (`0–31,60–91`). Production and frozen research releases are independent.
- **Prototype:** K=4/8 × G=656/2,624 × seeds 1/2, 10,000 updates at B=32,
  rate 0.003. All K=4 validation/matches finished. K=8 was at 7,590–7,970
  updates at 01:21; follow-up workers wait for final verified checkpoints and
  trainer exit before reusing `92–115` for full validation and fresh matches.
- **CPU optimizer:** all six global rates and two bias-rate variants completed
  1,000 updates at B=32 on V0. Full validation of every final checkpoint is now
  running on the released `32–55` lanes, two models per host. Evidence:
  `runs/optimizer-v0-full-validation-v1`.
- **TPU matched screen:** `matched-control-v1` runs one cohort at a time using
  all 16 devices. Both families get rates 0.001/0.003/0.01, seed 1, B=2,048
  and 256 updates (524,288 training exposures). All three CNN trials finished;
  two additional fly rates run next, adopting the completed fly 0.003 trial.
  The queue freezes source/runtime/configs and stops on a retained failure.

Runtime artifacts stay under `/dev/shm/flygo`. Floors remain **64 GiB free
shared memory**, **96 GiB available RAM**, and a **100 GiB own-file cap** with
reservations. At 00:56, free shared memory was 157–181 GiB and available RAM
336–362 GiB. Generation uses 256 physical cores across the fleet; CPU trials
and evaluations use another 192. TPU runtime/development uses spare cores;
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

- **Regression:** 44 Rust tests and all 50 current Python tests pass. Includes
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

**Optimizer screen, fixed validation slice:** rate 0.01 reaches KL 1.6725 /
value MSE 0.6261, versus 1.7382 / 0.6676 at 0.003. Rate 0.1 loses activity
and reaches only 1.9495 / 0.8927. Rate 0.03 with bias multiplier 0.01 reaches
1.6711 / 0.5930. These are one-seed, 32,000-exposure results; complete
validation and longer multi-seed confirmation are required.

**Matched-control screen, provisional:** the CNN at 0.003 reaches validation
KL 0.8684 / MSE 0.4285, versus the fly's 1.7646 / 0.6281 at the same rate,
524,288 exposures and approximately 126.7M nominal FLOPs/prediction. CNN
training takes 69 seconds versus 1,014 for the fly. Other fly rates are still
running. Padding, memory traffic, parameter counts and actual throughput differ;
these initial slice results do not establish playing strength or optimal tuning.

**Retinal ports:** 23,720 neurons have column annotations, but no sensory cells
do. Existing synapse-count-weighted column votes infer 3,721 photoreceptors;
confidence filtering selects 3,490. The first affine overlay covers 57/81
board points while other baseline ports remain. Spatial and side/type/channel
shuffled artifacts exist for seeds 1/2/3. No spatial learning result yet.

## Next gates and deliberate revisions

1. Finish both optimizer and compute-matched rate screens; confirm promising
   settings with three seeds and longer equal horizons, then full validation
   and fresh common KataGo panels. Retain every failed/unhelpful trial.
2. Run baseline/spatial/shuffled sensory attachment at a declared common
   optimizer. Then test one structured zeroth-order or plasticity mechanism
   with explicit state equations and counted queries.
3. Select a reproducible fly prior before online relabeling/distillation.

The TPU became available and its exact tiled learner passed qualification,
so it now accelerates bounded studies. The CNN control moved ahead of plasticity
while CPU optimization completed, making the grand-goal comparison measurable.
High-rate activity collapse motivated a separate bias-rate factor. Missing
sensory coordinates motivated an inferred, audited overlay with a matched
shuffle. Original prototype contracts and failed numerical evidence are retained.
Record further reasons and affected contracts before dependent work.
