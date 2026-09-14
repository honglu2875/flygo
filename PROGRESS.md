# Progress

Updated **2026-09-14 05:31 UTC**. Autonomous work is authorized until about
09:02 UTC. The goal is an efficient Go engine with fixed fly topology and
learned strengths. The current baseline is substantially weaker than the CNN
control; no fly advantage has been established.

[README](README.md): interfaces · [RESEARCH](RESEARCH.md): study contracts ·
[Results](docs/research-results.md): findings · [History](docs/qualification-history.md): detailed gates.

| Milestone | Status | Evidence / remaining work |
|---|---|---|
| M0–M2 — design, Go, expert pilot | Complete | Rust/Python rules/search parity, real teacher games, history/storage qualification |
| M3 — corpus | V0 complete; production continues | Balanced million-position release verified on four hosts |
| M4 — fly forward | Complete | All 165,122 nodes / 15,270,273 edges; CPU and actual TPU parity |
| M5 — CPU learning | Complete | Bitwise fresh local/peer recovery; exported legal Go games |
| M6 — studies | Active | Matched CNN confirmation complete; optimizer, schedule, spatial and smooth-rate refinements |
| M7 — TPU | Qualified and in use | Four-host SPMD, full-model gradients/updates, exact trainer replay |
| M8 — online refinement | Pending prior selection | Offline prior/PUCT/Gumbel works; controlled stronger-prior selection first |

## Live work and ownership

| Study | Contract | Current state / successor |
|---|---|---|
| Expert generation | 8 workers × 16 games per host; 64 pinned physical cores | All 32 workers healthy; continues independently |
| CPU optimizer confirmation | Three rates × three seeds; 128k exposures, B32 | Eight runs and their full validations complete; final bias-scaled seed 3 on w0 `32–55` |
| Spatial input, TPU | Baseline/spatial/shuffled × three seeds; 1,048,576 exposures, B2048 | All baseline and spatial training done or finishing; shuffled cases next |
| Spatial input followups | Full validation + 32 prior/PUCT/Gumbel games per case | Queued on w1–w3 `32–55` after optimizer validation; panels share declared openings |
| Visual readout, CPU | Input-only/spatial-output/shuffled-output; 128k exposures, B32, seed 1 | Final training/full validation on w0/w2/w3 `92–115` |
| Smooth rate, CPU | Softness .01/.05 × two optimizer choices; 32k exposures, B32, seed 1 | First case training on w1 `92–115`; others wait for visual validation |
| Schedule confirmation | Warmup/cosine, bias-scaled .03, seeds 2/3; 128k exposures | Queued on all `92–115` lanes after smooth validation; adopts seed 1 and constant controls |
| Smaller CNN, TPU | 8.362M nominal FLOPs; same three-rate screen and exposure budgets | Queued after spatial cohort; actual TPU shape parity must pass first |

Runtime artifacts live under `/dev/shm/flygo`. At **05:23**, production had
**80,391 games**, free shared memory was **125–176 GiB**, and available RAM
**297–351 GiB**. Floors remain 64 GiB free shared memory, 96 GiB available RAM,
and a 100 GiB own-file cap with reservations. Generation owns `0–31,60–91`;
research uses `32–55` and `92–115`; TPU/development uses spare cores and waiting
coordinators use 116. Peer copies and RAM remain volatile. Keep generation,
replication coordinators and owned SSH keepalives alive.

## Frozen data and reproducibility

V0 has 10,256 complete games / 1,000,201 positions: train 887,338, validation
70,425, test 42,438. It has 840,659 unique board/turn states, 63,354 D4-novel
validation inputs and 79.1% saturated teacher values. Compressed size is
241.1 MB; release/cache checks pass on all four hosts. Final test labels stay
outside tuning. Earlier releases and failed attempts remain retained.

| Qualified source | Use |
|---|---|
| `2185746d8cb5a7b93d1c` | All original CPU optimizer confirmations |
| `0d4e6e6ea6298d22370c` | Matched TPU studies and spatial-input study |
| `c149278eda30697520f9` | All schedule and visual-readout trials |
| `7a9c190de16ddd46f739` | Constant-state cache, smaller CNN preparation/queued TPU work |
| `6d9b14c70b154a5cb821` | Qualified parallel smooth-rate CPU trials |

Current regression: **48 Rust and 60 Python tests pass**. Full-graph Rust/JAX
CPU parity passes both smooth scales, including all gradients and three
free-running updates. Actual full-V0 smooth trainer checkpoints reproduce all
28 arrays and sampler across three fresh scheduled, scaled updates. The
parallel implementation preserves full smooth/baseline outputs and gradients
byte-for-byte at B1/B32. Actual TPU smooth-rate qualification is still pending.

Earlier hard-rate actual TPU gates include full states/all gradients/three
updates and 64 actual learner updates reproduced bitwise after fresh restore.
The CNN's 64-update recovery reproduces all 157 arrays and sampler. Its
cross-backend parity uses three checkpoint-aligned transitions; failed
free-running ReLU-boundary cases are retained, with tolerances unchanged.

## Findings that guide the next steps

- **Matched baseline:** at 1,048,576 exposures and ~126.7M nominal prediction
  FLOPs, full-validation CNN KL is .7401–.7556 versus fly 1.5037–1.5054.
  CNN prior wins 94/96 versus fly 0/96 against the declared early KataGo
  checkpoint; PUCT wins 93/96 versus 3/96. These panels are not Elo.
- **Compute accounting:** hard-rate CPU skipping and the constant initial-message
  cache lower counted warm B1 work to 8.31–8.87M operations across the three
  trained fly seeds. This is input-dependent arithmetic, not a hardware
  instruction count. The 8.362M CNN control tests this stricter cost scale.
- **Optimizer screen:** at 32k exposures, .01 outperforms smaller global rates.
  At .03, scaling bias updates by .01 improves KL/MSE and preserves nonzero
  gradients on 13.53M edges versus 19,036 without scaling. Multi-seed longer
  confirmation is nearly complete; high-rate activity collapse remains relevant.
- **Schedule screen:** at 128k exposures and seed 1, bias-scaled cosine reaches
  KL 1.5071 / MSE .5347; warmup reaches 1.5180 / .5219; constant reaches
  1.5343 / .5422. Both advance to paired seeds 2/3 before a general claim.
- **Prototype depth:** K8 does not consistently improve on K4. Wider readout
  helps policy but has mixed value effects; all eight runs and panels complete.
- **Spatial structure:** inferred input columns cover 57/81 board points;
  annotated visual readouts cover 60. These are audited affine overlays with
  matched shuffles, not calibrated retinal geometry or a demonstrated Go gain.

## Engineering changes and next gates

The CPU caches the first constant message, packs transpose weights once per
backward call, skips exact-zero rows and parallelizes independent rate/readout
work while preserving reduction order. On 24 cores, trained smooth B32
inference falls from .558/.673 s to .239/.242 s; gradients from 1.711/1.945 s
to .916/.925 s at softness .01/.05. Raw full-array parity and recovery evidence:
`runs/cpu-smooth-lane-v1`. Frozen running studies keep their original binaries.

1. Collect every optimizer, visual and smooth final validation; compare aligned
   positions and paired seeds without best-step selection.
2. Complete spatial/shuffled TPU cases and followups. Qualify/run the smaller
   CNN control and confirm its selected rate at the matching horizon.
3. Finish schedule confirmation, then choose a controlled stronger fly prior.
   Keep new adapter/dynamics/plasticity factors isolated and record revisions
   before dependent experiments. Online distillation follows prior selection.

`queue_cpu.py` freezes scripts/plans and waits for all lane dependencies and
worker exits. Its first launch failed before training because of inherited
housekeeping affinity; corrected attempt `smooth-rate-launch-v2` is running.
The original failure remains recorded. `dev.py` now preserves the installed
native package during rebuilds, preventing transient maintenance import failures.

Local implementation and commits continue. Upstream remains at `772db92`:
automatic approval review rejected publication and requires explicit user
confirmation of `git@github.com:honglu2875/flygo.git`. No further push is attempted.
