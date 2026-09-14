# Progress

Updated **2026-09-14 06:45 UTC**. Autonomous work is authorized until about
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
| Spatial input, TPU | Baseline/spatial/shuffled × three seeds; 1,048,576 exposures, B2048 | Seven cohorts complete; shuffled seed 2 retry training after zero-update RAM admission failure; original seed 3 follows |
| Spatial input followups | Full validation + 32 prior/PUCT/Gumbel games per case | All spatial and shuffled seed 1 panels complete; replacement seed 2 and original seed 3 waiters active |
| Visual readout, CPU | Input-only/spatial-output/shuffled-output; 128k exposures, B32, seed 1 | All three final full validations complete; spatial output beats its shuffle but loses to broad output |
| Smooth rate, CPU | Softness .01/.05 × two optimizer choices; 32k exposures, B32, seed 1 | All four final validations complete; selected .01/bias-scaled variant advances to three-seed 128k confirmation |
| Schedule confirmation | Warmup/cosine, bias-scaled .03, seeds 2/3; 128k exposures | All four additional seeds training on `92–115`; adopts seed 1 and constant controls |
| Readout conditioning | Fixed common-mean damping; registered 32k exposures | Deferred: B1 and actual-B32 parameter-update parity fail; no learning trials launched |
| Head-rate controls | Value-only / policy+value weight rates × hard / smooth .01; 32k exposures | All four specific rate/recovery gates pass; queued after schedule-confirmation validation |
| Smaller CNN, TPU | 8.362M nominal FLOPs; same three-rate screen and exposure budgets | Replacement launcher waits for recovered spatial study; actual TPU shape parity must pass first |

Runtime artifacts live under `/dev/shm/flygo`. At **06:23**, production had
**87,107 games**, free shared memory was **117–174 GiB**, and available RAM
**295–355 GiB**. Floors remain 64 GiB free shared memory, 96 GiB available RAM,
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
| `6d9b14c70b154a5cb821` | Qualified smooth-rate, confirmation and head-rate CPU trials |
| `599a1030d1aaf214fd8c` | Exact parallel parameter validation; experimental mean transform remains deferred |

Current regression: **48 Rust and 62 Python tests pass**. Full-graph Rust/JAX
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
- **Smooth screen:** at 32k exposures, softness .01 with bias-scaled .03
  improves KL from 1.6495 to 1.6233, with essentially unchanged value MSE.
  Three-seed longer confirmation is active/queued; other optimizer choices
  do not share this improvement.
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
`runs/cpu-smooth-lane-v1`. Frozen running studies keep their original binaries. Parallel parameter
validation additionally reduces trained hard B1 median inference from 54.6 to
35.2 ms and B32 from 130.7 to 105.6 ms on 24 cores, with exact numerical
parity. Training timing gains are smaller and workload-sensitive.

1. Collect every optimizer, visual and smooth final validation; compare aligned
   positions and paired seeds without best-step selection.
2. Complete spatial/shuffled TPU cases and followups. Qualify/run the smaller
   CNN control and confirm its selected rate at the matching horizon.
3. Finish schedule confirmation, then choose a controlled stronger fly prior.
   Keep new adapter/dynamics/plasticity factors isolated and record revisions
   before dependent experiments. Online distillation follows prior selection.

`queue_cpu.py` freezes scripts/plans and waits for lane dependencies and
worker exits. The original affinity failure and zero-update TPU admission
failure remain recorded. Recovery studies change only operational attempt IDs
and failed dependencies. Shuffled seed 2 is training; smooth confirmation seed 1
has started, with seeds 2/3 waiting. `small-control-launch-v2` follows the
recovered TPU study. Keep large CPU qualification allocations out of this
TPU window; the original storage floors remain enforced.

Readout conditioning remains deferred after two B1 and one actual-B32
parameter-parity failures, with unchanged tolerances. Its isolated default
bypass preserves the baseline. Head-rate controls pass their specific gates
on the earlier qualified model and are queued. The offline
HTML guide now includes completed comparisons and interactive firing curves;
browser interactions, offline loading and widths 320–1,440 pass.

Local implementation and commits continue. Upstream remains at `772db92`:
automatic approval review rejected publication and requires explicit user
confirmation of `git@github.com:honglu2875/flygo.git`. No further push is attempted.
