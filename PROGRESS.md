# Progress

Updated **2026-09-14 09:14 UTC**. Fixed fly topology, learnable strengths.
The main comparison is **prediction FLOPs and labeled-position exposures**;
parameter counts, training/tuning cost and latency are reported separately.
No fly advantage has been established.

[Interfaces](README.md) · [Study contracts](RESEARCH.md) ·
[Results](docs/research-results.md) · [Qualification history](docs/qualification-history.md) ·
[Next phase: neuron groups](docs/group-study.md).

| Milestone | Status | Remaining work |
|---|---|---|
| M0–M2: design, Go engine, expert pilot | Complete | Throughput improvements remain optional |
| M3: corpus | V0 complete; production continues | Freeze later releases under separate contracts |
| M4–M5: fly execution and learning | Complete | Current Rust/Python/JAX interfaces qualified |
| M6: controlled studies | Current registered batch complete | Use the results to register the next isolated hypotheses |
| M7: four-host TPU | Qualified | Nondefault epsilon and smooth-rate TPU gates remain separate |
| M8: online refinement | Pending useful prior | Prior/PUCT/Gumbel interfaces and evaluation panels work |
| Group study | Planned; descriptive audit complete | Anatomy → connectivity/dynamics → execution layout → group rules |

## Studies

| Study | Contract | Status |
|---|---|---|
| Large CNN control | ~126.7M nominal prediction FLOPs; 1,048,576 confirmation exposures | Three seeds, full validation and all panels complete |
| Small CNN control | 8.362M nominal FLOPs vs fly counted warm B1 means 8.31–8.87M; same exposures | Qualified shape, rate screen, three seeds and all panels complete |
| CPU optimizer confirmation | Three settings × three seeds; 128k exposures | Complete |
| Schedules | Warmup/cosine/constant; three paired seeds, 128k exposures | Complete |
| Head rates and epsilon | Separate seed-1 screens; 32k exposures | Complete; keep epsilon 1e-8 |
| Spatial input | Baseline/spatial/shuffled; three seeds, 1,048,576 exposures | Complete, including the operational retry |
| Visual readout | Three seed-1 variants, 128k exposures | Complete |
| Smooth firing | Softness .01, rate .03, bias multiplier .01; three seeds, 128k exposures | All training, full validations and paired analysis complete |
| Mean conditioning | Fixed-topology readout transform | Deferred after full-update parity failure; no scientific training |

All studies retain their declared immutable sources, failed attempts and fixed
final horizons. V0 has 10,256 games / 1,000,201 positions: train 887,338,
validation 70,425, test 42,438. Final test labels remain outside tuning.

## Findings and engineering

- Small CNN KL is **.8929–.9194**, versus fly **1.5037–1.5054**. Common-opening
  prior panels give **89/96 wins versus 0/96**, against the declared early
  KataGo checkpoint. These panels are not Elo. The large CNN also leads.
- Warmup improves mean KL by .01067 and MSE by .02749 at 128k exposures.
  Spatial attachment gives a small policy benefit; value changes are unresolved.
  Hard value-head rate scaling helps its short screen; the smooth interaction
  is worse. Larger epsilon fails the loss-based screen.
- Smooth .01 improves KL in all three observed seeds, by **.02689** on average
  at 128k exposures. Mean value change remains unresolved. More active edges
  can increase work; this is not yet an equi-FLOP or playing-strength benefit.
- Same-lane CPU B1 medians: fly **23.73 ms**, large CNN **2.43 ms**, small CNN
  **1.30 ms**. Backend overhead and memory work remain separate from FLOPs.
- Streaming prediction reduces the K32/B128 probe's peak RSS from **4.005 GB
  to 1.435 GB**, with exact outputs and no resolved speedup. A source-driven
  sparse-kernel candidate is slower and was removed; its patch/results remain.
- Current source **9058055f34adfea1b7bd** passes **48 Rust / 72 Python tests**.
  It also passes actual four-host TPU states/loss/all-gradient/three-update
  checks and exact fresh-process checkpoint/sampler continuation at default
  hard rates and epsilon. [Evidence](docs/results/current-tpu-default-v1.json).
- Streaming continuation reproduces every checkpoint array and sampler, but
  the extra exact diagnostic-norm check differs in the last FP64 bits. Those
  failed records remain. No numerical tolerance or storage limit was relaxed.
- Missing-only immutable bundle replication recovers pending deployments
  without retransferring existing feature data. Conflicts remain errors.

## Runtime and next work

At **09:10 UTC**, all 32 generation workers were healthy and had published
**105,902 games**. Free shared memory: **104–172 GiB**; available RAM:
**283–354 GiB**. Keep the 64 GiB free-filesystem floor, 96 GiB available-RAM
floor and 100 GiB own-file cap, including reservations. Runtime data and
checkpoints live in volatile `/dev/shm/flygo`.

Generation uses 64 pinned physical cores per host: `0–31,60–91`.
Research lanes are `32–55` and `92–115`; TPU/development uses spare cores.
Preserve production, checkpoint replicators and owned SSH keepalives.

The current registered studies are closed. Follow the deliberate
[group-study milestones](docs/group-study.md); a useful next optimizer test is
multi-seed confirmation of the isolated hard value-head rate screen.
The baseline already shares
leak/bias by cell type. Dense groups motivate hypotheses, not an established
functional advantage. [Regrowth](docs/structural-plasticity.md) is a proposed
later relaxation; no connection has been added.

Completed changes are committed locally. Publication remains blocked by automatic approval
review, which requires explicit confirmation of
`git@github.com:honglu2875/flygo.git`. Review also rejected removal of retired
checkpoint replicas after copying them to another host's volatile RAM.
No archive checkpoint was moved or removed; deployment recovered without it.
