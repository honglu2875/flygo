# Progress

Updated **2026-09-14 20:49 UTC**. Fixed fly topology, learnable strengths.
The main comparison is **prediction FLOPs and labeled-position exposures**;
parameter counts, training/tuning cost and latency are reported separately.
No fly advantage has been established.

[Interfaces](README.md) · [Study contracts](RESEARCH.md) ·
[Results](docs/research-results.md) · [Qualification history](docs/qualification-history.md) ·
[Neuron groups](docs/group-study.md) · [Active phase: biological interfaces](docs/biological-interfaces.md).

| Milestone | Status | Remaining work |
|---|---|---|
| M0–M2: design, Go engine, expert pilot | Complete | Throughput improvements remain optional |
| M3: corpus | V0 complete; production continues | Freeze later releases under separate contracts |
| M4–M5: fly execution and learning | Complete | Current Rust/Python/JAX interfaces qualified |
| M6: controlled studies | Earlier batch complete; supervised visual ablation running | Finish matched history/current/neutral endpoints before changing another factor |
| M7: four-host TPU | Qualified | Nondefault epsilon and smooth-rate TPU gates remain separate |
| M8: online refinement | Pending useful prior | Prior/PUCT/Gumbel interfaces and evaluation panels work |
| Group study | Spherical adapter and CPU embedding pilot qualified; no representation benefit observed | Isolate input/output attachment, persistent execution and optimization; measure signal throughout |

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
| Spherical motor embeddings | K8, 64 updates / 2,048 views per arm; learned versus frozen circuit | CPU pilot complete; both reach 50% branch ranking, with negligible circuit change |
| Supervised spherical inputs | K8, 2,129 individual motor readouts; history/current/neutral, 32k exposures per arm | CPU-qualified and running on workers 1–3; matched context and optimizer |

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
- Previously TPU-qualified source **9058055f34adfea1b7bd** passes **48 Rust / 72 Python tests**.
  It also passes actual four-host TPU states/loss/all-gradient/three-update
  checks and exact fresh-process checkpoint/sampler continuation at default
  hard rates and epsilon. [Evidence](docs/results/current-tpu-default-v1.json).
- Streaming continuation reproduces every checkpoint array and sampler, but
  the extra exact diagnostic-norm check differs in the last FP64 bits. Those
  failed records remain. No numerical tolerance or storage limit was relaxed.
- Missing-only immutable bundle replication recovers pending deployments
  without retransferring existing feature data. Conflicts remain errors.
- Biological interface audit: 652/730 qualified inferred visual columns
  left/right, and 2,129 motor/descending candidates. A 256-position training-only
  CPU pilot finds severe visual-to-motor attenuation at initialization. Its
  raw/conditioned response matrices and two explicit strength perturbations
  are descriptive; no action grouping or new scientific training was fitted.
  [Protocol and findings](docs/biological-interfaces.md).
- On those fixed probes, extending a K4-trained checkpoint to K8 sharpens
  predictions but worsens policy KL **1.28 → 13.92**. Confidence alone is not
  progress. This is a post-hoc training-input check, not a K4/K8 training study.
  Five diagnostic unit tests and the full-graph pilot pass.
- New CPU source **a64f480502a5b26d3fb7** passes **48 Rust / 84 Python tests**
  and full-CNS B32/K8 states, embedding gradients and three free-running Adam
  updates. The linear ranking objective needs epsilon **1e-4** for this gate;
  smaller tested values failed unchanged update tolerances. This is numerical
  qualification, not an optimizer recommendation or new TPU qualification.
- The coherent spherical bank has **540 training pairs / 39 families** and
  **94 probe pairs / 12 families**, with no training/probe visual overlap under
  D4. Both 64-update arms finish at **50%** pair accuracy (family macro **46.9%**).
  The learned arm changes only **1,759 / 15,270,273 edge parameters (0.0115%)**;
  response variation barely moves. Loss remains near ln(3) + ln(2). These
  weights are not substantially trained, so no functional groups are fitted.
  [Protocol, numerical gates and results](docs/embedding-study.md).
- [Doomfly source review](docs/doomfly-review.md): inferred planar eye viewports,
  four fixed descending-neuron controls, and experimental local dopamine
  plasticity; published checks do not establish learned survival. Our read-only
  comparison finds 312 R8→aMe12 edges with negative initial signs, including
  uncertain subtypes. Target-specific transmission is a candidate physiology
  audit; no sign change or training experiment was made from this finding.
- [Supervised attachment study](docs/attachment-study.md): 3,490 visual inputs
  plus 84 context features distributed over 11,530 nonvisual sensory cells.
  All 87 Python tests pass. Full-CNS B32/K8 states/losses/all-gradients/three
  Adam updates pass in every mode at epsilon 1e-6; the failed 1e-8 gate and
  its cancellation diagnosis remain recorded. The same epsilon is used in all
  arms, selected without validation. Initial checkpoints have identical weights,
  moments, ports and samplers, with exact direct/Go-interface B1/B32 predictions.
  At 8k exposures, KL is about 1.768/1.777/1.767 for history/current/neutral;
  these interim seed-1 results do not establish a visual benefit.

## Runtime and next work

At **20:03 UTC**, all 32 generation workers were healthy and had published
**181,064 games**. Free shared memory: **103–170 GiB**; available RAM:
**282–352 GiB**. Keep the 64 GiB free-filesystem floor, 96 GiB available-RAM
floor and 100 GiB own-file cap, including reservations. Runtime data and
checkpoints live in volatile `/dev/shm/flygo`.
The 18 biological-probe artifacts/source files (about 256 MB) have verified
copies on w0 and w1; no existing data or checkpoints were removed.
Each completed embedding pilot has 11 hash-verified files (about 269 MB),
including its checkpoint: learned on w2/w0, frozen on w3/w0. Exact restoration
and the next sampler batch/update pass using a fresh model in the same process.
These representation checkpoints do not implement a Go policy. All copies
remain volatile; no existing files were removed to make room.

The three attachment trainers are verified live on workers 1–3, pinned to
`32–55`, with 32,000 exposures planned per arm. Their initial checkpoints
(about 190 MB each) have verified copies on the owner and w0. The main node
is close to its own-file cap; reserve actual known transfer sizes and preserve
the cap when placing further artifacts. TPU use is paused pending fresh
coordination with the user; current training and qualification use CPU only.

Generation uses 64 pinned physical cores per host: `0–31,60–91`.
Research lanes are `32–55` and `92–115`; TPU/development uses spare cores.
Preserve production, checkpoint replicators and owned SSH keepalives.

The earlier registered studies are closed. B0, B1's coherent spherical adapter
and the bounded B1.5 embedding pilot are complete. Random/prior responses and
the minimally changed pilot weights cannot establish trained capacity. Keep
signal amplitude and gradient diagnostics before fitting functional groups;
flat contrastive loss does not justify simply extending that run.

**Plan revision:** the user's latest proposal gives both eyes the current
board and carries history in neural state. The current model resets between
positions. The [temporal vision design](docs/temporal-vision-study.md) orders
four supervised comparisons: historical montage, current board in unchanged
patches, one larger board per eye, then persistent state with the same input.
Keep outputs and optimizer fixed within these input/execution comparisons.
Trajectory training, consistent stone perspective and state-gradient parity
are required before testing learned memory. The first two inputs and a neutral
visual control are now running under the [registered attachment contract](docs/attachment-study.md).
Larger retinal allocation and persistent state remain planned.
Captures/older history through nonvisual source nodes remain a separate input
alternative. Source-graph degree, annotation and outgoing-path audits are gates.

Attachment, execution and optimization are separate research lines. The pending
hard value-head rate confirmation, physiology/sign changes and
[regrowth](docs/structural-plasticity.md) keep separate contracts. No edge was
added. Persistent models require a history-matched conventional control before
claiming an equi-FLOP algorithmic advantage.

The prior design milestone **f0ebcda** is published to `master` at
`git@github.com:honglu2875/flygo.git`. Deployment and artifact replication
preserve prior checkpoints and immutable sources.
