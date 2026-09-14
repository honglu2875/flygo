# Progress

Updated **2026-09-14 18:34 UTC**. Fixed fly topology, learnable strengths.
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
| M6: controlled studies | Current registered batch complete | Use the results to register the next isolated hypotheses |
| M7: four-host TPU | Qualified | Nondefault epsilon and smooth-rate TPU gates remain separate |
| M8: online refinement | Pending useful prior | Prior/PUCT/Gumbel interfaces and evaluation panels work |
| Group study | Spherical adapter and CPU embedding pilot qualified; no representation benefit observed | Calibrate signal/optimization → substantial pretraining → response validation → functional readouts |

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

## Runtime and next work

At **13:11 UTC**, all 32 generation workers were healthy and had published
**133,247 games**. Free shared memory: **103–171 GiB**; available RAM:
**282–353 GiB**. Keep the 64 GiB free-filesystem floor, 96 GiB available-RAM
floor and 100 GiB own-file cap, including reservations. Runtime data and
checkpoints live in volatile `/dev/shm/flygo`.
The 18 biological-probe artifacts/source files (about 256 MB) have verified
copies on w0 and w1; no existing data or checkpoints were removed.
Each completed embedding pilot has 11 hash-verified files (about 269 MB),
including its checkpoint: learned on w2/w0, frozen on w3/w0. Exact restoration
and the next sampler batch/update pass using a fresh model in the same process.
These representation checkpoints do not implement a Go policy. All copies
remain volatile; no existing files were removed to make room.

Generation uses 64 pinned physical cores per host: `0–31,60–91`.
Research lanes are `32–55` and `92–115`; TPU/development uses spare cores.
Preserve production, checkpoint replicators and owned SSH keepalives.

The current registered studies are closed. **Plan revision:** the user's
proposal now puts [contrastive motor-embedding learning](docs/embedding-study.md)
before response-guided groups within [G1–G3](docs/group-study.md). Random/prior
weights alone cannot establish the capacity of trained responses. B0 is complete;
**B1's coherent spherical adapter and the bounded B1.5 CPU pilot are complete.** The
engineered chart has full rank for all 324 historical board points; measured
optical registration remains a separate gate. Genuine 1–6-ply divergent replay
pairs share at least 20 prior plies and use one teacher/current-player frame.
The candidate miner and family holdout were revised before training after
data-admission failures; the criteria and failed attempts are recorded in the
study note. **Next:** measure neutral versus visually evoked response scales
and group gradients, then qualify training-only centering/scaling or a separate
adaptation intervention before extending pretraining. Preserve raw responses
and the frozen-circuit control; do not treat flat loss as a reason simply to
run longer. Longer history, temporal presentation, physiology changes and output
grouping are separate factors. The isolated hard value-head rate confirmation
remains a pending optimizer idea. The baseline already shares
leak/bias by cell type. Dense groups motivate hypotheses, not an established
functional advantage. [Regrowth](docs/structural-plasticity.md) is a proposed
later relaxation; no connection has been added.

**Deferred input study:** the user proposes the two most recent moves/states
through the two eyes, and older sequence/captures/komi/context through other
input nodes. Audit zero-in-degree candidates against sensory labels and source
filtering; freeze the exact encoding and matched controls before launch. This
does not replace the current signal/optimization gate.

The approved history through **ab4c7b9** is published to `master` at
`git@github.com:honglu2875/flygo.git`. This phase adds the spherical adapter,
embedding interface and its completed pilot records. Deployment and artifact
replication preserve prior checkpoints and immutable sources.
