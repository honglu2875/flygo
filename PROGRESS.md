# Progress

Updated **2026-09-15 00:55 UTC**. Fixed fly topology, learnable strengths.
The main comparison is **prediction FLOPs and labeled-position exposures**;
parameter counts, training/tuning cost and latency are reported separately.
No fly advantage has been established.

[Interfaces](README.md) · [Study contracts](RESEARCH.md) ·
[Results](docs/research-results.md) · [Qualification history](docs/qualification-history.md) ·
[Neuron groups](docs/group-study.md) · [Active phase: biological interfaces](docs/biological-interfaces.md).

| Milestone | Status | Remaining work |
|---|---|---|
| M0–M2: design, Go engine, expert pilot | Complete | Throughput improvements remain optional |
| M3: corpus | V0 complete; production stopped | Preserve stop markers; freeze later releases under separate contracts |
| M4–M5: fly execution and learning | Complete | Current Rust/Python/JAX interfaces qualified |
| M6: controlled studies | Visual confirmations and retinal comparison complete | Qualify optional output masks against the dense learned baseline; keep persistent state separate |
| M7: four-host TPU | Default path qualified; use paused | Spherical inputs, their nondefault epsilon, and smooth-rate TPU gates remain separate; no current CPU study depends on TPU |
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
| Supervised spherical inputs | K8, 2,129 individual motor readouts; history/current/neutral, 32k exposures per arm | All endpoints, full validation, paired family analysis and signal/cost probes complete |
| Spherical input confirmation | Same contract, paired seeds 2/3; adopt exploratory seed 1 | All six endpoints, full validations, motor/cost probes and paired analysis complete |
| Larger retinal allocation | One current 9×9 board per eye; paired seeds 1/2/3, K8, 32k exposures | All final checkpoints, replicas, full validations, diagnostics and paired analysis complete |

All studies retain their declared immutable sources, failed attempts and fixed
final horizons. V0 has 10,256 games / 1,000,201 positions: train 887,338,
validation 70,425, test 42,438. Final test labels remain outside tuning.

## Findings and engineering

- [Completed retinal comparison](docs/retinal-allocation-study.md): C minus B
  mean policy KL is **+.00996**, value MSE **−.04957**, teacher top-1 agreement
  **−1.007 percentage points**. The value gain comes from seed 1; seeds 2/3
  worsen. Conditional family intervals and seed variation are separate.
  Retain B's current-board montage for the next head study with fresh paired
  seeds and a dense control. C's varying motors number 496/305/272, with the
  same leading identities as B. Raw concentration alone is not a capacity test.
  Complete unpruned C arithmetic is **174.5–186.5M FLOPs at warm B1**;
  no new CNN or playing-strength advantage is established.
- [Learned readout contributions](docs/readout-roles-study.md): neutralizing the
  leading motor's visual component changes value by RMS **.525–.597**, and the
  top policy move on **4.69–15.23%** of training-probe positions, across three
  existing seeds. Policy logits reconstruct exactly; value error is at most
  1.2e-7. This measures decoder influence, not predictive usefulness. The
  report/source have verified replicas and two invariance/cancellation tests
  pass. Masked heads and per-motor conditioning remain separate pending factors.
- [Larger retinal map](docs/retinal-allocation-study.md): 97 Python tests pass;
  the original sampler rebuilds byte for byte, all native ports are unchanged,
  and each eye observes all 81 points at rank 81. Sampling conditioning worsens
  (about 14–15 → 27); no gain adjustment or map search follows this observation.
  Each actual seed passes CPU states/loss/all-gradient/three-update parity,
  identical initial arrays/sampler and exact fresh-process recovery.
- At fixed seed-1 weights, the larger angular image changes varying motors
  **242 → 251 initially**, **621 → 636 after B training**. The latter still has
  about **99.96%** of raw visual-response variance in DNg30 body 10123. This is
  a transfer probe, not learned-C capacity or an experiment with a larger Go
  board. [Evidence](docs/results/retinal-allocation-engineering-v1.json).
- [Readout design](docs/readout-roles-study.md): retain dense, jointly learned
  projections from all 2,129 motors to 82 policy logits and value. Raw variance
  concentration alone does not diagnose poor features. Dedicated value groups
  and soma-side policy blocks are optional separate factors. Motor soma labels
  are 1,065 left / 1,054 right / 10 midline; all motor `rootSide` fields are
  missing. No head mask or biological functional-role assignment is applied.
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
  Full-validation KL is **1.6674/1.6592/1.6520**, MSE **.6281/.8494/.8190**
  for history/current/neutral. History improves value in this seed; neither
  visual arm establishes a policy benefit. Family bootstrap intervals condition
  on these weights, not training-seed variation. About 98.4–98.9% of edge
  parameters change. [Results and provenance](docs/results/attachment-screen-v1.json).
- With nonvisual context held fixed, 415/621 motors have varying visual responses
  in the two trained image arms. **Over 99.9% of raw visual-response variance
  concentrates in DNg30, body ID 10123**. Other cells carry weaker distinct signals
  (unit-variance effective ranks 7.13/5.02). No action groups are fitted.
- New K8 warm B1 arithmetic is **173–182M FLOPs**, including the renderer;
  these models cannot inherit the old 8.3–8.9M comparison with the small CNN.
  Exact output dependencies allow about 22% fewer post-cache edge evaluations
  before zero skipping; the optional implementation is now measured below.
- [Confirmation gates](docs/results/attachment-confirmation-engineering-v1.json):
  both actual new seeds pass all three input modes at B32/K8, including states,
  losses, every gradient and three Adam updates at unchanged tolerances and
  epsilon 1e-6. Within each seed, all 31 initial checkpoint arrays, sampler and
  training contract match across modes. All six final endpoints are now available.
- Worker checkpoint relay passes eight regression tests and an actual 190 MB
  w1-to-w2 transfer through root's pipes. It verifies byte counts/hashes and
  publishes the destination receipt first. Interrupted receipts can retry;
  root holds no checkpoint payload. Existing study artifacts remain intact.
- [Confirmation results](docs/attachment-confirmation.md): in the two new seeds,
  history/current improve mean value MSE versus neutral by **.10320/.19931**,
  while policy KL worsens by **.04083/.03674**. History improves value in all
  three observed seeds; current versus history reverses ordering across seeds.
  Conditional family intervals and training-seed variation are reported separately.
  All nine endpoints retain their fixed 32k-exposure horizon; no final-test or
  playing-strength claim is made.
- The repeated motor probes remain concentrated, but dominant cell identity
  changes: DNg30 body 10123 in seed 1, DNg30 body 10237 in seed 2, DNpe018
  body 69173 in seed 3. The leading cell carries **92.9–99.99%** of raw visual
  variance across the six image-trained endpoints. Unit-variance ranks are
  4.26–7.13. Do not infer permanent action groups from one fitted seed.
  Recounted unpruned warm B1 arithmetic spans **171.2–184.9M FLOPs**.
- [Optional CPU dependency execution](docs/prediction-dependencies.md) passes
  **49 Rust / 95 Python tests** and all 12 full-CNS initial/final B1/B32 cases.
  Predictions are exact; history cases also preserve every gradient and three
  updates' parameter/moment arrays. The separate FP64 norm difference is at
  most 1.78e-15. Balanced same-lane timing measures **14.7%/12.6%** median
  warm latency reductions at B1/B32 versus the same binary with pruning off.
  Cold preparation costs more. Default inference/training remain unchanged;
  full traces/persistent state require full execution. A pruned FLOP ledger
  and actual TPU qualification remain separate.

## Runtime and next work

At **23:09 UTC**, all 32 generation workers were healthy and had published
**202,496 games**. Stop markers subsequently appeared on all four hosts at
about **23:46 UTC**; the generation processes have exited. Preserve those
markers. Some terminal status files still say generating, so actual `/proc`
checks take precedence. The frozen corpus remains available for research.
Keep the 64 GiB free-filesystem floor, 96 GiB available-RAM
floor and 100 GiB own-file cap, including reservations. Runtime data and
checkpoints live in volatile `/dev/shm/flygo`.
The 18 biological-probe artifacts/source files (about 256 MB) have verified
copies on w0 and w1; no existing data or checkpoints were removed.
Each completed embedding pilot has 11 hash-verified files (about 269 MB),
including its checkpoint: learned on w2/w0, frozen on w3/w0. Exact restoration
and the next sampler batch/update pass using a fresh model in the same process.
These representation checkpoints do not implement a Go policy. All copies
remain volatile; no existing files were removed to make room.

The three attachment trainers and subsequent full validations completed on
workers 1–3, pinned to `32–55`, with 32,000 exposures per arm. Their initial and
final checkpoints (about 190 MB each) have verified copies on the owner and w0.
The 22 signal/overlap/arithmetic files and six final analysis/source/status files
have verified copies on w0/w1. The main node uses about 99.0 GiB of its own-file
budget; reserve actual known transfer sizes and preserve
the cap when placing further artifacts. TPU use is paused pending fresh
coordination with the user; current training and qualification use CPU only.

All six [confirmation learners](docs/attachment-confirmation.md) completed
1,000 updates on `32–55` and `92–115` of workers 1–3. Their initial and final
checkpoints have verified copies on another worker; root stores no new
checkpoint payload. Full validation and subsequent motor/arithmetic diagnostics
have completed for every run. No TPU is used or reserved.
The candidate CPU runtime and completed qualification, timing, validation,
motor and analysis evidence—1,427 files, about 171 MB—have verified copies on
w0/w1. Existing files were checked before transferring missing entries;
resource limits and older artifacts were preserved.

**TPU queue:** nothing urgent or blocking. The default hard-rate model already
passes four-host checks. Before moving the newer spherical model to TPU,
qualify its K8/2,129-readout input contract and epsilon 1e-6, including gradients,
updates and fresh-process recovery. Then measure end-to-end larger-batch
throughput, including encoding, loading and communication, under a separate
batch/exposure contract. Smooth firing and later persistent-state models need
their own gates if selected. Request a coordinated TPU window only when a
specific experiment justifies it; current attachment analysis stays on CPU.

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
visual control complete under the [registered attachment contract](docs/attachment-study.md).
The registered paired seeds 2/3 confirm a history-versus-neutral value signal
at the same 32k horizon, with no policy improvement. Full validation and
motor/cost probes are complete. The larger current-board comparison is now
complete too: `retinal-large-seed{1,2,3}-v1` ran on workers 1–3, each pinned to
`32–55`, at exactly 1,000 updates. All learners and followup workers have exited,
and the paired analysis reproduces the adopted B metrics and family summaries.
Initial/final checkpoints have verified second-worker copies. The 78-file
evaluation, signal, arithmetic and analysis bundle has verified copies on
w0/w1, alongside the 1,048-file map/runtime/qualification bundle (about 87 MB).
Root holds no new checkpoint payload. The original immutable learner and dense
learned heads were retained. [Launch evidence](docs/results/retinal-allocation-launch-v1.json)
and [completed results](docs/results/retinal-allocation-v1.json) remain linked.
Persistent state remains planned. The larger map's mixed outcome leads us to
retain B for the next output comparison; this validation-informed decision
does not change completed contracts or launch persistent training.
Captures/older history through nonvisual source nodes remain a separate input
alternative. Source-graph degree, annotation and outgoing-path audits are gates.

**Readout revision:** the user explicitly permits learned combinations of motor
signals for this artificial task. The existing all-motor linear policy/value
heads remain the baseline. A concentrated cell or group feeding value, and
left/right policy blocks, are optional separate attachment studies. Their
implementation and qualification remain pending; the retinal study does not
apply them. [Design and anatomical-side audit](docs/readout-roles-study.md).

Attachment, execution and optimization are separate research lines. The pending
hard value-head rate confirmation, physiology/sign changes and
[regrowth](docs/structural-plasticity.md) keep separate contracts. No edge was
added. Persistent models require a history-matched conventional control before
claiming an equi-FLOP algorithmic advantage.

The authorized publication destination is `master` at
`git@github.com:honglu2875/flygo.git`. Deployment and artifact replication
preserve prior checkpoints and immutable sources.
