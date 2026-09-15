# Progress

Updated **2026-09-15 10:12 UTC**. Fixed fly topology, learnable strengths.
The main comparison is **prediction FLOPs and labeled-position exposures**;
parameter counts, training/tuning cost and latency are reported separately.
No advantage over the matched CNN has been established.
The smooth-rate study reaches full-validation policy KL **1.4866–1.4930**
(three-seed mean **1.4903**) at 128k exposures. The **1.5037–1.5054** fly
range in the matched CNN comparison refers to its separate hard-rate control;
the current spherical/readout screens use a shorter 32k-exposure contract.

[Interfaces](README.md) · [Study contracts](RESEARCH.md) ·
[Results](docs/research-results.md) · [Qualification history](docs/qualification-history.md) ·
[Neuron groups](docs/group-study.md) · [Active phase: biological interfaces](docs/biological-interfaces.md).

| Milestone | Status | Remaining work |
|---|---|---|
| M0–M2: design, Go engine, expert pilot | Complete | Throughput improvements remain optional |
| M3: corpus | V0 complete; production stopped | Preserve stop markers; freeze later releases under separate contracts |
| M4–M5: fly execution and learning | Complete | Current Rust/Python/JAX interfaces qualified |
| M6: controlled studies | Clipping qualification and exact legacy regression pass; six CPU learners running | Finish the six fixed endpoints, full validation and paired clipping analysis; keep dense and hold propagation fixed |
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
| External readout masks | Dense / five-cell value / soma-side / shuffled-side; fresh seeds 4/5/6 | All 12 discovery endpoints complete; subsequent soma policy confirmation fails |
| Readout confirmation | Dense / soma-side / same shuffled-side; fresh seeds 7/8/9, 32k exposures | All nine endpoints, replicas, full validations, audits and contrasts complete; retain dense |
| Motor learnability | Four residual decoders × three rates × three frozen cores; 131,072 head exposures per case | All 36 endpoints and replicas complete; raw mixing helps modestly, tested gating/high rates worsen validation |
| Motor convergence | Bias-only / standardized linear / gated linear × three ridge penalties × three frozen banks | All 27 fits, original-coordinate audits, paired analysis and second-worker copies complete |
| Signal and gradient flow | Existing dense seeds 4/5/6, initial/trained; 128 training views and 32 gradient views | All 48 recurrence and 75 VJP group checks pass; reports and 105 owner files replicated |
| Clipping scope | Global / per-parameter-group, fresh paired seeds 10/11/12, unchanged 32k-exposure contract | 12 numerical / six recovery / six exact regression cases pass; all six CPU trials advancing, initial replicas verified, followups queued |
| Research notebook | Public `quintic/go9x9` slice → full Rust model → plots → training → held-out KL | All cells execute on CPU; actual outputs included |
| Persistent state | Separate Rust core primitive and training trajectory audit | Ten core tests pass; model/Python/JAX integration and scientific training remain pending |

All studies retain their declared immutable sources, failed attempts and fixed
final horizons. V0 has 10,256 games / 1,000,201 positions: train 887,338,
validation 70,425, test 42,438. Final test labels remain outside tuning.

## Findings and engineering

- [Fresh-seed readout confirmation](docs/readout-confirmation.md) fails its
  declared policy conditions. Soma-minus-dense KL is **+.01874**, worsening
  in all three seeds; value MSE improves **−.02609**, also in all three.
  Soma policy is worse than shuffled in two seeds. All nine full validations,
  source/mask/decoder checks and paired contrasts are complete; retain dense.
  Dense/soma/shuffled means are **1.66656 / 1.68530 / 1.67766** KL. These short
  K8 studies do not inherit the earlier matched-CNN budget.
- [Signal and gradient audit](docs/signal-flow-study.md): initial motor visual
  variation is about **2,322×** smaller than total variation on the fixed
  probes. Trained motor activity is positive in **16.6–22.7%** of node-position
  pairs; one motor carries **97.2–99.9%** of summed visual variances. All motors
  have a two-to-five-hop path from an eye. Native gradients independently
  reproduce, but most edge Adam denominators are dominated by epsilon, while
  type-bias gradients dominate the norm. Eight focused tests pass; the summary
  reproduces byte for byte on another worker. Each audit took 212–227 s on
  four CPU cores; full arrays have verified second-worker copies.
- [Next optimizer contract](configs/group-clipping-study-v1.json): compare
  global and per-group clipping with dense heads, unchanged inputs, equations,
  rate, epsilon and horizon. This prioritization follows measured gradient
  scales and the failed readout confirmation. Rust/Python/JAX now expose the
  clipping scope and actual pre-update group norms/factors; checkpoints reject
  a mode mismatch. All 56 Rust and 121 Python checks pass, including seven
  clipping tests and the additional validation/analysis checks. The
  [full-circuit protocol](configs/group-clipping-numerics-v1.json) passes all
  12 numerical and six exact recovery cases on CPU. Source
  `1f19481174f3bc8e5449` is frozen; default global behavior also matches the
  previous binary exactly in all six native seed/rate cases. All six learners
  have verified CPU affinity and initial replicas; full validation, motor and
  FLOP probes are queued. The 456 qualification files have verified owner/peer
  archives. [Implementation and study](docs/group-clipping-study.md),
  [launch](configs/group-clipping-trials-v1.json) and
  [analysis](configs/group-clipping-analysis-v1.json). Isolated optimizer
  overhead and scientific conclusions remain pending. The first
  signal attempt failed only on an unavailable PyArrow import; its preserved
  recovery uses a lossless annotation export. TPU stays paused.
- [Research walkthrough](notebooks/flygo_research_walkthrough.ipynb): 512 training
  and 256 validation positions from 96 downloaded games, with disjoint opening
  families. It explores weights, both spherical eye maps and recurrent signal
  attenuation, then trains the full model for 64 B32 updates. The pinned public
  Hub sample reaches KL **1.90137**, from **2.03807**, in **387 s on four CPU
  cores**. This is an execution example on a different sample, not a new V0
  benchmark. [Setup](notebooks/README.md) and
  [execution record](docs/results/research-notebook-hf-v1.json).
- [Motor learnability](docs/motor-learnability-study.md): mean subset KL
  **1.68393 → 1.58499** with an additional raw linear residual at rate .03.
  Standardized top-128 gating at .003 fits training KL **1.0092** but gives
  validation **2.0042**. Much larger rates yield extremely confident, incorrect
  policies. Raw and standardized linear heads have the same function class;
  this screen does not establish a motor-information limit. The completed
  convergence followup separates optimization and overfitting. The completed
  circuit signal/gradient audit motivates isolating its optimizer
  interaction before a separate propagation change.
- [Complete readout screen](docs/readout-roles-study.md): soma-side minus dense
  mean KL is **−.01401**, MSE **−.06375**, with both losses improving in each of
  three paired seeds. Teacher top-1 falls **.200 pp**. Mean counted B1 FLOPs are
  essentially equal. Soma-side also improves policy KL against the frozen
  shuffled split in all three seeds; the value gain is shared by the generic
  split. The five-cell value arm retains its mixed tradeoff (KL −.03468,
  MSE +.03730). Dense remains the reference; the fresh-seed confirmation above
  rejects the provisional soma-side policy benefit.
  All 12 discovery endpoint and decoder audits pass; policy logits reconstruct exactly,
  value within 1.2e-7. Ten focused analysis tests pass. No CNN benefit is established.
- [Revised numerical checks](docs/results/readout-confirmation-numerics-v2.json):
  **18/18** numerical and **9/9** exact recovery cases pass. Independent head
  arithmetic reconstructs all captured policy gradients and Adam updates
  exactly. Tiny initial motor rounding is amplified by nearly cancelling
  gradients. The registered v2 checks align checkpoints for peak-rate
  transitions and separately run the actual warmup trajectory; learner source,
  tolerances and scientific settings are unchanged. The original two free
  constant-peak failures remain failed. The controller now binds both protocols
  and the recovery evidence to the actual jobs; eight focused tests pass. All
  nine cases pass deployment admission, and the six dense/soma learners have
  completed 1,000 updates with verified initial/final replicas. All nine full
  validations and final analyses are now complete. The first followup launch stopped before starting workers because
  host-specific Python bytecode conflicted during immutable replication.
  Recovery v2 excludes caches from new bundles and starts six validation waiters;
  learner sources, old artifacts and scientific settings are preserved.
- [Motor convergence](docs/motor-convergence-study.md): all **27/27** numerical
  optimization-error bounds pass. At ridge .01, gated validation-subset KL is
  **1.54872**, versus original **1.68393** and paired bias-only **1.65879**.
  Weakly regularized gating fits mean training KL **.68778** but validation
  worsens to **2.07265**. This identifies optimization and overfitting concerns,
  not a motor-information ceiling or a new CNN comparison. All **97.42M** solver
  label exposures are reported. Four objective and three analysis tests pass.
  The 153 owner files have verified second-worker copies (about 99.5 MB per
  archive); root stores only small reports. Each nine-fit suite took 648–697 s
  on four pinned CPU cores. [All results](docs/results/motor-convergence-v1.json).
- The shuffled-side controls and followups are complete. The queue's first startup failed because the entry
  filename `queue.py` shadowed Python's standard library. Recovery v3 changes
  that filename to `worker.py`, then failed CPU discovery under housekeeping
  affinity. Direct deployment from the identical frozen helpers and plan
  restores discovery affinity before pinning. Neither failure started learners;
  both attempts and scientific plans are preserved. [Final evidence](docs/results/readout-confirmation-v2.json).
  No TPU is initialized or reserved.
- [State core and trajectory audit](docs/temporal-vision-study.md): explicit
  Rust initial/final states and their gradients pass ten core tests, including
  finite differences and split-trajectory composition. The new core is not in
  the installed training binary. Replay checks pass for 128 training families /
  13,085 positions. All 854 occupied nonterminal passes reverse the current-player
  image while preserving absolute stone colors. Qualify an absolute-color reset
  control separately before attributing any future gain to persistent memory.
- [External head qualification](docs/readout-roles-study.md): all **12/12 CPU
  numerical and 12/12 recovery gates pass**, with identical initial arrays and
  samplers across four arms per seed. Coverage is **50 Rust / 110 Python core
  tests**, plus three deployment-evidence tests. Fixed masks cannot contribute
  disabled gradients/moments and are embedded in portable Go checkpoints.
  The accepted-target-mass derivative, deterministic norm and FP64 head/loss
  reductions correct demonstrated numerical problems. Prior failed cohorts
  remain recorded; rates, epsilon and tolerances were not relaxed.
  Fixed-weight CPU latency rises about **2.3% / 3.9% at B1/B32** in the bounded
  benchmark. Two paired 32k-exposure waves use the common corrected source and
  fresh dense controls. TPU remains paused; no learnability gain is claimed.
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
  pass. Masked heads are qualified for the registered screen; per-motor conditioning
  remains a separate pending factor.
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
  missing. All four comparisons are complete; fresh-seed confirmation is
  registered. Anatomical labels remain proxies for functional roles.
- Small CNN KL is **.8929–.9194**, versus fly **1.5037–1.5054**. Common-opening
  prior panels give **89/96 wins versus 0/96**, against the declared early
  KataGo checkpoint. These panels are not Elo. The large CNN also leads.
- Warmup improves mean KL by .01067 and MSE by .02749 at 128k exposures.
  Spatial attachment gives a small policy benefit; value changes are unresolved.
  Hard value-head rate scaling helps its short screen; the smooth interaction
  is worse. Larger epsilon fails the loss-based screen.
- Smooth .01 improves KL in all three observed seeds, by **.02689** on average
  at 128k exposures, reaching **1.493042 / 1.486574 / 1.491410** on full
  validation (mean **1.490342**). Mean value change remains unresolved. More
  active edges can increase work; this is not yet an equi-FLOP or playing-strength benefit.
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

Both original readout waves have completed all learners, full validations,
motor probes, counts and owner audits. All initial and final checkpoints have
verified worker recovery copies. The [full report](docs/results/readout-roles-v1.json)
contains all four declared contrasts and decoder interventions. The earlier
second-wave inventory-affinity failure and recovery remain retained.

The [fresh confirmation](configs/readout-confirmation-analysis-v1.json) is
registered before any new scientific update: dense / soma-side / shuffled-side,
seeds 7/8/9, same inputs, source, optimizer and fixed 32k-exposure horizon.
Actual numerical checks finish **7/9**, with exact recovery passing for all
seven qualified cases. Soma-side seeds 7/8 fail the unchanged gates; no
scientific confirmation learner has started. Source and owner failure records
have verified replicas. A bounded FP64-head oracle passes 2/3 soma-side cases
but does not clear seed 8. Completed cross-evaluation at identical weights
localizes most of its failing logit discrepancy to policy-weight drift. The
next numerical task is to audit the gradient/Adam rounding that creates this
drift; no replacement gate is accepted yet.
The first transfer failed before launching due to host-local Python bytecode;
the retained recovery excludes caches while verifying every source artifact.
No cache or experiment data was overwritten. Training will use the existing
24-core lanes in two waves. Root holds no new training checkpoint payload.
TPU remains paused. Explicit-state core engineering has its own evidence and
has not changed the scientific learner.

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
implementation, numerical and recovery qualifications are complete. The new
screen uses source `2ba439e309df1e902b58` for every arm, with fresh dense controls
and the unchanged B current-board montage. The 379-file engineering bundle has
verified metadata/source copies on w0/w1. The retinal study remains unchanged.
[Design, numerical corrections and registration](docs/readout-roles-study.md).

Attachment, execution and optimization are separate research lines. The pending
hard value-head rate confirmation, physiology/sign changes and
[regrowth](docs/structural-plasticity.md) keep separate contracts. No edge was
added. Persistent models require a history-matched conventional control before
claiming an equi-FLOP algorithmic advantage.

The authorized publication destination is `master` at
`git@github.com:honglu2875/flygo.git`. Deployment and artifact replication
preserve prior checkpoints and immutable sources.
