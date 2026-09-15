# Learnable policy and value readouts

The user's readout proposal treats Go actions as combinations of motor signals.
The current baseline already has a jointly learned dense projection from all
2,129 individual motor features to 82 policy logits, including pass, and a
separate learned projection to a scalar before tanh. Each motor's readout gain
also learns. There is no requirement that one biological motor represent one
Go action. Keep dense learned mixing as the baseline.
For 9×9, the policy matrix has shape 82×2,129. Both heads learn jointly with
the circuit through the existing policy/value losses and Adam updates.

For raw motor rates r, diagonal learned gains D, policy weights P, value
weights v and biases:

```
policy_logits = P D r + b_policy
value         = tanh(v^T D r + b_value)
```

Raw activation variance alone does not establish whether these features are
adequate for either task. A high-variance neuron could supply a global value
signal while smaller, distinct signals support policy. The user proposes
deliberately assigning such a neuron, or a small group, to the value readout.
That is an output-attachment experiment, separate from changing retinal inputs,
neuronal equations or optimization.

## Proposed factors

1. **Selective value readout.** Restrict the trainable value projection to a
   declared candidate group, while retaining the dense trainable policy head.
   Candidate discovery uses training-position responses only. Freeze the
   selection and initialization rule before scientific training; do not pick
   an endpoint or cell using its validation score. The dominant identity varies
   across the existing seeds, so a universal single-cell role is unproven.
   If discovery uses a pretrained circuit, report its exposures and computation;
   a matched comparison cannot treat that pretraining as free.
2. **Side-aware policy readout.** Keep learned coefficients within anatomically
   defined left/right blocks. Left-associated motors supply board columns 0–3;
   right-associated motors supply columns 5–8. Column 4 and pass use both sides;
   unassigned cells remain shared. Audit anatomical side fields and their
   uncertainty first. Soma side is an explicit proxy, not proof of receptive
   field or motor output laterality. A shuffled-side control can distinguish
   anatomical organization from a generic projection constraint.

The [annotation audit](results/retinal-allocation-engineering-v1.json) finds
no populated `rootSide` field among these 2,129 motors. `somaSide` is populated:

| Motor superclass | Left soma | Right soma | Midline |
|---|---:|---:|---:|
| Descending | 656 | 648 | 10 |
| Central-brain motor | 54 | 53 | 0 |
| Ventral-nerve-cord motor | 355 | 353 | 0 |
| Total | 1,065 | 1,054 | 10 |

A soma-side policy experiment would name that proxy explicitly and retain the
ten midline cells as shared features. These counts do not establish functional
laterality or require restricting the dense decoder.

Test these factors separately against the dense baseline on the same frozen
input mapping, K, recurrence, optimizer and exposure horizon. Only consider a
combined head after the individual comparisons. All allowed projection weights
and biases remain learnable. Preserve every CNS neuron and edge; head masks
would constrain the external decoder, not recurrent connectivity.

An exploratory diagnostic may reconstruct the existing learned heads from
motor responses and retained coefficients, examine value versus relative-action
contributions, and temporarily remove a neuron's readout contribution. A common
shift to all legal action logits leaves policy probabilities unchanged. Such
fixed-checkpoint diagnostics help define candidates; they do not establish that
a newly trained selective decoder is better.

Full state/loss/all-gradient/optimizer parity, checkpoint continuation and actual
decoder arithmetic must be qualified before training a new head. Count its
work in the complete prediction budget. Compare fixed-horizon validation and
later controlled play; anatomical resemblance alone is not the acceptance
criterion. TPU remains paused. All twelve endpoints and the four declared
contrasts are complete; fresh-seed confirmation is registered below.

## What the existing learned projections use

The [three-seed decoder diagnostic](results/readout-contributions-v1.json)
uses the existing B current-input checkpoints and the same 256 training-family
positions as the motor probes. It reads retained responses and coefficients;
no neural updates or new target labels are used. Reconstructing the native head
in its canonical FP32 summation order reproduces every policy logit exactly;
value differences are at most 1.2e-7 from the tanh implementation.

For each seed, select the motor with the largest across-position variance in
`r_current - r_neutral`. At the decoder only, replace that motor's current
response with its neutral-eye response at the same context. Leave other motor
features unchanged. This intervention gives:

| Seed | Leading motor | Value RMS change | Policy top-choice changes | Mean policy KL |
|---|---|---:|---:|---:|
| 1 | DNg30, body 10123 | .538 | 7.81% | .01961 |
| 2 | DNg30, body 10237 | .525 | 4.69% | .01562 |
| 3 | DNpe018, body 69173 | .597 | 15.23% | .02309 |

Values lie in [-1,1]. Policy KL compares the original distribution with the
intervened distribution over the same legal moves. These results demonstrate
substantial value influence and measurable policy influence. They do not score
either prediction against the teacher or establish a biological value function.

The diagnostic also centers each motor's visual response across positions and
projects that variation through the learned heads. For policy it removes each
position's mean legal logit, since a common shift has no policy effect. Removing
the leading motor leaves relative-action signal-energy ratios of 6.97e-6,
2.33e-7 and .00356, and pre-tanh value-energy ratios of 4.97e-9, 1.78e-9 and
.000129. The smaller motor signals have not been substantially amplified by
these fitted linear heads. Ratios describe the residual of a sum; they are not
additive variance allocations and can exceed one when contributions cancel.

The [diagnostic plan](../configs/readout-contributions-v1.json) declares top-one
and top-eight response-ranked subsets; both are retained in the report. Tests
cover invariance to common legal-logit shifts and cancellation. Raw responses,
source checkpoints, diagnostic code and report identities remain linked. This
supports testing deliberate value routing while retaining the dense policy
decoder as its paired control. It does not justify discarding the dense baseline.

## Qualified implementation

Use binary masks on the external policy and value matrices. A disabled
coefficient contributes neither to the forward result nor to any gradient,
optimizer moment, or clipping norm. Preserve the original recurrent neuron and
edge identities. An all-enabled mask must reproduce the original dense model's
states, losses, gradients and updates. The mask and its provenance belong in the
checkpoint contract; recovering with a different mask must fail.

The three proposed alternatives remain separate:

- **Value candidate group:** allow only the complete annotated DNg30 and DNpe018
  types to feed value; keep policy dense. The current motor inventory contains
  five such cells: DNg30 bodies 10123/10237 and DNpe018 bodies
  69173/113166/165031. This includes every member of the two types, rather than
  choosing one seed's winning body ID. It is a response-motivated candidate,
  not an established physiological value circuit.
- **Soma-side policy:** the previously defined left/right/midline blocks, with
  dense value. It keeps 98,294 of the 174,578 policy coefficients enabled.
- **Shuffled-side policy:** permute left/right assignments within each motor
  superclass using one frozen seed, preserving each class's side counts and
  the ten shared midline identities. It has the same per-action fan-in as the
  soma-side mask. This tests whether the anatomical assignment adds value
  beyond the block constraint itself.

Use a newly trained dense control and new paired head/sampler seeds 4/5/6 for
the screen. Seeds 1/2/3 informed candidate discovery. Record those earlier
exposures as development cost. Keep all enabled initial coefficients at their
original values, with no fan-in gain correction. Use the B current-board
montage for every arm (attachment SHA-256
`9ce4b1d99e499871a540d06f06dccc563e2e9c77e9ad277a6f2399a04345e087`).
The [completed retinal study](retinal-allocation-study.md) found no consistent
C improvement across seeds and both losses; this validation-informed input
choice is recorded before new head training. Do not change inputs inside a
head comparison. Initial core strengths remain
identical across seeds. Fix K8, B32, 1,000 updates, the same losses and the
qualified epsilon/rate schedule unless a separate numerical gate prevents it.

The binary masks are implemented in Rust and the independent JAX reference.
They are prepared from the checked annotation table under the
[frozen preparation contract](../configs/readout-mask-preparation-v1.json).
Each checkpoint contains its mask and provenance. Restoration rejects different
masks and nonzero moments on disabled coefficients before mutating the model.
The all-enabled artifact uses Rust's original dense execution path. The Go
player restores the mask directly from its checkpoint.

Rust visits only enabled decoder coefficients; the arithmetic ledger counts
those products. The JAX reference applies a binary mask to a dense matrix, so
its enabled coefficient count is not a claim that the compiler skips that work.
All CNS neurons and connections remain present. Both scientific waves were
registered before launch; all four contrasts are complete. Subsequent persistent-state
and optimization experiments retain their own contracts.

## Numerical qualification

[All twelve full-CNS cases pass](results/readout-mask-engineering-v1.json),
covering four arms and seeds 4/5/6 at B32/K8. States, losses, every gradient
and three free-running Adam updates satisfy the unchanged tolerances. All
twelve fresh-process recovery checks pass too, including complete B1/B32
Go-player predictions, the next sampler batch/update and counted arithmetic.
Within each seed, all four arms have the same 31 initial parameter, moment and
port arrays and the same sampler. All-enabled masks match unmasked controls
exactly through four updates. Unit coverage is 50 Rust tests, 110 Python core
tests and three deployment-evidence tests; the direct Rust/Python Go test was
run separately after configuring its executable.

The first implementation passes 49 Rust and 108 Python tests, including
independent masked gradients, zero disabled moments, portable Go inference and
fresh recovery. Its full-CNS B32/K8 checks pass 11 of 12 seed/arm combinations.
Seed 5's shuffled-side arm misses the existing forward-logit tolerance after
one Adam update. Cross-evaluation with identical parameters locates the main
disagreement in optimizer-amplified parameter drift, rather than the final
head sum at those failing cells. This failed run is retained.

A separate all-enabled/control check finds identical parameter and moment
arrays but a last-bit difference in the reported FP64 norm. Two shared
numerical corrections follow:

- The accepted FP32 teacher targets need not sum to exactly one. For target
  mass `s = sum(q)`, cross-entropy has derivative `s * softmax(logits) - q`.
  Rust previously assumed `s = 1`. A regression using accepted target mass
  .99992 reproduces the error before the correction.
- Global gradient norms now reduce fixed indexed chunks and combine their
  FP64 sums in a fixed order, independent of Rayon work stealing. A regression
  verifies identical norm bits at one, two and seven worker threads.

That intermediate candidate passes 50 Rust and 109 Python tests, but only
nine of the twelve actual full-CNS gates. A double-precision head oracle finds
small rounding errors in nearly cancelling head gradients that Adam amplifies.
The final candidate accumulates task-head products, head derivatives and policy
normalization in FP64, rounding outputs and gradients back to FP32. The
recurrent circuit, stored parameters and Adam moments stay FP32. JAX's
log-softmax uses a stopped-gradient common shift, also used by its installed
standard implementation. Its independent reference remains FP32/highest.
A cancellation regression fails under the earlier decoder (zero instead of
about .015) and passes with the corrected accumulation.

The qualified source is `2ba439e309df1e902b58`, numerical runtime
`rust-fp32-circuit-f64-head-reductions-v3`. Rates, epsilon, masks and comparison
tolerances stay fixed through these engineering corrections. Training
continuation now rejects a checkpoint from a different numerical runtime;
inference can still load its weights. Old frozen scientific runs retain their
original sources and results. Every new arm uses the same corrected runtime
and a fresh dense control.

The [fixed-weight CPU benchmark](results/readout-head-precision-v1.json) uses
four fresh processes per source/batch size, in balanced ABBA/BAAB order, with
ten warm predictions per process. Median latency changes **71.93 → 73.57 ms**
at B1 and **302.15 → 313.95 ms** at B32: about **2.3% / 3.9%** overhead.
Peak RSS is similar. These are step-3 engineering weights and include the
renderer; they are not trained-endpoint or general hardware performance claims.
The 379-file metadata/source/failure bundle has verified copies on w0/w1;
the engineering checkpoint payloads remain on their worker owners.

## Registered learning screen

[Wave 1](../configs/readout-roles-wave1-v1.json) compares dense and five-cell
value heads; [wave 2](../configs/readout-roles-wave2-v1.json) compares soma-side
and shuffled-side policies. Each arm has new paired seeds 4/5/6, K8, B32,
1,000 updates / 32,000 labeled exposures, rate .03, 100-update warmup,
epsilon 1e-6, clipping 1 and bias-rate multiplier .01. No gain correction is
applied to the enabled coefficients. Both waves were registered before launch.
Full final-horizon validation, motor probes and prediction arithmetic precede
the paired analysis. Final test labels remain closed.

The [first launch audit](results/readout-roles-launch-v1.json) matched actual
initial arrays, samplers and contracts, all 150 thread affinities, and six
verified initial checkpoint replicas. All six first-wave endpoints and their
validation, motor probes and arithmetic counts are now complete. Both initial
and final checkpoints have verified recovery copies; an owner-side audit checks
the same mask, unchanged disabled coefficients and zero disabled moments at
both endpoints.

The [second launch audit](results/readout-roles-wave2-launch-v1.json) verified
the same gates on all six soma-side/shuffled-side learners at updates
230–250. These learners have since completed all 1,000 updates, full validation,
motor probes, arithmetic counts and endpoint audits. Both initial and final
checkpoints have verified worker replicas. Its first queue attempt stopped before creating a learner: the
housekeeping affinity hid the host's CPU inventory from the planner. The
retained recovery restores inventory visibility before the planner pins itself.
The scientific source, settings and worker lanes did not change.

| External head | Enabled policy coefficients | Enabled value coefficients | Decoder FLOPs per prediction |
|---|---:|---:|---:|
| Dense control | 174,578 | 2,129 | 353,414 |
| Five-cell value | 174,578 | 5 | 349,166 |
| Soma-side policy | 98,294 | 2,129 | 200,846 |
| Shuffled-side policy | 98,294 | 2,129 | 200,846 |

These counts describe just the external decoder. Recurrent activity can change
as training diverges, so the complete prediction budget must be recounted at
each endpoint. The screen cannot inherit an earlier CNN FLOP comparison.

## First completed contrast: selective value versus dense

The [paired first-wave report](results/readout-roles-wave1-v1.json) includes
every endpoint at 32,000 labeled exposures. Full validation has 70,425 positions
from 254 opening families. The [analysis rules](../configs/readout-roles-analysis-v1.json)
were frozen after training finished and before inspecting its new validation
metrics. Training contracts were registered before launch. All four contrasts
now appear in the [complete report](results/readout-roles-v1.json), including
the full decoder interventions and second-wave results.

| Value-group minus dense | Seed 4 | Seed 5 | Seed 6 | Mean |
|---|---:|---:|---:|---:|
| Policy KL | −.06047 | +.00665 | −.05020 | **−.03468** |
| Value MSE | +.15469 | −.00126 | −.04154 | **+.03730** |
| Teacher top-1 agreement, percentage points | +1.127 | +.643 | −.108 | **+.554** |

Mean policy KL is 1.69163 for dense and 1.65696 for the value group; value MSE
is .64922 and .68652. This is a policy/value tradeoff with inconsistent seed
ordering. Retain dense as the default. Restricting the value decoder also
changes gradients through the shared circuit, so it can affect policy even
though policy retains all its learned coefficients.

Conditional opening-family 95% intervals for the mean differences are
[−.04079, −.02167] KL and [+.02368, +.05977] MSE. Their paired-seed sample
standard deviations are .03615 and .10364. The family intervals condition on
the three fitted models; they do not resolve uncertainty across training seeds.
The 62,984-position current-source-novel slice has mean differences −.03059 KL
and +.02016 MSE. All four arms use current input, so this slice uses the existing
current-source novelty mask, not the earlier current/neutral intersection.
It precedes retinal rendering and does not certify absence of rendering collisions.

| Complete warm B1 prediction, MFLOPs | Seed 4 | Seed 5 | Seed 6 |
|---|---:|---:|---:|
| Dense | 177.81 | 181.09 | 186.42 |
| Value group | 181.00 | 194.95 | 187.98 |

Despite saving 4,248 decoder FLOPs, the selective value models require an
average **6.20M more complete warm B1 FLOPs**, and 6.00M more at B32. Their
trained recurrent activity changes how much work the unpruned runtime can skip.
These counts include rendering and both heads; they exclude optimization,
memory work, nonlinear functions and search. No CNN or playing-strength
advantage is established by this short screen.

All six retained-response reconstructions reproduce every policy logit exactly;
value errors are at most 1.2e-7. They respect both binary masks and the qualified
FP64 head accumulation. The first-wave diagnostics have 227–348 varying motor
coordinates and standardized participation ranks 3.67–5.92. Raw variance remains
concentrated. In the new dense seeds, the leading identities are **DNa14 body
13410, DNpe018 body 165031 and DNpe015 body 35519**; two lie outside the
five-cell candidate group. No candidate membership is changed after observing
these endpoints. These are model-response findings, not established biological
value roles or a reason to fit permanent motor groups from one seed.

The numerical reconstruction, disabled-state endpoint audit and paired-summary
helpers pass ten focused tests, including checks that bind decoder interventions
to their exact checkpoint and probe selection. Full-run audits and all twelve
actual diagnostic checks pass. Their source, metrics, responses, counts and reports have
verified copies; checkpoint payloads remain on worker owners and their replicas.

## Complete screen: a promising side restriction

The [closed twelve-endpoint report](results/readout-roles-v1.json) uses the
declared analysis rules, all 70,425 validation positions and the same 254 opening
families. Every arm receives 32,000 labeled exposures per seed. Lower policy KL
and value MSE are better; higher teacher top-move agreement is better.

| Head, mean across seeds 4/5/6 | Policy KL | Value MSE | Teacher top-1 agreement |
|---|---:|---:|---:|
| Dense | 1.69163 | .64922 | 19.179% |
| Five-cell value | 1.65696 | .68652 | 19.733% |
| Soma-side policy | 1.67762 | .58547 | 18.979% |
| Shuffled-side policy | 1.69732 | .57676 | 18.609% |

| Candidate minus reference, mean | Policy KL | Value MSE | Top-1, percentage points |
|---|---:|---:|---:|
| Five-cell value − dense | −.03468 | +.03730 | +.554 |
| Soma-side − dense | **−.01401** | **−.06375** | −.200 |
| Soma-side − shuffled-side | **−.01970** | +.00872 | +.370 |
| Shuffled-side − dense | +.00568 | −.07246 | −.569 |

Soma-side improves both losses against dense in **each paired seed**. The
policy differences are −.00963/−.02395/−.00846; value differences are
−.05728/−.06017/−.07378. It also improves policy KL against the frozen shuffled
partition in each seed, by −.01124/−.01691/−.03095. This is a promising
within-fly attachment result. Teacher top-move agreement nevertheless falls
against dense in all three seeds. Shuffled-side shares the value improvement,
so an anatomical explanation for the value gain is not established.

For soma-side minus dense, conditional opening-family 95% intervals are
[−.01593, −.00958] KL and [−.07566, −.03190] MSE; paired-seed sample SDs are
.00863 and .00881. Against shuffled-side they are [−.02374, −.01125] KL and
[−.04960, +.03491] MSE, with seed SDs .01015 and .00945. Family intervals
condition on the three fitted weights, do not measure training-seed uncertainty
and are not corrected for multiple contrasts. The current-source-novel slice
has soma-side minus dense differences −.01705 KL / −.06089 MSE, and soma-side
minus shuffled-side differences −.01347 KL / +.01140 MSE.

| Complete warm B1 prediction, MFLOPs | Seed 4 | Seed 5 | Seed 6 |
|---|---:|---:|---:|
| Soma-side | 187.56 | 176.51 | 181.20 |
| Shuffled-side | 191.26 | 176.23 | 183.23 |

Soma-side and dense have essentially the same mean counted prediction work:
soma-side is only .018M FLOPs lower at B1 (about .01%), with substantial paired
seed variation. The mean B32 difference is −.151M. These are complete unpruned
warm reset-state counts, including rendering and both heads. They are not
latency measurements or a comparison with the earlier CNN budget.

All twelve endpoint audits verify unchanged disabled coefficients and zero
disabled optimizer moments. Reconstructing the retained decoder responses gives
exact policy logits and value error at most 1.2e-7. Soma-side has 461/317/371
varying motor coordinates, versus dense's 274/277/253, but its standardized
participation ranks are only 3.68/3.59/2.58. Raw variance remains concentrated;
more varying cells alone does not establish richer or useful representations.

Keep dense as the reference and soma-side as a provisional candidate. No mask
membership, loss, optimizer or horizon was changed after inspecting these
endpoints. No physiological Go role, playing-strength gain or CNN advantage
is established.

## Fresh-seed confirmation

[Wave 1](../configs/readout-confirmation-wave1-v1.json) registers dense and
soma-side seeds 7/8/9; [wave 2](../configs/readout-confirmation-wave2-v1.json)
registers their shuffled-side controls. The
[analysis contract](../configs/readout-confirmation-analysis-v1.json) freezes
all three contrasts before new scientific training. The confirmation asks
whether both losses improve versus dense in every fresh seed and whether
policy KL improves versus the same frozen shuffled partition in every seed.
Report teacher agreement, arithmetic and any contrary result. Confirmation
will be reported separately from discovery; reusing the validation families
makes this a training-seed confirmation, not an untouched-data test.

Keep B's current-board input, K8, B32, the qualified `2ba439e309df1e902b58`
source, rate .03, warmup 100, epsilon 1e-6, bias multiplier .01, original enabled
initialization and 1,000-update endpoints. All nine actual numerical and exact
recovery gates must pass before training. Worker allocations retain disjoint
24-core lanes and checkpoint replicas. The five-cell value restriction is not
carried forward after its mixed loss tradeoff. No new shuffled partition is
selected, so this does not test robustness across arbitrary partitions.

Persistent-state engineering remains a separate phase. A successful attachment
confirmation would select a provisional decoder for that study; it would not
silently combine new memory rules with this readout comparison.

The [fresh-seed gate report](results/readout-confirmation-gates-v1.json) retains
**seven passing numerical cases and two failures**. All seven qualified cases
pass exact recovery and preserve their paired initial arrays and samplers.
Soma-side seed 7 differs in one policy weight after the second update by
7.70e-6; seed 8 differs in one logit before the third update by 6.10e-6. These
exceed the existing elementwise tolerances. At this original gate, no scientific
confirmation learner had started, and no failed seed was replaced.

A separate independent FP64-head oracle keeps FP32 recurrence and Adam and
uses the same examples, settings and tolerances. It clears soma-side seeds 7/9,
but seed 8 still differs by 4.88e-6 at the same logit. This is a diagnostic,
not an accepted substitute for the registered gate. The learner source remains
unchanged while identical-parameter cross-evaluation investigates the remaining
error. Numerical artifacts and failures have verified owner/root archives.

The [completed cross-evaluation](results/readout-confirmation-drift-v1.json)
finds maximum logit differences below 4.8e-7 at identical weights with the
original reference, and below 2.4e-7 with the FP64-head oracle. At the failing
coordinate, swapping only the policy-weight array reduces the original
6.10e-6 difference to 1.61e-7 (FP64 oracle: 4.88e-6 → 1.46e-7). This locates
most of the discrepancy in policy-weight drift across optimizer updates;
it does not yet identify which gradient or rounding boundary should change.
The original failed cohort remains recorded; the subsequent explicit protocol
revision is described below.

The contribution audit also motivates a later conditioning experiment. For an
invertible diagonal scale S, `policy = W S (r - mean) + bias` has the same affine
function class as the dense decoder. It could make weak motor variations easier
to optimize, but supplies no new information. Statistics must come from training
positions, with a declared variance floor and fixed fitting cost. Treat this as
an optimization/parameterization factor; do not silently combine it with the
value-group or side-mask trials. The earlier mean-conditioning qualification
failure remains part of the numerical history.

## Numerical protocol revision and larger decoder study

The [v2 numerical plan](../configs/readout-confirmation-numerics-v2.json) was
registered before its checks and before confirmation learning. It tests three
peak-rate transitions from identical native checkpoints/moments, plus three
independent updates under the actual 100-step warmup. All existing numeric
tolerances, examples, model/optimizer settings and the scientific runtime remain
unchanged. This changes the engineering test inputs; it does not turn the two
original free constant-peak failures into passes.

The [completed report](results/readout-confirmation-numerics-v2.json) contains
18/18 numerical and 9/9 exact-recovery passes. Each numerical case checks full
states, outputs, losses, all gradients, parameters and both moments. Independent
canonical head arithmetic reconstructs every captured policy gradient and Adam
update exactly for seeds 7/8. Substituting the reference motor pool into the
native head reproduces first-update differences around 4.6e-6 from initial
motor discrepancies around 1e-7. Explicit FP32 barriers do not resolve this.
The evidence supports rounding amplified by cancellation rather than a
demonstrated native head/Adam equation bug. The combined-report controller now
passes eight focused tests and admits all nine actual jobs. The
[wave 1 launch](../configs/readout-confirmation-wave1-v2.json),
[wave 2 launch](../configs/readout-confirmation-wave2-v2.json) and
[analysis contract](../configs/readout-confirmation-analysis-v2.json) bind this
evidence while preserving the original scientific settings and run IDs. Six
dense/soma learners have started, with full-validation, signal and arithmetic
followups. The shuffled controls are queued behind those completed analyses.

The first followup deployment stopped before starting any evaluation worker
because host-specific Python bytecode conflicted during immutable replication.
Recovery v2 excludes caches from new bundles. The second-wave queue initially
failed during imports because its `queue.py` entry shadowed Python's standard
library; recovery v3 changes only that entry filename to `worker.py`. Failed
attempts remain intact. No scientific horizon, learner or mask changed.

In response to the remaining large CNN gap, priority moved to the independent
[36-case motor learnability screen](motor-learnability-study.md). It probes raw
and standardized linear mixing, competitive gating and a nonlinear decoder at
three rates on three frozen cores. This larger screen is complete; no recurrent
equations were changed. It demonstrates that sharp outputs alone do not solve
the held-out learning problem and motivates separating decoder convergence
from missing or poorly trained circuit features.
