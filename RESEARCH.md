# Scientific programme

The target is a useful 9×9 Go engine whose fixed fly circuit competes with a
conventional neural engine at matched inference arithmetic and training-data
exposure. This is a hypothesis, not an established advantage. Engineering and
scientific progress are evaluated separately. The original MaleCNS node and
edge identities remain fixed in every fly variant.

## Comparison contract

The primary comparison fixes prediction arithmetic and the labeled-position
training horizon. For input-dependent work, report
`C(f; Q) = mean_{x in Q} FLOPs(f, x)` on a declared input sample Q, together
with its distribution and batch/cache convention. Freeze the counting rule,
reference sample and candidate architecture before candidate training; report
the achieved mismatch across seeds. Parameter count is descriptive, not the
matching target. Wall time is a separate engineering comparison.

- A training exposure is one labeled board position processed by the learner,
  including repeated samples. Also report unique games, unique positions,
  augmentation, optimizer updates, global batch and total training compute.
  Extra forward evaluations in zeroth-order methods count as exposures and work.
- Match features, legal masks, teacher targets, opening-family splits, sampling
  and augmentation. Leave final test labels and final match openings closed
  until configuration selection. Give both model families equal tuning budgets.
- For the initial control use a small residual CNN with policy and value heads.
  Choose its width/depth from an arithmetic ledger, before training. The control
  is a benchmark; it does not replace or augment the fly's recurrent circuit.
- Count multiply and add as two operations. A fly prediction includes all K
  recurrent passes, approximately `2 K E` sparse aggregation FLOPs, plus neuron
  dynamics and both adapters. Count actual remaining work for padded kernels,
  nonlinear operations and weight transforms separately. Report effective
  nonzero arithmetic, executed/padded arithmetic, precision, batch and latency.
  Compiler FLOP estimates and device occupancy alone are insufficient.
- A change in firing rules can change effective arithmetic without changing
  the topology. The current exact zero-skipping ledger supports hard rates
  only and rejects smooth rates. Qualify an extended ledger before claiming
  an equi-FLOP advantage for a new dynamics model; equal exposures alone do
  not establish that advantage.
- First compare prior-only predictions and paired fresh games. Then use the
  same native PUCT/Gumbel settings and count every neural evaluation per move.
  Search is not a free improvement under a prediction-compute budget.
- Report learning curves versus exposures and time; validation KL, value MSE,
  calibration, phase/value-band/D4-novel slices; and actual playing strength
  with uncertainty. Confirm promising results with at least three seeds.

## Active next phase: biological interfaces

The first optimizer/spatial/control batch is complete. The user's next
proposal prioritizes coherent retinal input, bilateral historical context,
and motor/descending readouts inferred from robust response correlations.
Follow the [B0–B5 contract](docs/biological-interfaces.md) within the deliberate
[neuron-group study](docs/group-study.md). The initial atlas/correlation pilot
is complete; it establishes neither biological tuning nor improved learnability.
Earlier studies and the FLOP/exposure comparison contract remain intact.
The user's subsequent proposal puts [quality-aware motor embedding training](docs/embedding-study.md)
before functional partitioning. Coherent spherical views, genuine shared-prefix
branches and a frozen-circuit control form a bounded pipeline pilot. Its
engineering gate and selected branch probes do not constitute Go validation.

## Optimization study contracts

The existing 2×2 depth/readout study uses Adam at 0.003, batch 32 and 10,000
updates. Preserve it. It does not establish that the optimizer is well tuned.

1. On balanced V0, screen learning rates `0.0003, 0.001, 0.003, 0.01, 0.03,
   0.1` from identical initial states and paired minibatches. Start with K=4,
   656 groups and a bounded equal-exposure budget. Include the original rate.
   A TPU batch change requires a separate experiment, not a silent continuation.
2. Record global clipping frequency, gradient and parameter-update norms by
   group, edge-magnitude changes, active-neuron fraction, recurrent-state scale
   by pass, value saturation and non-finite failures. Inspect the effect of
   softplus parameterization on initially tiny strengths. A small raw edge
   gradient alone does not imply a small Adam update.
3. Confirm the best stable rates and the reference with additional seeds and
   longer training. Then compare a schedule, warmup and separate core/adapter
   rates with equal tuning budgets. Treat clipping, epsilon, normalization and
   reparameterization as separate factors.
4. Only after a calibrated gradient-based reference, test a small structured
   zeroth-order hybrid. Perturb low-dimensional type/gain coordinates rather
   than immediately estimating a 15M-dimensional gradient. Use antithetic
   directions and identical minibatches for both signs; count all queries.
   Keep ordinary Adam, perturbation-only and hybrid controls.

The first rate screen exposes loss of activity at high global rates: at 0.1,
type-bias gradients become zero and final-state activity falls to roughly 2.5%.
The next optimizer factor is a smaller **type-bias rate**, keeping the global
gradient clipping and moment equations unchanged. Multipliers apply only to
the final parameter update; zero freezes a parameter group while estimating
its moments. First qualify identity scaling and Rust/JAX parity, then compare
smaller bias multipliers at an otherwise fixed rate and exposure horizon.

The completed 1,000-update seed-1 screen ranks global rate 0.01 ahead of the
other global rates on the fixed 2,048-position validation slice. Global 0.03
with bias multiplier 0.01 improves value MSE further at similar policy KL.
These are hypotheses for confirmation: full validation of all six rates and
both bias variants precedes additional seeds and longer training. High-rate
failures remain in the comparison; they are not evidence against the topology.

For a structured perturbation `theta = theta0 + A z`, the proposed estimator is

```
g_z = (1 / (2 m sigma)) sum_i [L(z + sigma u_i) - L(z - sigma u_i)] u_i.
```

Specify the distribution and scaling of `u`, the immutable map `A`, and which
parameter groups are affected. The map cannot create a missing edge. This is
an optimization experiment, not a claim about biological learning.

## Spatial vision: a concrete biological hypothesis

The fly optic lobe has repeated cell types arranged in retinotopic hexagonal
columns. The flyvis work models spatially repeated connectivity as filters and
optimizes a recurrent mechanistic model for optic flow. Its evidence concerns
visual computation and neural-response prediction, not Go superiority.
[Lappalainen et al., Nature 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/).

Our retained MaleCNS annotation table contains `assignedOlHex1/2`, cell type,
side and soma metadata. The first audit found 23,720 neurons with both column
coordinates and **zero sensory neurons with direct column annotations**.
Photoreceptor coordinates must therefore be inferred and qualified, or input
must be attached to explicitly declared annotated downstream cells.

Planned sequence:

1. Infer each photoreceptor's column from its existing weighted connections to
   annotated lamina/medulla cells. Record target-type agreement, spatial spread,
   side consistency, coverage and ambiguous/missing mappings. Never present an
   inferred coordinate as a measured annotation. Use synapse counts for the
   geometric audit independently of learned strengths.
2. Compare spatial attachment against a shuffled attachment using **the same
   photoreceptor subset, feature coverage, gains and output pools**. Keep the
   original all-sensory random baseline as an additional control. This separates
   spatial organization from changing the selected cell population.
3. Map Go's square board into the hexagonal coordinates explicitly; document
   resampling, aspect ratio, boundaries, left/right orientation and channel
   assignment. Audit how much square-grid adjacency and D4 symmetry survives.
4. Separately test spatial output pools, then parameter tying on existing visual
   edges by pre-type, post-type and relative column offset. One possible rule is

```
w_e = sign(pre_e) * softplus(theta0_e + k[type_pre, type_post, dq, dr]).
```

This shares a learned correction across repeated connections while retaining
every original edge and its individual prior. A fully shared-magnitude variant
would be a separate comparison. Missing boundary edges remain missing; this is
a masked, convolution-like recurrent circuit, not an exact translation-invariant
CNN. A scalar activation change by itself does not create convolutional weight
sharing or spatial organization.

The first executable attachment study is a **sensory overlay**: preserve all
baseline ports except visual sensory cells whose same-side downstream column
has a unique majority, at least 50% of eligible synapse mass and at least four
synapses. Ties and missing coordinates are excluded. Rescale the two column
indices independently per side to board coordinates 0..8 and round to the
nearest point. Keep each cell's original feature channel. The matched control
shuffles assigned board points within side × photoreceptor type × channel,
preserving feature multiplicities and every output pool. This tests a simple
affine adapter; it does not establish calibrated retinal orientation, physical
isotropy or exact D4 equivariance. Record coverage and excluded cells before
training; compare against the unchanged all-sensory random baseline as well.

The structural audit finds possible sensory paths to 148,736 of 149,210
readout cells by pass 4. Extra passes therefore add processing time more than
newly reachable cells. This ignores nonlinear gating and signal attenuation.
The 23,720 directly annotated column cells belong to 15 repeated visual types.
These observations motivate a separate **visual readout** screen after the
input-only registration: retain its spatial sensory map, then compare the
original all-cell pools with spatial and matched-shuffled pools of directly
annotated visual cells. All circuit nodes and edges still execute.

Map those measured column indices through the same per-eye bounds as the
sensory overlay; exclude coordinates outside those bounds rather than clip
them. Annotated interneurons use `somaSide` when L/R (falling back to
`rootSide`); the sensory cells use `rootSide`. The first artifact attempt
incorrectly reused the sensory side field and correctly stopped with no
eligible readout cells; it remains recorded. Assign each of the 15 types to one of eight seeded channels, consistently
across columns. A readout group is `8 * board_point + channel`; normalize by
the square root of its actual cell count. Other neurons have no direct adapter
readout, while remaining in the recurrent circuit. The shuffle permutes group
assignments within side and type, preserving selected cells, all group counts,
channel identities, input ports and trainable-parameter shapes. Report empty
groups and excluded cells. This is a spatial aggregation hypothesis, not a
calibrated model of the fly's motor output or an exact convolution.

The first comparison starts from initialization at K=4, G=656, B=32, rate 0.01
and 4,000 updates, seed 1. Include spatial-input/original-readout as its direct
control at this same horizon; the existing random-input/original-readout
confirmation is an additional reference. Retain all three new outcomes before
selecting further seeds or any TPU continuation.

## Neuromodulation and plasticity

Drosophila experiments and circuit models support dopamine-dependent learning
in specific mushroom-body pathways and interacting memory timescales. These
results motivate a local plasticity experiment; a universal dopamine scalar
does not specify an effective whole-graph Go learner.
[Huang et al., Nature 2024](https://www.nature.com/articles/s41586-024-07819-w),
[Bennett et al., Nature Communications 2021](https://pubmed.ncbi.nlm.nih.gov/33963189/).

A proposed three-factor family is

```
eligibility_e[t+1] = lambda_e * eligibility_e[t] + psi(r_pre[t], r_post[t])
plastic_e[t+1] = (1-rho_e) * plastic_e[t] + eta_e * m_region[t] * eligibility_e[t]
w_e[t] = sign(pre_e) * softplus(theta_e + plastic_e[t]).
```

This is our mathematical experiment, not an asserted reconstruction of every
fly synapse. First specify `psi`, modulator source, selected existing synapses,
centering/bounds, time constants, gradients and state lifetime. Compare zero
plasticity, ungated activity-dependent plasticity, random modulation and
task-linked modulation with matched parameter and computation budgets.

Begin with bounded state on an annotated subset or low-dimensional groups.
Dense per-edge, per-example eligibility is expensive: 15,270,273 × B × 4 bytes
for each FP32 trace, before optimizer state or recurrence. Report that cost.
Biological timescale labels do not excuse unspecified memory or state leakage.

Teacher labels can drive updates **after** a training prediction. They cannot
be supplied as a neuromodulator during validation, search or inference. Any
within-game adaptation must use causally available information, explicit game
boundaries and isolated state for separate search branches. Reset-within-board
and persistent-across-game variants are different algorithms.

## Evidence discipline and order

The order is TPU measurement → optimizer calibration → spatial attachment →
one plasticity/zeroth-order comparison → matched neural control and larger
strength panels. Independent engineering gates and current trials continue.
Change this order only for a recorded bottleneck or experimental result.

While optimizer and spatial studies execute, implement the independent CNN
control so the grand-goal comparison can begin without waiting for plasticity.
`configs/matched-control-v1.json` registers an equal three-rate large-batch
screen. A 64-channel, ten-block residual CNN with a 3x3 policy-hidden layer
matches the K=4 fly's nominal arithmetic within 1%; the operation ledger is
`python/flygo/cost.py`. It uses the same learner, data and loss code. Report
boundary/channel padding and actual latency separately: matching nominal
FLOPs does not match memory traffic or accelerator utilization.

Do not interpret an insufficiently trained baseline as a failed topology, or a
successful biological variant as proof of topology's causal superiority. A
recent flyvis preprint illustrates how initialization and control design can
change apparent topology advantages; it is a caution about experimental design,
not a verdict on this Go task.
[Dhiman, arXiv 2026, preprint](https://arxiv.org/abs/2604.04033).

Every executed study gets a frozen configuration, source/runtime IDs, a bounded
budget, retained failed attempts and an entry in `PROGRESS.md`. Register
equations and acceptance checks before implementing a new neuron rule.

Numerical qualification retains the established full-state, full-gradient and
parameter/moment tolerances. Also report relative update differences and check
Adam's arithmetic independently from identical moment arrays. On real Go data,
one nearly cancelled type-bias gradient caused a 1.29e-5 update difference and
failed an additional relative-delta check despite passing the original gates.
That sensitivity is retained as a diagnostic, not hidden by parameter-scale
relative errors. CPU/TPU trajectories need not remain bitwise identical;
same-backend fresh-process recovery must reproduce the next update exactly.
CNN qualification compares three transitions starting from identical CPU
checkpoint states: this keeps derivative checks at the same parameters. Initial
free-running comparisons exposed near-cancelled Adam updates and subsequent
ReLU-boundary crossings; retain both failures. Checkpoint alignment changes the
test input, not model equations, optimizer settings or pointwise tolerances.

CPU profiling on three spare cores identifies indirect transpose reads as a
concrete cost: at B=1, roughly 0.197 s versus 0.031 s for the forward multiply;
at B=32, 0.433 s versus 0.166 s. The next storage-only optimization keeps
destinations contiguous in transpose order and packs its weights once per
backward call, amortized over K passes. Preserve canonical parameter IDs and
each row's summation order. Require byte-identical full outputs and all gradients
on real initial and trained V0 fixtures before adoption, plus regression checks.
Current study binaries remain frozen. Extra layout memory and preparation time
must appear in the performance report.

After qualifying the packed transpose, test exact zero-activity skipping.
Build one Boolean flag per node when fewer than 80% of batch rows contain a
nonzero value. A multiplication by an all-zero row contributes zero; skip its
batch arithmetic while retaining the edge and its position in summation order.
Apply the same rule to zero input/cotangent rows in edge gradients. Preserve
the dense path when activity is high. This is an execution optimization, not
topology pruning or a new neuron rule. Require the same byte-level checks and
measure dense-path overhead. Report data-dependent executed work separately
from the unchanged nominal graph-FLOP comparison.

Both CPU optimizations pass the original full-state/all-gradient byte checks
on real initial and trained V0 batches. Three-core trained B=32 inference falls
from 0.69 to 0.44 seconds and loss-plus-gradient time from 4.32 to 1.70 seconds.
These are bounded microbenchmarks with concurrent fleet workloads, not a
24-core throughput or universal speedup claim. Keep the reference artifacts
and separately qualify actual research-lane throughput.

## Next optimizer factor: deterministic schedules

The longer constant-rate trials still trade policy improvement against activity
loss and unstable value error. Before a zeroth-order hybrid, compare two
separate schedule factors from initialization: 500-update linear warmup to the
declared peak, or cosine decay from that peak to 10% at update 4,000. Keep the
128,000-position horizon, batch 32, clipping, diagnostics, ports and sampling
fixed. Apply each factor to peak 0.01 and to peak 0.03 with bias multiplier
0.01; adopt the corresponding constant-rate confirmations as controls. The
first screen uses seed 1 and retains all four outcomes before further seeds.

For one-based update `t`, warmup uses `peak * t / warmup_steps`. After warmup,
the constant schedule stays at peak. A cosine schedule starts at
`s = max(1, warmup_steps)` and uses
`peak * [r + (1-r) * (1+cos(pi*q))/2]`, where
`q = clip((t-s)/(decay_until-s), 0, 1)` and `r = 0.1`. The absolute decay endpoint
is an immutable optimizer setting, independent of how many updates a process
runs before checkpointing. Resume must retain the schedule contract and use
the restored optimizer step. Qualify boundary behavior, restart continuity and
Rust/JAX update parity before launching these cases.

This is a recorded change in ordering: ordinary optimizer calibration remains
the prerequisite for a meaningful zeroth-order comparison. The existing
constant-rate and spatial-input studies continue unchanged.

## Constant-state execution and counted work

The first recurrent state is always 0.01 for every neuron and board. Cache
`W * relu(h0)` alongside the existing transformed weights, computing it once
at B=1 in canonical edge order. Broadcast that message across a batch before
the unchanged bias/drive update. Rebuild after every parameter update or
restore. This removes a redundant first sparse multiply from warm inference;
training still computes the message once per update. Backward arithmetic and
all neuron/edge identities remain unchanged. Require byte-identical real V0
initial/trained B=1/32 outputs, states, losses and all gradients, next-update
recovery and the existing regression suite before adoption. Record the extra
N FP32 cache, cold preparation and warm latency separately. Frozen studies
keep their original binaries.

This also motivates an explicit execution ledger. On a declared sample of V0
validation positions, count active source rows at each pass using the actual
CPU batch-wide 80% threshold, then count the corresponding traversed-edge
multiply-adds. Report B=1 and B=32 distributions, with constant-state caching
separate. Add adapter, dynamics and head arithmetic under the existing nominal
convention; do not call this a hardware instruction counter. Masks, memory
traffic, nonlinear operations and latency remain separate. The current CNN
comparison matches nominal architecture FLOPs; it does not establish equal
executed CPU work. Any smaller executed-cost control needs its own registered
training comparison, without retrospectively replacing the original study.

The completed 256-position seed-1 audit gives 8,339,945 mean warm B=1
operations for the matched fly checkpoint (range 7,594,014–9,084,738), under
that accounting convention. Register an additional 17-channel, nine-block
CNN: 8,361,890 nominal operations, +0.263% relative to that mean. Width/depth
were chosen from cost before any new training outcome. The same three rates
(0.001/0.003/0.01) receive 256 updates at B=2,048, seed 1; minimum final
validation KL+MSE selects the rate, then confirm from initialization at 512
updates for seeds 1/2/3. Adopt the already completed fly comparison cases at
the same horizons. Audit the other fly seeds' work separately and report their
variation; no per-position equality is claimed. CNN boundary/channel padding,
memory traffic and latency remain explicit limitations of this arithmetic
match. Preserve the original nominal 126.7M study unchanged. Qualify the small
CNN shape before its first TPU training run, and queue it after the active
spatial study so a single cohort continues to own all devices.

## Smooth firing-rate screen

The trained hard-rectifier baseline spends little arithmetic on later passes
because many rates are exactly zero. Together with the high-rate gradient
collapse, this motivates one explicit neuron-rule factor while existing
optimizer and adapter studies continue unchanged. Flyvis's dynamics interface
also exposes softplus as an activation option for its graded-release model.
[Flyvis dynamics source](https://github.com/TuragaLab/flyvis/blob/main/flyvis/network/dynamics.py).

Define a fixed softness `s >= 0`, with `r_0(v) = max(v,0)`. For `s > 0`,

```
r_s(v) = max(v,0) + s * log1p(exp(-abs(v)/s))
dr_s/dv = sigmoid(v/s).
```

Use this same rate in recurrent messages and output pooling. The leaky state
update, h0=0.01, signed softplus edge strengths, initialization, ports and all
trainable arrays retain their existing definitions. This smooth function is
also the expected rectified response under additive zero-mean logistic noise
of scale s; our implementation evaluates the expectation deterministically.
Treat it as a hypothesis about threshold variability and gradient flow.
FP32 underflow can still produce zero at sufficiently negative voltages.

Register softness 0.01/0.05 crossed with global rate 0.01 or rate 0.03 plus
bias multiplier 0.01. Each starts from initialization at K4/G656, seed 1,
batch 32 and 1,000 updates (32,000 exposures), on V0 with original random
ports. Adopt the corresponding completed hard-rate screen checkpoints as
controls. Keep all four final natural/novel validations and diagnostics before
selecting longer or additional-seed work. These are exploratory one-seed
screens, separate from the matched million-exposure comparisons.

Keep the scalar rate rule in one Rust module with an independent JAX function.
The baseline configuration must retain old-checkpoint readability and exact
output/update bytes. Give smooth models a distinct checkpoint model version
and serialize softness. Require finite differences through negative, zero
and positive states, all-group Rust/JAX derivative/update checks, full-graph
V0 parity for both softness values and fresh-process continuation before
training. Report nonlinear functions and loss of zero-skipping opportunities
in performance figures; a better loss does not imply better compute efficiency.

The first smooth implementation passes full-graph CPU parity but exposes serial
rate evaluation and readout work. Parallelize independent rate elements and
readout groups, retaining ascending canonical neuron order inside every pool
and the original per-neuron gradient sum order. Cache only readout membership
indices, not new learned state. Require byte-identical baseline and both smooth
models' full outputs/gradients against the preceding implementation, plus
fresh continuation, before launching the smooth screen. Measure the extra
readout-index memory and timings under the same CPU allocation.

## Schedule confirmation after the seed-1 screen

All four schedule cases completed full validation before this revision. At
rate 0.03 with bias multiplier 0.01, cosine improves policy KL more, while
warmup improves value MSE more. Both advance to seeds 2/3 at the unchanged
128,000-exposure horizon; seed 1 and all three constant-rate controls are
adopted. `configs/optimizer-schedules-confirm-v1.json` records this decision
before any new training. Analyze separate paired three-seed contrasts, retain
all outcomes, and do not reinterpret the result-driven choice as equal total
family tuning compute. The same seed-1 source environment executes the new
trials, after each host's smooth-rate validation releases its lane.

## Readout conditioning after the offset audit

The retained initial B32 fixture has 99.923% of pooled squared magnitude in
the component common to all 656 groups. Pooled RMS is .3082, centered RMS
.00855 and across-input variation RMS .00308. Six-update smooth qualification
fixtures retain 99.46% / 99.79% common energy and saturate every value output.
The trained hard-rate bias-scaled fixture instead has 5.20% common energy.
This read-only audit reconstructs the pools and checks policy/value against
native outputs; it adds no optimization exposures. Evidence:
`next-gates/readout-offset-audit-v1.json`. These fixtures suggest an early
conditioning problem, not a demonstrated general training failure.

Before training a new variant, define an external readout transform

\[
 \bar z=G^{-1}\sum_g z_g,\qquad
 z'_g=z_g-(1-\lambda)\bar z,\qquad \lambda=G^{-1/2}.
\]

This symmetric linear map leaves group differences unchanged and scales the
common component by lambda. It is invertible for positive lambda and introduces
no learned parameters, new recurrent edges, extra recurrent passes or direct
board-to-head path. Its transpose is itself. The baseline lambda=1 bypasses
all new arithmetic. The mean is per prediction, independent of other batch
members; it is not a running statistic or batch normalization. It costs 3G+4
additional counted arithmetic operations per prediction, excluding constant
configuration preparation.

`configs/readout-conditioning-v1.json` registers four seed-1, 32k-exposure
cases: hard-rate mean damping at .01 and bias-scaled .03; an unchanged hard
readout with only policy/value weight learning rates multiplied by G^-1/2 at
bias-scaled .03; and mean damping with softness .01 at bias-scaled .03. Adopt
the matching hard/smooth screen controls. This distinguishes a readout
reparameterization from merely slowing the output weights, and explores one
explicit smooth-rate interaction. Keep initial parameter draws, data, all
ports, topology and parameter shapes unchanged. All final full validations
precede selection. Require independent finite differences, Rust/JAX states,
all gradients, three updates, legacy byte identity and fresh actual-trainer
continuation before launch. Actual TPU qualification remains a separate gate.

The first full-graph free-running conditioning qualification passed states,
losses and gradients, but at the third update 38/53,792 policy weights exceeded
the original tolerance (maximum error 7.42e-5). The nearly common pooled values
made direct FP32 mean subtraction sensitive to reduction error near zero
features; Adam amplified that error. The failure remains in
`readout-mean-hard-cpu-parity-v1`. Evaluate the same linear map by first
subtracting one pool, summing those deviations, then restoring the scaled
common term. Both implementations use the same stable algebra, and the declared
symmetric transpose in backward. The revised cost is 3G+4, with unchanged
parameters, mathematical hypothesis and tolerances. Repeat qualification before
any scientific trial; do not weaken the acceptance threshold.

The stable formula's B1 attempt also fails the unchanged parameter tolerance:
one leak variable differs by 8.15e-6 at the first update, despite passing full
states, losses and gradients. Stable mean evaluation alone therefore does not
resolve every Adam sensitivity. Retain both failures. Before deciding whether
this variant can enter the registered B32 screen, qualify all three updates at
the actual training batch of 32 with the same tolerances and a separately
reserved full-reference memory budget. This is an additional qualification
scope, not evidence that B1 cross-backend training trajectories agree. Do not
launch any dependent trial if that B32 gate fails; actual TPU remains separate.

The actual-B32 hard-rate mean-conditioning gate fails at its first update:
5/53,792 policy weights exceed the unchanged tolerance, with maximum error
2.13e-4. No dependent scientific trial is launched. The source, tests, all
failures and configuration remain retained for future investigation; this
variant has not met the declared cross-backend update gate. Mean conditioning
is deferred rather than promoted from its small-fixture results.

`configs/head-rate-v1.json` advances the simpler optimizer hypothesis from the
same offset audit: multiply value weights alone, or both policy/value weights,
by G^-1/2 in the existing post-moment learning-rate mechanism. Cross these two
choices with hard rates and softness .01, at global .03 / bias multiplier .01,
32,000 exposures, seed 1. The original biases of both output heads keep their
rates. Adopt both matching existing controls. Reuse source
`6d9b14c70b154a5cb821`, which predates mean conditioning, and qualify the
specific output-group scaling plus fresh V0 continuation before launch.

## Exact CPU parameter-validation work

`Model::validate` checked core parameter values, then checked those same arrays
again with the adapters/heads. Warm inference and each optimizer validation
therefore scanned the full edge vector twice. Its nine shape checks already
cover all core dimensions. Remove the redundant core scan and parallelize the
single finite-value check through the model's assigned executor. Preserve all
rejections, atomic failed restore, parameter values and numerical reductions.
Before accepting a performance claim, require original/trained full B1/B32
outputs, losses and all gradients to match prior fixture bytes, plus fresh
checkpoint continuation. Keep this implementation change separate from the
deferred mean-conditioning hypothesis and from frozen scientific trial sources.

## Smooth-rate confirmation after the full screen

All four final full validations are complete before registering
`configs/smooth-rate-confirm-v1.json`. At global .03 / bias multiplier .01,
softness .01 reaches KL 1.623348 / MSE .570713, versus hard-rate 1.649547 /
.571104. It has the lowest KL and KL+MSE among the four new candidates. Advance
this case from initialization to 128,000 exposures, B32, seeds 1/2/3, with all
matching hard-rate confirmations adopted. Source `6d9b14c70b154a5cb821`, original
ports and no mean-conditioning transform. Queues use released 32-55 lanes;
other settings and the million-exposure TPU comparisons retain their contracts.
This extra result-driven tuning is reported as project research compute.

### Operational recovery, 2026-09-14 06:35 UTC

The shuffled-input seed 2 TPU cohort stopped at step-zero checkpoint admission: w0 would have crossed the unchanged 96 GiB available-RAM floor after conservative reservations, during an overlapping short CPU qualification. All four workers exited and every learner log contains only step-zero validation, so this failed attempt consumed zero optimizer updates. The qualification has finished. Each host now passes an additional 58 GiB heap plus 4 GiB files preflight for the outer TPU reservation, trainer and checkpoint overhead. No storage floor is reduced.

`spatial-input-recovery-v2` adopts the seven successful cohorts, restarts shuffled seed 2 as `tpu-spatial-v0-shuffled-s2-attempt2` with unchanged source `0d4e6e6ea6298d22370c` and scientific contract, then runs the original unstarted shuffled seed 3. A new seed 2 evaluation follows its original full validation and common 32-game panels. The smooth confirmation queue failed before launching anything; its recovery changes only this failed lane dependency, keeping the original three training run IDs and qualified source. Existing healthy seed 1/3 validation waiters remain. The small CNN launcher is separately retried after the recovery study. Failed queue/cohort/evaluation records remain intact. Avoid additional large CPU qualification reservations while these cohorts run.

### Adam epsilon screen, registered 2026-09-14 07:22 UTC

Completed optimizer confirmation improves policy at larger rates but worsens value relative to .003. The mean-conditioning investigation also exposed the sensitivity of near-zero gradients under Adam. Before structured zeroth-order changes, isolate the denominator scale: `delta theta = -eta m_hat / (sqrt(v_hat) + epsilon)`, where epsilon is outside the square root. This is an optimizer factor, not a new biological claim.

`epsilon-v1` tests 1e-6 and 1e-4 separately under hard and smooth .01 firing, at global .03 and bias multiplier .01. All output-weight multipliers remain one, independent of the head-rate screen. K4/G656/B32, seed 1, 1,000 updates and 32,000 exposures match existing epsilon-1e-8 controls. All final full validations and paired differences are reported. No test labels, best-step selection or joint head/epsilon selection. A favorable screen requires a separately registered multi-seed confirmation.

Before launch, require full-graph independent JAX CPU states/loss/all-gradient/three-update checks at unchanged tolerances for every epsilon/dynamics pair, and actual six-update scheduled training plus fresh three-update checkpoint/sampler reproduction. Save epsilon in each training contract; legacy absence means 1e-8 and CLI resume rejects an unintended change. Scientific workers use one immutable qualified source and start on 92–115 only after the head-rate validation workers exit. TPU epsilon qualification remains pending.

### Small control confirmation and operational recoveries — 2026-09-14 08:15 UTC

The previously declared small-CNN screen selects .01 by final KL+MSE; all
three rates and the exact selection are retained in the results. The
three-seed, 512-update confirmation and common-opening panels were frozen
before confirmation outcomes in `small-control-confirm-v1` and its followup
plan. The actual shape qualification succeeds as `tpu-small-cnn-parity-v2`;
v1 failed on a frozen script path before launching any worker.

Confirmation seed 2's original attempt stops at initial checkpoint admission
while a development reservation overlaps. It consumes zero optimizer updates;
all controllers exit. `small-control-confirm-recovery-v2` adopts successful
seed 1, repeats seed 2 under an attempt-specific name and runs original seed 3.
All scientific settings and source `7a9c190de16ddd46f739` remain unchanged.
The recovery follows a passed all-host additional 4 GiB file / 58 GiB heap
preflight. The failed seed-2 followup stays recorded beside the successful
replacement, and every final checkpoint is evaluated.

Similarly, `epsilon-recovery-v2` repeats only hard epsilon 1e-6 after original
host 0 initial-checkpoint admission fails; only step-zero validation existed.
The replacement uses host 2 after its prior validation releases 92–115,
with identical source `c65c9808cdf39ee16587`, seed, inputs and optimizer.
Smooth confirmation seed 2 resumes its unchanged plan/source after a
pre-launch redundant bundle archive is avoided. These are operational
recoveries, not independent training seeds or additional successful screen
trials. All zero-update failures and repeat diagnostic/generation costs remain
part of project compute; no storage floor or numerical tolerance is relaxed.

### Next research phase: neuron groups

The user requests a deliberate study of neuron grouping after current
engineering and registered studies. Follow the [group-study milestones](docs/group-study.md):
annotation atlas, connectivity and null models, measured model dynamics,
execution-layout profiling, then a small number of physiological hypotheses.
The [initial descriptive audit](docs/structural-plasticity.md) preserves source
labels, counts and uncertainties. It does not infer functional equivalence
from soma proximity or raw density. The baseline already shares leak and bias
by cell type. Missing local physiology is an explicit unknown.

Local regrowth and shared retinal kernels remain proposed later experiments;
no edge has been added and no changed-topology training is registered. The
main comparison remains equal prediction FLOPs and labeled-position exposure
horizon. Parameter counts, total tuning/training work, padding and latency
are separate reported quantities, as confirmed by the user.

### Epsilon decision and source-driven execution probe — 2026-09-14 08:45 UTC

All four epsilon cases complete full validation. Every larger epsilon worsens
policy KL; only hard 1e-6 avoids a resolved MSE regression. Retain 1e-8 and
do not advance a larger epsilon to confirmation under this screen. The
single-seed head-rate result and remaining smooth confirmation stay separate.

A read-only source-driven sparse-kernel prototype preserves canonical
per-destination summation order and matches output bits, but is slower for
all six initial/trained hard/smooth B1/B32 fixtures. Preserve its source and
measurements; remove the unused candidate from the engine. Production dispatch
was never changed. Later group-based layouts must pass whole-model performance
and parity gates before adoption, rather than being inferred from edge density.
