# Scientific programme

The target is a useful 9×9 Go engine whose fixed fly circuit competes with a
conventional neural engine at matched inference arithmetic and training-data
exposure. This is a hypothesis, not an established advantage. Engineering and
scientific progress are evaluated separately. The original MaleCNS node and
edge identities remain fixed in every fly variant.

## Comparison contract

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
- First compare prior-only predictions and paired fresh games. Then use the
  same native PUCT/Gumbel settings and count every neural evaluation per move.
  Search is not a free improvement under a prediction-compute budget.
- Report learning curves versus exposures and time; validation KL, value MSE,
  calibration, phase/value-band/D4-novel slices; and actual playing strength
  with uncertainty. Confirm promising results with at least three seeds.

## First priority: optimization

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
