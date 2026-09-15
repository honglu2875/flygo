# Clipping scope

Per-group clipping improves full-validation policy KL in all three paired seeds
and value MSE in all three. Mean KL is **1.70966 → 1.63205**, with MSE
**.64897 → .58767**, at 32,000 training-position exposures per endpoint.
It passes the registered provisional-candidate rule. Counted prediction work
also increases: **185.12M → 201.43M FLOPs** at batch one. This is an optimizer
ablation at equal exposure, with no matched-CNN claim.

The network, spherical input B, dense motor projection, K8 hard recurrence,
reset state, rate and epsilon remain fixed. All six final checkpoints,
full validations, motor probes and cost measurements are complete.
[Complete results](results/group-clipping-v1.json) and
[evidence closure](results/group-clipping-closure-v1.json).

A separately registered [128k-exposure confirmation](clipping-confirmation-study.md)
is running with fresh paired seeds 13/14/15. Its six learners retain the same
inputs, heads, propagation and optimizer settings; only clipping scope differs
within a pair. No longer-horizon result is available yet.

The [signal audit](signal-flow-study.md) found weak visual responses and
epsilon-dominated edge moments. Existing joint-loss diagnostics showed that
type-bias gradients supplied more than 99.95% of the squared gradient norm on
the measured trained checkpoints. This motivates an optimizer test; it does
not establish that clipping caused the weak representation.

For parameter group `p`, compare:

```text
global:          c   = min(1, 1 / sqrt(sum_p ||g_p||²))
parameter-group: c_p = min(1, 1 / ||g_p||)
```

Zero-norm groups use factor one. The selected factor multiplies the gradient
before the existing FP32 Adam moments are updated. Rust norms retain the same
deterministic FP64 reduction. There are nine groups: edges, leak, type bias,
input gain, readout gain, policy weight/bias, and value weight/bias.

This is different from raising the learning rate. For the first Adam update,
clipping a scalar gradient by a positive factor `c` gives
`−η g / (|g| + ε/c)`. Small global factors can therefore suppress tiny
gradients through epsilon even when Adam would otherwise cancel a constant
gradient rescaling. Later updates also depend on the accumulated moments.
The bias learning-rate multiplier `.01` still applies after Adam normalization.

Both `RustFly` and `JaxFly` accept `clip_mode='parameter-group'`; global is the
default. The training CLI exposes `--clip-mode`. Metrics retain the actual
pre-update group norms and applied factors. Checkpoint metadata and arrays
bind the mode; a different mode cannot silently resume the same trajectory.

## Qualification

Source `1f19481174f3bc8e5449` uses native SHA
`9b34b4da315fda00686e501debe507fdeb76d06fa07f153fcb2d4c7b0074536f`.

- 56 Rust tests pass; the existing optional long Go stress test remains ignored.
- 121 Python checks pass, including seven clipping tests and two new validation
  concentration tests. Rust/JAX group updates were tested on four virtual CPU
  devices. The existing B2 masked fixture uses a divisible two-device mesh.
- All 12 full-CNS numerical cases pass: two modes × three seeds × matched
  peak-rate and independent actual-warmup protocols, at the previous tolerances.
- All six fresh-process recovery cases pass, including B1/B32 Go inference,
  sampler/schedule/moments, the next update and atomic rejection of a wrong mode.
- The default matches the previous binary exactly in six native seed/rate cases.
  Three actual legacy checkpoints load and continue identically.

The preserved engineering receipt counts **3,456 native update exposures**.
Including **1,152 independent JAX reference exposures**, the qualification used
**4,608 V0 update exposures**, separately from synthetic fixtures and scientific
training. Initial launcher, device-mesh and transfer-staging failures remain
recorded; none changed the learner equations or numerical tolerances.
The owner evidence and actual recovery checkpoints have second-worker archives
verified member by member. Large archives stay off the root host.
[Complete engineering record](results/group-clipping-engineering-v2.json).

## Registered comparison

[Trials](../configs/group-clipping-trials-v1.json) use fresh paired seeds
10/11/12 and 1,000 B32 updates per endpoint: 192,000 scientific exposures total.
Initial parameters, moments and sampler are identical within each pair.
Each host ran two learners on separate 24-core lanes. The actual initial
checkpoint arrays, moments, ports and sampler states match exactly within all
three pairs. TPU stays paused.

[Analysis](../configs/group-clipping-analysis-v1.json) uses all 70,425 validation
positions, paired opening-family resampling and separate seed variation. Policy
KL is primary; value MSE, teacher agreement, entropy and top probability are
retained alongside motor responses and complete prediction arithmetic. Sharper
policies alone do not establish better learning. A provisional candidate must
improve KL in every seed and must not worsen mean value MSE. All fixed endpoints
and failures are reported; no CNN comparison is inherited from this screen.

## Learning and signal results

| Seed | Global KL | Per-group KL | Difference | Global value MSE | Per-group value MSE |
|---|---:|---:|---:|---:|---:|
| 10 | 1.70178 | 1.62569 | −.07609 | .62240 | .59155 |
| 11 | 1.70578 | 1.62371 | −.08208 | .74196 | .60085 |
| 12 | 1.72141 | 1.64675 | −.07466 | .58254 | .57062 |
| Mean | **1.70966** | **1.63205** | **−.07761** | **.64897** | **.58767** |

On 70,425 positions from 254 opening families, the conditional paired-family
95% interval for KL difference is **[−.09027, −.05187]**; for value MSE it is
**[−.07563, −.03063]**. These intervals condition on the fitted seeds. The report
retains all seed differences and their sample SD separately. Validation families
have been reused in earlier development.

The 62,984-position current-source-novel slice also improves: mean KL difference
**−.05032**, value MSE **−.04839**. Teacher top-1 agreement is less consistent:
the natural-slice mean increases .453 percentage points, while the novel-slice
mean decreases .235 points; both family intervals include zero.

Policy entropy falls only **3.36238 → 3.32191** and mean top-move probability
rises **9.730% → 9.948%**. On the novel slice, top probability falls. This is
not evidence of uniformly spikier or better calibrated predictions.

The registered training probes show larger raw edge-parameter movement:
L2 changes are **263.5–269.3** under global clipping and **397.6–418.0** under
per-group clipping. Across the 101 logged updates in each run, the per-group
edge norm stays below one and its factor is always one. Global edge factors
have medians **.0633–.0828**. These are logged-step distributions; the separate
all-update clipping counter and final post-update gradients retain their own
meaning. Changing all nine clipping groups does not isolate an edge-only cause.

The largest motor's share of visual variance falls from **98.96–99.999%** to
**80.55–98.53%**. Standardized participation rank rises in every seed, but the
number of motors exceeding the fixed variation threshold falls in two seeds.
The leading principal component still explains **96.95–99.71%** of raw visual
variance. The circuit has not acquired a broadly distributed representation.

Complete warm prediction counts include visual encoding and dense heads:

| Prediction batch | Global mean FLOPs | Per-group mean FLOPs | Increase |
|---|---:|---:|---:|
| 1 | 185.12M | 201.43M | 8.81% |
| 32 | 192.35M | 214.85M | 11.69% |

The topology and nominal architecture are identical; learned activity changes
how much zero-skipping work the runtime performs. These counts are separate
from the small optimizer-only timing difference below. This short K8 study
does not replace the earlier 128k-exposure smooth-rate result near 1.49 KL.
The next priority is a separately registered longer-horizon comparison with
propagation held fixed.

Twelve focused analysis checks pass, and all six closed training logs reproduce
their applied clipping factors. The first frozen test bundle lost its package
layout and failed two imports before owner audits began; the preserved recovery
restores the layout and existing dependency path without changing analysis or
learner code. All 216 owner evidence files have member-verified second-worker
archives, alongside the separately verified initial/final checkpoint replicas.

## Optimizer cost

The optimizer-only benchmark uses the qualified Rust library, the exact nine
parameter dimensions, and synthetic gradients. It alternates global/group order
for 48 paired timings per profile after four warmups per mode, on four pinned
CPU cores. Validation jobs were running on separate cores of the same host.

| Synthetic gradient profile | Global mean | Per-group mean | Difference |
|---|---:|---:|---:|
| Bias dominated | 72.858 ms | 73.318 ms | +0.63% |
| All groups below threshold | 72.932 ms | 72.947 ms | +0.02% |

These measurements include validation of parameter arrays, norm reductions and
Adam. They exclude recurrence, input sampling, Python calls, logging and
checkpoint I/O. Shared memory bandwidth and timing noise remain relevant; this
is neither a learning result nor an end-to-end speed measurement. The 240
synthetic updates, including a small smoke check, use **zero Go labels**.
[All timings and build identity](results/group-clipping-overhead-v1.json).

The [Rust example](../crates/fly-core/examples/optimizer.rs) accepts thread count,
neurons, edges, types, input features, motor groups, actions and repetitions:

```bash
RUSTFLAGS='-C target-cpu=native' cargo run --release -p fly-core --example optimizer -- \
  4 165122 15270273 11752 3574 2129 82 48
```

Pin it to available CPU cores when comparing timing results. It constructs a
synthetic graph to size optimizer arrays; it does not simulate the fly circuit.
