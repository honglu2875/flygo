# Clipping scope

The optimizer comparison is qualified and the six CPU trials have been launched.
There is no completed learning result yet. The network, spherical input B,
dense motor projection, K8 hard recurrence, reset state, rate, epsilon and
32,000-position training horizon remain fixed.

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
Each host runs two learners on separate 24-core lanes. TPU stays paused.

[Analysis](../configs/group-clipping-analysis-v1.json) uses all 70,425 validation
positions, paired opening-family resampling and separate seed variation. Policy
KL is primary; value MSE, teacher agreement, entropy and top probability are
retained alongside motor responses and complete prediction arithmetic. Sharper
policies alone do not establish better learning. A provisional candidate must
improve KL in every seed and must not worsen mean value MSE. All fixed endpoints
and failures will be reported; no CNN comparison is inherited from this screen.

Isolated optimizer overhead remains to be measured. Training throughput and
prediction FLOPs are separate measurements.
