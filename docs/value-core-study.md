# Separate value-head learning from circuit learning

The completed [clipping confirmation](clipping-confirmation-study.md) supports
per-parameter-group clipping as the next optimizer control. The
[loss-balance audit](loss-balance-study.md) found that value gradients dominate
and oppose policy gradients at the shared circuit on one fixed training probe.
The next intervention tests that optimization pressure directly, before changing
visual attachments or recurrence. A learnability benefit is not yet established.

## One gradient-routing coefficient

For shared parameters \(\theta\), policy-head parameters \(\phi_P\), and
value-head parameters \(\phi_V\), use

\[
g_\theta=\nabla_\theta L_P+\lambda\nabla_\theta L_V,\qquad
g_{\phi_P}=\nabla_{\phi_P}L_P,\qquad
g_{\phi_V}=\nabla_{\phi_V}L_V.
\]

The proposed scales are **1, .1 and 0**: ordinary shared learning, reduced
value pressure, and a value head trained on policy-learned features. Shared
parameters include edge strengths, type leak/bias, input gains and readout
gains. Both head gradients remain fully trained at common parameters; later
trajectories can diverge as their representations change. Adam and clipping
operate after this routing. Under global clipping, changing the shared norm
can also change head updates; per-group clipping avoids that particular coupling.

This is a specified gradient rule, rather than multiplying the reported value
loss or weakening the value head. Report the original policy CE/KL and value
MSE throughout. Prediction equations, topology, sensory mapping, dense motor
heads and counting rules stay fixed. Generic embedding objectives retain their
literal cotangents. No new biological mechanism is claimed.

## Implementation and gates

`RustFly(..., value_core_scale=.1)` and the CPU JAX reference implement the
same rule; the training CLI exposes `--value-core-scale`. Rust scales the
value contribution at the head-to-embedding derivative boundary. JAX uses an
identity forward expression with a scaled embedding derivative. At scale one,
the previous numerical arrays and arithmetic are preserved.

Nondefault checkpoints carry the exact FP32 coefficient and an objective
version. Loading into a different gradient contract must fail before changing
the model or sampler. Qualification and launch checks also bind the coefficient;
old default evidence cannot authorize a nondefault trial. The CLI rejects
nondefault TPU training until it receives a separate actual-device qualification.

The [engineering contract](../configs/value-core-engineering-v1.json) is frozen.
The new native package passes **56 Rust tests** (one existing ignored test)
and **178 Python tests**. Seven new tests cover independent finite differences,
unchanged heads/predictions, scale-zero policy gradients, Rust/JAX derivatives
and updates, masked heads, exact recovery and rejected contract mismatches.
Two additional deployment tests reject reuse of unscaled evidence.

The first isolated Python bundle ran 169 tests with eight setup errors: it
lacked PyArrow and existing test fixtures. Its seven new gradient tests passed.
The complete bundle uses the same learner bytes and adds the missing dependency
and fixtures; all 178 tests pass. Both attempts are retained. Invalid extreme
coefficients are correctly rejected; the oversized-number fixture emits a
NumPy cast warning. That warning does not occur for the proposed coefficients.

An [exact full-CNS regression](../configs/value-core-regression-v1.json) passes
against the frozen prior native package, in both clipping modes. It checks
B1/B32 inference, gradients, parameters, moments, sampling,
peak and warmup transitions, and restoration of real prior-runtime checkpoints.
All four paired cases agree bit for bit. It used **896 engineering update
exposures**, separate from any science. Completed test and regression evidence
has **226** member-verified archive entries on second workers; the build and
runtime package are independently verified on three workers.
[Build, tests, compatibility and evidence receipt](results/value-core-engineering-v1.json).

## Next scientific decision

All **18** full-CNS Rust/JAX numerical cases and **nine** fresh-process recovery
cases pass for seeds **16/17/18**, each with scales **1 / .1 / 0**. The
[numerical protocol](../configs/value-core-numerics-v1.json) uses three updates
per case; the [recovery protocol](../configs/value-core-io-v2.json) verifies
exact continuation and rejects incorrect scale/mode restores. Their engineering
exposure is **4,896**, in addition to 896 default-compatibility exposures:
**5,792 total**, separate from science. Tolerances are unchanged.

All **159** qualification archive entries, including continuation checkpoints,
have verified peer copies. Worker 2 additionally validates all nine exact
prospective deployment contracts, including common initial arrays/samplers and
source, coefficient, schedule and recovery identities.
[Complete qualification record](results/value-core-qualification-v1.json).
Scientific trials are **not registered or launched** yet.

The initial IO draft retained the earlier clipping protocol's digest. It was
never launched. IO v2 corrects that reference before staging; the launcher
checks that both plans point to the same numerical protocol. The original
draft and correction are retained without changing seeds, settings or source.

The subsequent comparison will retain B's current-board spherical input,
dense 2,129-motor projections, K8 hard recurrence and reset state, with the
confirmed per-group optimizer settings. Register paired initializations,
sampling streams, fixed exposure endpoints and analysis before scientific
updates. Evaluate full validation, source-novel positions, motor responses and
actual prediction work. A policy gain with worse value is a tradeoff, not an
unqualified improvement. Matched-CNN and playing-strength claims need their
own fair comparison. Persistent state and precision changes remain separate.

TPU stays paused. New runtime and checkpoint payloads stay on workers within
the existing shared-RAM limits; root receives bounded control records.
