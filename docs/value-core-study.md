# Separate value-head learning from circuit learning

The completed [clipping confirmation](clipping-confirmation-study.md) supports
per-parameter-group clipping as the next optimizer control. The
[loss-balance audit](loss-balance-study.md) found that value gradients dominate
and oppose policy gradients at the shared circuit on one fixed training probe.
The registered comparison tests that optimization pressure directly, before changing
visual attachments or recurrence. A learnability benefit is not yet established.

## One gradient-routing coefficient

For shared parameters \(\theta\), policy-head parameters \(\phi_P\), and
value-head parameters \(\phi_V\), use

\[
g_\theta=\nabla_\theta L_P+\lambda\nabla_\theta L_V,\qquad
g_{\phi_P}=\nabla_{\phi_P}L_P,\qquad
g_{\phi_V}=\nabla_{\phi_V}L_V.
\]

The registered scales are **1, .1 and 0**: ordinary shared learning, reduced
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

## Scientific comparison

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

The [study](../configs/value-core-study-v1.json),
[nine trials](../configs/value-core-trials-v1.json) and
[analysis](../configs/value-core-analysis-v1.json) are now frozen and deployed.
Each seed uses all three arms, with identical numerical initial arrays and
sampling streams within the seed:

| Arm | Shared circuit gradient | Both output heads |
| --- | --- | --- |
| full | policy + value | Full gradients |
| reduced | policy + .1 × value | Full gradients |
| policy-only | policy | Full gradients |

Every endpoint receives **4,000 B32 updates / 128,000 labeled exposures**;
all nine total **1,152,000** scientific training exposures. The primary metric
is full-validation KL on **70,425** positions, with two declared contrasts:
reduced minus full and policy-only minus full. Keep value MSE, teacher top-1,
entropy/peak probability, the **62,984** source-novel positions, fixed training
motor probes and actual B1/B32 prediction FLOPs. Family bootstrap intervals
condition on fitted weights; report seed differences and their SD separately.
There is no multiplicity correction.

Each candidate is independently provisional only if every paired seed improves
KL and mean value MSE does not worsen. Report every tradeoff and failed case.
This report selects no winner between eligible candidates; promotion requires
a separate fresh-seed confirmation.

Six learners for seeds 16/17 are running on two disjoint 24-core lanes per
worker. The three seed-18 learners wait for their preceding same-lane validation,
motor probe and prediction count to finish. Arms rotate across hosts. Actual
process checks verified all **150** learner threads, the loaded native-library
hashes, exact within-seed initial pairing, and all six initial checkpoint peers.
A second observation confirmed optimizer-step growth on the same six PIDs and
all nine live followup processes. Final scientific results remain pending.

The initial IO draft retained the earlier clipping protocol's digest. It was
never launched. IO v2 corrects that reference before staging; the launcher
checks that both plans point to the same numerical protocol. The original
draft and correction are retained without changing seeds, settings or source.

The comparison retains B's current-board spherical input, dense 2,129-motor
projections, K8 hard recurrence and reset state, with the confirmed per-group
optimizer settings. Matched-CNN and playing-strength claims require their own
fair comparison. Persistent state and precision changes remain separate.

The analysis/control bundle passes **20 tests**, including both contrasts,
family/seed weighting, objective-tag pairing, process/lock waits and transient
SSH observation retries. It also handles an explicitly queued trial before its
status file exists; a missing unqueued trainer remains an error. The preceding
19-test bundle is retained. These changes perform no learner updates.

The first launch stopped at bundle validation because the controller lacked the
raw V0 game payloads; all three workers retained complete copies. No learner
started in that attempt. Restoring **241,128,228 bytes / 10,256 games** from
worker 2 passed every registered checksum, after which the unchanged deployment
resumed. Dataset, split, sampler, source, seeds and endpoints were preserved.
Registration, both test bundles, attempts and live audits have **276**
member-verified archive entries on two workers.
[Registration, launch and evidence record](results/value-core-registration-v1.json).

TPU stays paused. Scientific checkpoint and bulk-response payloads stay on
workers. The controller retains the qualified runtime and restored V0 files
needed by the existing deployment interface, plus bounded control/log storage.
All original shared-RAM caps and buffers remain in force.
