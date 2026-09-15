# Policy and value pressure on the shared circuit

The value objective produces much larger raw shared-circuit gradients than
the policy objective on the fixed training probe. At the three passing
per-group-clipping endpoints, the norm ratio is **8.20–29.38**, and the two
gradients oppose each other. This motivates testing how strongly value loss
updates the shared representation. It does not yet establish a cause of poor
validation KL or justify changing the running confirmation experiment.

The native decomposition audit covers all **nine** registered cases: **seven
pass and two fail**. Both failures involve one shared type-bias coefficient.
An independent audit closely reproduces the discrepancy through long FP32
summation. The failed numerical gates remain failed; no tolerance or production
learner changed. [Native results](results/loss-balance-v1.json),
[independent bias audit](results/bias-reduction-audit-v2.json).

## Measurement

The [plan](../configs/loss-balance-v1.json) uses the first 32 positions of the
existing 256-position training-family probe, without augmentation or selection
by gradients. It evaluates three shared initializations and both clipping
discovery endpoints for seeds 10/11/12 at 1,000 updates. All use the same full
CNS, current spherical montage, dense motor heads and K8 hard recurrence.
The longer, fresh-seed clipping confirmation is a separate experiment.

For policy cross entropy and value MSE, define

\[
g_P=\nabla_\theta L_P,\qquad g_V=\nabla_\theta L_V,\qquad
g_J=\nabla_\theta(L_P+L_V).
\]

The [probe](../scripts/probe_loss_balance.py) keeps the actual native policy
derivative, cancels the value residual by setting its target to the exact
native prediction, and obtains the value-only gradient through the native
linear-score VJP. It checks that \(g_P+g_V\) reproduces \(g_J\) for all nine
parameter groups at **rtol .003 / atol 5e-6**. It also verifies the inactive
head's gradients are exactly zero. Successful cases fingerprint all parameters,
Adam moments, optimizer step and clipping tag before and after the read-only
probe. Source checkpoint hashes are bound by the plan.

The following table concatenates the five shared groups: edges, type leaks,
type biases, input gains and readout gains. Head coefficients are excluded.
These are Euclidean gradient measurements in the stored parameter coordinates;
the large type-bias gradients dominate the aggregate.

| Seed | Endpoint | Value/policy norm | Cosine | \(g_P\cdot(g_P+g_V)\) | Native gate |
|---|---|---:|---:|---:|---|
| 10 | Shared initial | 5.197 | .0308 | .00903 | Pass |
| 11 | Shared initial | 6.159 | −.0744 | .00514 | Pass |
| 12 | Shared initial | 5.940 | −.0413 | .00797 | Pass |
| 10 | Global, 32k exposures | 19.175 | −.8049 | −25.97073 | Fail: bias |
| 10 | Per-group, 32k exposures | 8.202 | −.5707 | −7.08295 | Pass |
| 11 | Global, 32k exposures | 100.049 | −.0098 | .00485 | Pass |
| 11 | Per-group, 32k exposures | 8.302 | −.8441 | −11.67789 | Pass |
| 12 | Global, 32k exposures | 44.784 | .0983 | 1.70400 | Fail: bias |
| 12 | Per-group, 32k exposures | 29.384 | −.4429 | −8.39387 | Pass |

A negative last numeric column means an infinitesimal **joint SGD** step would
increase this batch's policy loss. It is not an actual Adam update: clipping,
parameter learning-rate multipliers, first-moment history and denominator
updates all matter. The saved-Adam-diagonal and motor-cotangent diagnostics in
the successful case records have the same limitation. Do not interpret the
single 32-position batch as a population estimate or a validation prediction.
Rows with failed gates report the observed native arrays, with their numerical
qualification explicitly unresolved at the original gate.

The original owner-1 process stopped on its second case. A registered
[completion attempt](../configs/loss-balance-completion-v1.json) moved the
unattempted per-group seed-12 endpoint first, then repeated the two seed-10
cases with the same source, positions and tolerances. The missing case passes;
all **27** gradient arrays in each repeated case are exactly equal to their
originals. All original and repeated results remain in the archives.

## Why two numerical checks failed

Each failure has one out-of-tolerance coefficient among 11,752 type biases:
canonical index 8,850, annotated **Mi4**, shared by **1,772 neurons**. Its native
bias reduction performs **453,632 FP32 additions** for K8 and B32. See the
[backward reduction](../crates/fly-core/src/recurrent.rs).

| Global endpoint | Native policy + value bias | Native joint bias | Absolute difference | Original tolerance |
|---|---:|---:|---:|---:|
| Seed 10 | −.000588678929 | −.000574697682 | .000013981247 | .000006724093 |
| Seed 12 | −.003664201708 | −.003647030331 | .000017171376 | .000015941091 |

The independent diagnostic first exactly reproduces the original joint
gradients and then checks separately supplied head cotangents against the
native shared-circuit VJP. All **30** head-boundary group checks are exact.
It reconstructs a FP64 bias adjoint at the native state trace and, separately,
sums the reconstructed Mi4 messages serially in FP32. The serial FP32 values
are within **2.5e-8** of the native results across both cases and all three
objectives. This strongly supports accumulation error as the explanation.
The reconstructed messages are not captured native FP32 messages, so this is
not a bitwise proof of the internal cause.

The independent full-bias comparison retains **one failure** for the isolated
seed-10 value objective. Both actual joint-training bias vectors pass that
reference comparison. The diagnostic is a bias adjoint, not a new full-gradient
reference or evidence that altered propagation would fix learnability. FP64
bias accumulation could be qualified separately; it was not adopted here.

## Evidence and follow-up

The initial small fixture failed because the diagnostic confused the pre-tanh
linear score with the value. Correcting the callback interpretation gives two
passing checks, including legal masking, nonunit target mass, nonzero Adam
state and detection of an intentionally corrupted decomposition. The first
independent bias-adjoint attempt had a transposed policy-matrix contraction;
that failed attempt and the corrected source are both retained. Neither
correction changed the learner. [Engineering and archive receipt](results/loss-balance-engineering-v1.json).

All **83** native archive entries have owner and member-verified peer copies,
including every failed case and both repeats. The archives total 2,074,593,280
bytes; bulk gradient payloads never pass through a root file. The small reference
audits, failed fixtures and source receipts also have a second-host copy.
The complete accounting is **2,592 native forward position evaluations**, **1,344
native backward position evaluations**, **192 independent bias-adjoint position
views**, and **10 synthetic fixture updates**. There are **zero new V0 optimizer
updates**. These are evaluation counts, not FLOPs.

A next candidate is a value-to-circuit gradient scale \(\lambda\):

\[
g_{\text{shared}}=g_P+\lambda g_V,
\qquad g_{\text{value head}}=\nabla L_V.
\]

Prediction and full-strength value-head training would stay the same. Values
such as 1, .1 and 0 would separate ordinary shared training, reduced pressure,
and a value head trained on policy-learned features. Clipping confirmation is
now complete and supports per-group clipping as the control. The
[separate implementation and qualification](value-core-study.md) passes its
synthetic checks, exact legacy regression, and all 18 actual-seed numerical /
nine recovery cases. The scientific comparison is the next step. Keep any precision change separate
from the loss-routing comparison. TPU remains paused.
