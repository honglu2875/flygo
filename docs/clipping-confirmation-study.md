# Clipping confirmation at 128k exposures

Six CPU learners compare global and per-parameter-group clipping at **4,000
B32 updates**, using fresh paired seeds **13/14/15**. Each endpoint receives
128,000 labeled-position exposures; the total is **768,000**. Training and
queued endpoint probes are in progress. No confirmation result is available.

The completed [32k-exposure discovery](group-clipping-study.md) improved mean
policy KL by .07761 and value MSE by .06130, but raised counted warm B1
prediction FLOPs by 8.81%. The new experiment tests whether the optimizer
benefit persists at a longer fixed horizon. Both seeds and horizon differ
from discovery, so differences between the two studies cannot isolate the
effect of training longer.

## Fixed comparison

The only difference within each paired seed is clipping scope. Both arms
start from identical numerical arrays and sampler state, with fresh Adam
moments. They retain the full CNS topology, B's current-board spherical
montage, dense projections from all 2,129 motors, eight hard-rate passes,
reset state .01, peak rate .03, 100-update warmup, epsilon 1e-6, bias-rate
multiplier .01 and clipping threshold one. No discovery weights are adopted.

The [study](../configs/clipping-confirmation-study-v1.json),
[numerical protocol](../configs/clipping-confirmation-numerics-v1.json),
[trial plan](../configs/clipping-confirmation-trials-v1.json) and
[analysis](../configs/clipping-confirmation-analysis-v1.json) were frozen
before any scientific update. Checkpoints are at updates 0 and 4,000; fixed
slice evaluations and signal diagnostics occur every 500 updates. Final
checkpoints are chosen by the registered horizon.

Full validation uses all **70,425** V0 validation positions. Report policy
KL, value MSE, teacher top-move agreement, policy entropy and peak probability;
also report the fixed 62,984-position source-novel slice. Average paired
differences per position before resampling opening families 1,000 times with
seed 709. Report conditional family intervals and variation across training
seeds separately. The provisional-candidate rule remains lower policy KL in
every paired seed and nonworse mean value MSE.

These validation families have been reused during development. This is an
equal-exposure optimizer study; actual prediction work can change as activity
learns. Measure complete warm B1/B32 prediction arithmetic on the existing 64
training probes, and motor responses on the same 256 training positions as
discovery. Confidence alone does not establish better predictions. This
study is not a matched-CNN comparison or an untouched-test result.

## Qualification and execution

The learner source remains `1f19481174f3bc8e5449`. All **12** fresh-seed
full-CNS numerical cases and **six** exact fresh-process recovery cases pass,
including B1/B32 Go prediction, sampler, schedule, moments, next update and
rejection of a different clipping mode. The previous source's 56 Rust / 121
Python checks and exact legacy regression remain applicable. Qualification
uses **2,112 native + 1,152 JAX-reference = 3,264** additional V0 update
exposures, separate from scientific training.

The analyzer now accepts registered seeds and horizons, while enforcing the
same pairing, decision direction and family resampling. Its implementation
hash is bound by the new analysis contract. All **13** focused analysis
checks pass on root and worker 2. The learner implementation is unchanged.
[Engineering record](results/clipping-confirmation-engineering-v1.json).

| Worker | CPUs 32–55 | CPUs 92–115 |
|---|---|---|
| 1 | Global, seed 13 | Per-group, seed 15 |
| 2 | Per-group, seed 13 | Global, seed 14 |
| 3 | Global, seed 15 | Per-group, seed 14 |

Actual learner commands and every native thread's affinity were checked;
all six initial checkpoints have verified recovery copies. Followup workers
wait for trainer exit and the verified final replica before using the released
lane for full validation, motor probes and arithmetic. All 306 qualification
evidence files, including continuation checkpoint arrays, have verified
owner/peer archives. TPU remains paused.

## Closing the study

The root host is near its own-file allowance. Retain the existing 100 GiB
cap, 64 GiB free-filesystem floor and 96 GiB available-RAM floor, including
reservations. A live reservation protects additional coordinator log space.
Bulk response arrays and checkpoint payloads stay on workers.

Worker 2 is the registered analysis host. Its frozen analysis tools pass the
same tests, and the existing PyArrow dependency was copied and checked file by
file; reading all 165,122 canonical annotation rows succeeds. Common novelty
and probe selections are prepared there with verified identities. After all
endpoints finish, perform the owner-side checkpoint audits and relay completed
evidence through root pipes to worker 2. Keep member-verified peer archives,
then run the frozen analyzer. Root receives small reports and receipts.
The default `finish_study.py collect` copies bulk arrays to root and must not
be used for this study. Final collection, paired analysis and the scientific
decision remain pending.
