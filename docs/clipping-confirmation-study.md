# Clipping confirmation at 128k exposures

Per-parameter-group clipping improves mean full-validation policy KL
**1.64356 → 1.53649** and value MSE **.60288 → .55816**. Policy improves in all
three fresh seeds, satisfying the registered candidate rule. Retain it as the
optimizer control for the next separately qualified study. Counted warm B1
prediction work rises **5.73%**; this does not establish an advantage over a CNN.

All six CPU endpoints and their followups are complete at **4,000 B32 updates**,
using paired seeds **13/14/15**. Each receives 128,000 labeled-position exposures;
the total is **768,000**. [Complete result](results/clipping-confirmation-v1.json),
[verified evidence closure](results/clipping-confirmation-closure-v1.json).

The completed [32k-exposure discovery](group-clipping-study.md) improved mean
policy KL by .07761 and value MSE by .06130, but raised counted warm B1
prediction FLOPs by 8.81%. The new experiment tests whether the optimizer
benefit persists at a longer fixed horizon. Both seeds and horizon differ
from discovery, so differences between the two studies cannot isolate the
effect of training longer.

## Fixed-endpoint results

Lower KL and MSE are better. These are all 70,425 natural validation positions,
covering 254 opening families; no endpoint is selected by its best observed step.

| Seed | Global KL | Per-group KL | Global MSE | Per-group MSE |
|---|---:|---:|---:|---:|
| 13 | 1.674849 | 1.527102 | .595777 | .539034 |
| 14 | 1.618026 | 1.553017 | .581559 | .564239 |
| 15 | 1.637798 | 1.529350 | .631313 | .571215 |
| Mean | 1.643557 | 1.536490 | .602883 | .558162 |

The paired mean KL difference is **−.10707**, with conditional opening-family
95% interval **[−.12566, −.07015]** and paired-seed sample SD **.04139**.
The MSE difference is **−.04472**, interval **[−.06742, −.03575]**, seed SD
**.02379**. Family intervals condition on these trained weights; three seeds
do not characterize all training randomness. On the fixed 62,984-position
source-novel slice, mean KL improves **1.58683 → 1.52293** and MSE
**.63359 → .59409**. Validation families have been reused during development.

Teacher top-move agreement increases **19.68% → 21.16%**. Policy entropy falls
**3.30413 → 3.20173** and mean peak probability rises **.10442 → .12255**.
The policy becomes sharper while improving KL, but motor diversity remains
limited: candidate varying-motor counts are **125 / 148 / 85**, versus
**97 / 78 / 111** for global clipping. The leading component still explains
**90.96–99.70%** of raw current-minus-neutral motor variance. These probes do
not establish biological function or resolve the representation bottleneck.

Complete warm prediction arithmetic on the same 64 training probes changes
as learned activity changes, despite identical topology and inference rules.

| Batch | Global FLOPs / position | Per-group FLOPs / position | Increase |
|---|---:|---:|---:|
| 1 | 163.81M | 173.19M | 5.73% |
| 32 | 171.60M | 186.40M | 8.63% |

These counts include visual encoding and dense heads. Equal exposure and
fixed inference equations do not imply equal executed sparse work. There is
no new matched-CNN comparison, playing-strength panel or untouched-test result.

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
all initial and final checkpoints have verified recovery copies. Followup workers
waited for trainer exit and the verified final replica before using the released
lane for full validation, motor probes and arithmetic. All 306 qualification
evidence files, including continuation checkpoint arrays, have verified
owner/peer archives. TPU remains paused.

## Closing the study

The root host is near its own-file allowance. Retain the existing 100 GiB
cap, 64 GiB free-filesystem floor and 96 GiB available-RAM floor, including
reservations. A reservation protected additional coordinator log space.
Bulk response arrays and checkpoint payloads stay on workers.

Worker 2 is the registered analysis host. Its frozen analysis tools pass the
same tests, and the existing PyArrow dependency was copied and checked file by
file; reading all 165,122 canonical annotation rows succeeds. Common novelty
and probe selections were prepared there with verified identities. After all
endpoints finished, the [frozen collector](../scripts/collect_clipping.py) performed
the owner-side checkpoint audits and relayed completed evidence through root
pipes to worker 2. All **183** archived evidence files were checked member by
member on a second host. The frozen analyzer succeeded, and its full report
has verified copies on workers 2 and 3. Root retains small reports and receipts. Its
[operational configuration](../configs/clipping-confirmation-collection-v1.json)
binds the collector, analysis and endpoint-helper hashes. Five checks cover
corrupt and conflicting archives, path containment, idempotent extraction,
and waiting for actual process exit and the followup lock. They pass on worker
2. An earlier remote-wrapper syntax failure occurred before tests or collector
launch; its record is retained. The corrected wrapper launches the unchanged
collector. This does not alter the scientific registration.
The default `finish_study.py collect` copies bulk arrays to root and must not
be used for this study. Collection, paired analysis and the optimizer decision
are now closed; the registered endpoints were not rerun.

The separate [loss-balance audit](loss-balance-study.md)
measures gradients at completed discovery checkpoints. Its nine cases,
including two numerical failures, are closed and archived. No resulting
loss-routing or precision change has been applied to this confirmation.
