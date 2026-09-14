# Spherical input confirmation

Registered after the [seed-1 screen](attachment-study.md), before scientific
training of seeds 2 and 3. The history arm's value improvement motivates this
confirmation; the seed-1 data are exploratory and will be identified separately.

Run the same history/current/neutral comparison with learner and sampler seeds
2 and 3. Keep the geometry, context allocation seed, topology, sign convention,
K8 rate equations, 2,129 individual readouts, initialization recipe, loss and
optimizer unchanged. Only the head/sampler seed changes; this is not a jitter
of initial synaptic strengths. Each arm receives exactly 1,000 updates of 32
positions, with rate .03, 100-update warmup, epsilon 1e-6, clip 1 and bias rate
multiplier .01. Adopt the existing seed-1 endpoints without extending them.

[The launch plan](../configs/attachment-confirmation-v1.json) names every run,
CPU lane, replica destination and qualification report. Rotate modes across
workers so each mode occupies each worker once over the three seeds. Production
retains its 64 physical cores per host. New runs use the two disjoint 24-core
research lanes. No TPU is used or reserved.

Before launch, qualify each actual new seed and all three inputs at B32/K8
against the independent JAX CPU reference: states, losses, every gradient and
three free-running Adam updates. The seed-1 epsilon and tolerances remain fixed.
A failed numerical gate is retained and prevents that seed's scientific launch;
it is not permission to change its optimizer during this confirmation.

Retain initial/final checkpoints. Their verified recovery copies live on another
worker, streamed through the main node's pipes. Transfer byte counts and SHA-256
are checked at both ends; destination receipt publication precedes marking the
source recoverable. The main node stores control records and analysis, preserving
its remaining storage space. Keep the 100 GiB own-file cap, 64 GiB free-SHM floor
and 96 GiB available-RAM floor on every host, including reservations.

Evaluate the same fixed 2,048-position slices and endpoint visual perturbations,
then all 70,425 validation positions. Retain aligned per-position metrics and
opening-family IDs. Report the three paired contrasts separately for each new
seed, their mean over the two confirmation seeds, and the three-seed result
including the exploratory reference. Family bootstrap intervals condition on
the fitted weights; seed variation is reported separately. No final-test labels
or checkpoint selection enter the study. Reuse the frozen reduced-input novelty
masks with their pre-render limitation.

Repeat the 256-family motor probe to ask whether DNg30 concentration persists
across learned weights. Do not fit action groups from a favorable single seed.
Prediction work is activity-dependent: recount each final model before any
comparison with a conventional network. Larger retinal allocation, output
selection, neuronal physiology and optimizer changes remain separate studies.

## Engineering and launch status

At 2026-09-14 23:05 UTC, all six CPU learners have completed 1,000 updates.
Their initial/final checkpoints have verified recovery copies on another
worker. All six full validations and subsequent motor/arithmetic probes have
completed. The main node holds their small records and response/metric arrays;
the new checkpoint payloads remain on workers.

[Qualification and launch evidence](results/attachment-confirmation-engineering-v1.json)
records successful B32/K8 full-circuit checks for seeds 2 and 3 in all three
modes, at the original tolerances and epsilon. Each gate takes about 196–197
seconds. The deployment rejects a deliberately mismatched seed qualification.
Within each seed, all 31 initial arrays, the sampler state and training contract
match across modes. All six initial checkpoints have verified worker replicas.

The existing `deploy_research.py` accepts the attachment contract and the
registered seed-specific qualifications. `finish_study.py` also checks the
input map, mode and qualification before validation. The coordinator freezes
its replication helper. The learner remains the original immutable environment
`861175bf88bf1ba6b4f5`; relay utilities use `b6db2960698424122c5f`.

Eight replication tests pass, including truncated/extra data, immutable conflicts
and recovery after interrupted receipt publication. A real 189,666,024-byte
checkpoint copy from w1 to w2 verifies both hashes and receipts in about ten
seconds, including an idempotent retry and source/destination checks. It creates
no checkpoint file on root and preserves the prior study's checkpoint receipt.
This is an operational check, not a sustained transfer benchmark.

The two unsuccessful readiness calls started no learners: one used the system
Python interpreter instead of the qualified Python 3.12, and one hid the fleet
CPU allocation by pinning the launcher before its inventory. Both were corrected
without changing the study, numerical gates or resource limits.

## Results

Each row uses all 70,425 validation positions, grouped into 254 opening
families. Lower KL and MSE are better. Seed 1 is the adopted exploratory screen;
seeds 2 and 3 are the registered confirmations. The horizon remains 32,000
labeled-position exposures per model.

| Seed | Input | Policy KL | Value MSE | Teacher top-1 |
|---|---|---:|---:|---:|
| 1, exploratory | History | 1.66743 | .62807 | 19.13% |
| 1, exploratory | Current | 1.65923 | .84937 | 20.44% |
| 1, exploratory | Neutral | 1.65202 | .81902 | 21.33% |
| 2 | History | 1.67927 | .65496 | 19.05% |
| 2 | Current | 1.68454 | .59784 | 18.99% |
| 2 | Neutral | 1.65559 | .80754 | 20.72% |
| 3 | History | 1.70138 | .77722 | 18.04% |
| 3 | Current | 1.68794 | .64212 | 18.24% |
| 3 | Neutral | 1.64341 | .83104 | 20.30% |

Across the **two new seeds**, history-minus-neutral changes KL by **+.04083**
and value MSE by **−.10320**; current-minus-neutral gives **+.03674** and
**−.19931**. Conditional paired-family 95% intervals for these MSE differences
are [−.27266, −.02110] and [−.32599, −.14319]. Both visual inputs improve value
in each new seed, while policy KL and top-1 agreement worsen. On the common
61,541-position reduced-source-novel slice, the value reductions remain, but
the conditional policy-difference intervals include zero.

Including exploratory seed 1, mean history/current-minus-neutral MSE changes
are −.13245/−.12276. Current versus history reverses its value ordering across
seeds: +.22130, −.05712, −.13510. The three-seed average is +.00969, despite
current doing better in both new seeds. Family intervals condition on fitted
weights and do not resolve this training-seed variation. These observations
support a visual value signal at this short horizon, without establishing that
historical montage or current input is generally superior.

The fixed training-family probe also repeats the concentration finding, but
**the dominant cell is not invariant across seeds**. Raw rates are measured
before learned readout gains, with identical nonvisual context in each visual
perturbation.

| Seed | Input | Varying motors, SD > 1e-6 | Dominant cell | Raw visual variance | Unit-variance rank |
|---|---|---:|---|---:|---:|
| 1 | History | 415 | DNg30, body 10123 | 99.9773% | 7.13 |
| 1 | Current | 621 | DNg30, body 10123 | 99.9621% | 5.02 |
| 2 | History | 306 | DNg30, body 10237 | 99.9898% | 5.20 |
| 2 | Current | 336 | DNg30, body 10237 | 99.9908% | 5.07 |
| 3 | History | 252 | DNpe018, body 69173 | 92.9414% | 4.26 |
| 3 | Current | 243 | DNpe018, body 69173 | 96.3677% | 4.65 |

These are model responses on 256 training positions, not physiological
recordings or evidence for permanent action partitions. Head/sampler seeds
vary; initial core weights do not. Weak distinct signals survive outside the
dominant cell, but the present trained representation remains concentrated.

Recounted unpruned K8 warm B1 arithmetic spans **171.2–184.9M FLOPs** across
the nine endpoints; B32 means span 177.4–190.4M per position. Rendering is
included. These runs cannot inherit the earlier small-CNN comparison, and
the separate optional CPU dependency optimization has no complete FLOP ledger
yet. No matches or final-test evaluation were performed.

The [result and provenance](results/attachment-confirmation-v1.json) contain
all paired contrasts, both seed aggregates, conditional intervals, seed
variation, parameter changes, response hashes and arithmetic counts. The
analysis reproduces the adopted seed-1 metrics and family intervals exactly.
The next input factor remains one larger board per eye, followed by separately
qualified persistent state. Output attachment and physiology keep separate
contracts. TPU remains paused.
