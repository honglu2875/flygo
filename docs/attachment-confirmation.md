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

At 2026-09-14 22:04 UTC, all six CPU learners are training at updates 60–70
of 1,000. Full-validation helpers in `attachment-confirmation-validation-v1`
wait for each trainer to exit and its final checkpoint replica to be verified.
No scientific confirmation endpoint is available yet.

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
