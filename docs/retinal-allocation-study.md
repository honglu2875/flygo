# One current board per eye

Registered design, 2026-09-14, following the completed
[paired input confirmations](attachment-confirmation.md). This is stage C of
the [temporal vision study](temporal-vision-study.md). It changes retinal
allocation while retaining current-board information, all context, outputs,
fixed topology, recurrence and optimizer. TPU remains paused.

The reference B places the current board in both established patches of each
eye. Their centers are azimuth −25 degrees, elevation −30/+30 degrees, each
with halfwidth 25 degrees. C uses **one patch per eye at (−25, 0)**, with
azimuth/elevation halfwidths **(25, 55) degrees**: exactly the bounding rectangle
of those two patches. This fills their 10-degree vertical gap and spreads one
9×9 board over their combined extent. It changes angular aspect ratio as part
of the allocation factor; it does not invent a measured optical registration.

The common inferred-column spherical chart, orientation, 3,490 photoreceptor
identities and side assignments remain identical. Both eyes receive lag zero
in the current player's stone perspective. Keep the compact Gaussian kernel
at sigma 5 degrees, cutoff 10 degrees and four neighboring board points;
keep its original row normalization, neutral .5 and own/opponent .95/.05
luminances. Do not add a gain correction to equalize the different maps'
drive. Report illumination and drive differences by eye/receptor type.

All 84 context features retain exactly the existing assignment to 11,530
nonvisual sensory neurons. The 2,129 anatomical output features, initial
parameters, input/readout gains, loss and K8 reset execution remain fixed.
The source attachment is `ports/spherical-context-v1/attachment.npz`, SHA-256
`9ce4b1d99e499871a540d06f06dccc563e2e9c77e9ad277a6f2399a04345e087`.
The new map gets its own immutable attachment identity.

## Gates before learning

1. Reconstruct the reference sampler from the generalized builder and require
   every original index, weight and direction byte to match the frozen map.
   The new map must preserve all sensor, motor, context and native port arrays.
2. Require every board point to be observed in each eye and per-eye sampling
   rank 81 at the existing relative singular-value threshold 1e-6. Record
   singular values/condition, illumination, kernel mass and spatial footprints.
   A geometry failure is retained; it does not authorize unreported map search.
3. Check identical current imagery at identical bilateral directions, absence
   of older-frame dependence, neutral response and local single-stone support.
   Use the same 256 training-family probes to describe drive; no teacher labels
   or final-test inputs select the geometry.
4. Freeze source and map, then qualify actual B32/K8, epsilon 1e-6, rate .03,
   clip 1 and bias multiplier .01 against independent JAX CPU: states, losses,
   all gradients and three free-running Adam updates for each intended seed.
   Preserve existing tolerances. Qualification failures prevent dependent
   training; no optimizer change belongs inside this input comparison.
5. Verify identical initialization/ports/sampler against the paired B reference,
   direct versus Go-interface predictions, and fresh checkpoint continuation.
   Any changed CPU implementation must pass numerical regression against the
   frozen reference before adopting its already trained B endpoints.

## Learning and analysis

Subject to these gates, compare C at head/sampler seeds 1, 2 and 3 with the
existing B current-input endpoints. Fix 1,000 updates × B32 = 32,000 labeled
exposures, rate .03, 100-update warmup, epsilon 1e-6, clip 1 and bias multiplier
.01. Initial core strengths are not randomized by these seeds. A launch plan
will bind qualified source/map hashes, all run IDs and worker replicas before
the first scientific update. Do not extend or retune the B endpoints.

Retain the same fixed-slice learning curves, final full validation of 70,425
positions, paired opening-family contrasts and the common reduced-source-novel
slice. Report per-seed differences and pooled conditional family intervals
separately from training-seed variation. The novelty audit precedes FP32
rendering; it does not certify the absence of rendering collisions.

Repeat initial/final visual perturbations, raw motor amplitudes/concentration
and actual unpruned prediction arithmetic. No action groups are fitted during
this input study. Do not interpret sharper policies or more active neurons
as improved learnability unless losses and subsequent controlled games agree.
A useful result still requires a newly matched conventional control before
an algorithmic-efficiency claim. Final test labels stay closed.

Persistent state, absolute-color perspective, output attachment, physiological
transmission changes and optimizer changes remain separate factors. Keep the
existing shared-memory headroom limits and pinned CPU production lanes.

## Engineering and fixed-weight probe

[Evidence and provenance](results/retinal-allocation-engineering-v1.json).
The generalized builder reproduces every reference sampler byte. The candidate
preserves all sensor, motor and context identities and all native ports.
Both eyes still observe all 81 board points with sampling rank 81:

| Map | Eye | Illuminated receptors | Sampling condition number |
|---|---|---:|---:|
| B, repeated current board | Left | 778 / 1,328 | 15.18 |
| B, repeated current board | Right | 1,299 / 2,162 | 13.77 |
| C, one larger image | Left | 769 / 1,328 | 27.26 |
| C, one larger image | Right | 1,282 / 2,162 | 27.38 |

The larger image illuminates slightly fewer receptors and has worse sampling
conditioning under the unchanged local kernel. It passes the registered rank
gate; no geometry search or gain adjustment follows this observation. This is
a larger angular image of the same **9×9 board**, not a 19×19 Go experiment.

At the same weights and 256 training-family positions, varying only the visual
input while holding context fixed gives:

| Weights | Varying motors, B → C | Standardized participation rank, B → C |
|---|---:|---:|
| Initialization, seed 1 | 242 → 251 | 1.56 → 2.07 |
| Trained B, seed 1 | 621 → 636 | 5.02 → 5.37 |

The variation threshold is standard deviation greater than 1e-6. The dominant
trained cell remains DNg30, body 10123, with about 99.96% of raw visual-response
variance in both maps. These are transfer measurements at weights trained with
B; they do not predict what training C will learn. Raw variance concentration
alone does not establish a policy or value limitation. The retained raw responses
include learned head coefficients for later contribution analysis; output
specialization remains a [separate experiment](readout-roles-study.md).

All 97 Python tests pass. Each intended seed passes full-CNS B32/K8 independent
JAX CPU states, losses, all gradients and three Adam updates at the unchanged
tolerances. Each also matches all 31 baseline initial checkpoint arrays, ports,
sampler and training contract, with exact B1/B32 direct/player states and
predictions, plus the next batch, loss, parameters and moments after fresh-process
recovery. The original immutable learner is retained; the previously qualified
visual-player package has identical numerical code. Failed operational gate
attempts and their diagnoses are retained in the evidence.

The [launch plan](../configs/retinal-allocation-v1.json) binds map, reference and
qualification identities. Three CPU runs used 24 pinned cores each on workers
1–3 and replicated checkpoints to another worker through root's pipes. Training,
full validation, motor probes and arithmetic counts are complete. TPU remains
paused.

## Completed paired results — 2026-09-15

[Results and provenance](results/retinal-allocation-v1.json). Each endpoint
completed exactly 1,000 updates / 32,000 exposures. The original learner,
initial core strengths, paired head/sampler seeds and dense learned heads were
preserved. The analysis reproduces the adopted B results exactly and aligns all
70,425 validation positions across 254 opening families. Final test labels
remain closed.

| Seed | B policy KL | C policy KL | B value MSE | C value MSE | B top-1 agreement | C top-1 agreement |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.65923 | 1.69224 | .84937 | .61774 | 20.44% | 16.69% |
| 2 | 1.68454 | 1.67101 | .59784 | .65141 | 18.99% | 20.03% |
| 3 | 1.68794 | 1.69834 | .64212 | .67146 | 18.24% | 17.94% |

Lower KL/MSE and higher teacher top-1 agreement are better. Pooled C minus B
is **+.00996 policy KL**, **−.04957 value MSE** and **−1.007 percentage points
top-1 agreement**. Conditional opening-family 95% intervals are respectively
[+.00727, +.01628], [−.05821, −.02840] and [−1.661, −.735] percentage points.
These intervals hold the three fitted pairs fixed; they do **not** quantify
training-seed uncertainty. Per-seed KL changes are +.03301, −.01353, +.01040;
value changes are −.23163, +.05357, +.02934. The pooled value improvement comes
from seed 1, while seeds 2 and 3 worsen. The report retains seed variation
separately. On 61,541 common reduced-source-novel positions, pooled changes
have the same signs: KL +.01474, MSE −.04925, top-1 −1.115 percentage points.

The 256-position visual perturbation probe after training C gives:

| Seed | Varying motors | Standardized participation rank | Leading motor | Share of raw visual variance |
|---|---:|---:|---|---:|
| 1 | 496 | 6.55 | DNg30, body 10123 | 99.991% |
| 2 | 305 | 5.53 | DNg30, body 10237 | 99.987% |
| 3 | 272 | 4.94 | DNpe018, body 69173 | 97.003% |

The larger map does not remove raw variance concentration. This statistic
alone does not establish that a learned policy or value decoder lacks useful
features; the [separate contribution diagnostic](readout-roles-study.md)
measures how the B heads actually use their motor signals.

Actual complete unpruned prediction arithmetic, including retinal encoding
and both heads, is **174.5–186.5M FLOPs per position at warm B1** and
**180.7–191.9M at warm B32**, across the three C endpoints. Optional dependency
pruning was disabled. Production occupancy changed between measurements, so
this comparison does not establish a CPU latency improvement. These models
cannot inherit the old roughly 8.4M-FLOP CNN comparison.

**Decision for the next output study:** retain the B current-board montage
as the common input to every new head arm. C has mixed results and no consistent
improvement across seeds and both losses. This choice follows validation and is
recorded as a development decision, not a preregistered preference. Use new
paired seeds 4/5/6 and a fresh dense control; keep candidate-discovery costs
visible. This does not change any completed run or register persistent training.

All final checkpoints have verified second-worker copies. The 78-file result,
source and diagnostic bundle also has verified copies on w0/w1; those copies
remain volatile. The public JSON omits only per-batch arithmetic traces and
identifies the retained full report by SHA-256. No new playing-strength or
equi-FLOP advantage is established.
