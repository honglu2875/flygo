# Spherical vision → individual motor embeddings

2026-09-14 implementation and pilot, following the user's proposal: **train the
representation before deciding its functional readout groups**. The untrained
synapse-count prior is a reference, not evidence of what a trained circuit can
represent. No anatomical candidate is declared dysfunctional from that prior.

## Narrow first experiment

Keep all 165,122 neurons and 15,270,273 directed edges. Learn strengths and
the existing type-shared bias/leak and sensory gains. Read the 1,314 descending,
708 nerve-cord motor and 107 head motor outputs separately: 2,129 numbers,
with unit, frozen readout gains. There is no random motor pooling, learned
projection layer or action grouping in this experiment.

The first visual adapter renders four historical boards as coherent luminance
images. The left eye receives lags 0/2 and the right 1/3, with the current
player's stones bright, opponent stones dark and empty points neutral gray.
All qualified receptors at one inferred column see the same local image.
The renderer stores a compact angular sampling operator and unit viewing
directions; it does not alter CNS edges. A stone affects neighboring visual
cells together. Repeated receptor classes are not independent image pixels.

**Geometry limitation:** the initial directions are an explicitly engineered
hexagonal chart mapped to a spherical cap. Both eyes use a common reference
lattice and angular scale. Missing reconstructions do not stretch either eye.
Its axis convention and orientation are recorded, not claimed to be measured
MaleCNS optical axes. Equal-sized patches are an initial control; frontal
acuity, optic-chiasm registration and recency-weighted allocation remain a
separate anatomical qualification. A registered direction table can replace
the chart without changing the sampler or network. The published eye-map
code illustrates why a real registration needs optical directions and
landmarks. [Primary eye-map code](https://github.com/reiserlab/eyemap_T4/blob/main/proc_eyemap.R).

The initial side-by-side placement failed coverage in the older patches.
An anatomy-only scan selected two patches centered at chart coordinates
(-25°, -30°) and (-25°, +30°), with 25° halfwidth, a 5° compact angular kernel
and four nearest samples per receptor. They lie in the common observed region
of both eyes. Each of the 324 board/history points is observed; the sampling
matrix has rank 324 and condition number about 30.9. This checks linear input
distinguishability, not preservation of information through the recurrent CNS.

This visual-only pilot omits the four non-occupancy feature channels. Pair
members have the same ply, player, komi and pass count. Missing early history
would render neutral; the shared-prefix requirement makes all four frames
available here. The scalar output is an **ordinal quality score**, not a
calibrated value estimate or complete Go policy. Additional context and longer
history require separate contracts and matched information in neural controls.

## Genuine recent-divergence pairs

Use only the immutable V0 **training** games, with the same raw KataGo teacher.
Two games must share an exact complete prefix of at least 20 plies. Compare
their pre-action states at the same later ply, 1–6 plies after their first
different move. Thus both labels refer to the same player; no alternating-turn
sign error or game-phase shortcut defines which branch is better. Every pass
counts as a ply. Require a raw teacher-value gap of at least .2 on its [-1,1]
scale (10 percentage points of the corresponding win probability).

Compare at most 256 uniformly sampled game pairs per exact 20-ply prefix,
before inspecting targets. Keep at most 32 eligible pairs per opening family;
sample families uniformly during training. This is a bounded sampler, not an
exhaustive corpus search. Retain identities, prefix/divergence lengths, teacher
values and hashes. Reject identical four-board visual observations and
mismatched global context. The first adjacent-history miner yielded too few
held-out pairs; its failed attempt remains. The wider candidate sampler keeps
the prefix and quality thresholds unchanged.

A fixed hash order reserves exactly 25% of eligible original training opening
families for representation probes. Freeze that family list with the bank;
do not reassign it while extending a run. The original 10% hash rule left only
two eligible probe families, so this allocation was revised before training,
without inspecting model performance. The source's D4 opening-family split is preserved.
Neither validation nor final test positions enter pretraining or grouping.
Remove probe pairs if either endpoint duplicates a training visual input
under D4; seven endpoints in the first bank triggered this additional filter.
These probes are selected for branch contrast and cannot estimate general
Go playing strength. Later data should include generated alternative branches
from independently sampled parents, with a new labeling contract.

## Objective

For each better/worse pair, make two views of **each same position** by
changing global image contrast around neutral gray. Stones, history and
orientation stay identical within a positive pair. One shared D4 augmentation
may be applied to the entire quartet. We do not force rotated raw motor
vectors to coincide, since that could discard action-location information.

Let `r(x)` be the individual motor rates and
`z = r / sqrt(sum(r²) + 10⁻⁶)`. The norm floor limits derivatives; no neuron
whitening or learned projection is applied. Each of the four views has one
positive (its other contrast view) and two negatives (the two views of the
other branch). With temperature .1:

```
Lcontrast = mean_a −log[exp(z_a·z_positive/.1)
                       / sum_{b in same quartet, b≠a} exp(z_a·z_b/.1)]
s(x) = wᵀ r(x) + b, with b frozen at zero
Lrank = mean_pairs softplus[−(mean(s_better) − mean(s_worse))/.25]
L = Lcontrast + Lrank
```

Separation alone gives no direction of quality; the ranking term supplies it.
The first bounded-score implementation and its numerical epsilon screen failed
update parity. Before any training, the ranking score was changed to linear:
pairwise ordering does not need a saturating value transform, and its common
intercept is unidentifiable. The ordinary Go value output retains `tanh`.
Unrelated games are not negatives. Positions of similar quality are not
automatically positives, which avoids deliberately collapsing every winning
position into one vector. Teacher uncertainty and quality thresholds need
sensitivity checks before interpreting the learned geometry. Continuous
label-aware contrastive methods are relevant alternatives, rather than proof
that this particular objective will work. [Rank-N-Contrast](https://arxiv.org/abs/2210.01189).

## Engineering and research gates

1. **Input and objective qualification.** Causal history/side tests, spherical
   neighborhood and missing-coverage tests, all 81 points observed in every
   patch, objective finite/autodiff checks, Rust/JAX gradients, stale-gradient
   rejection and checkpoint continuation. Keep production JAX output shapes
   unchanged unless embedding output is requested explicitly.
2. **Bounded CPU pilot.** K8, hard rates, seed 1, Adam .01, bias multiplier .01,
   clip 1; 64 updates of eight quartets (32 views, 2,048 total exposures). Compare a
   trainable circuit with the same circuit frozen while its scalar head trains.
   Readout gains and dummy policy parameters stay frozen. All settings and
   probe choices are fixed before launch. This small run tests whether the
   learning path works; it is not an optimization sweep or superiority claim.
   Default epsilon 1e-8 failed full-CNS update parity despite passing state and
   gradient checks: tiny near-cancelling gradients can receive large Adam
   updates. A numerical-only epsilon screen precedes training, keeping the
   established tolerances fixed. Record the chosen epsilon in both contracts;
   it is not automatically the best optimizer for this objective. Epsilon
   **1e-4** passed the actual B32/K8 three-update CPU gate and was used in
   both arms; 1e-8, 1e-6 and 1e-5 failed the linear-score numerical screen.
3. **Before/after probes.** Preserve individual rates, amplitude/variance,
   participation rank, raw and nuisance-conditioned correlations, and
   held-out branch-ranking accuracy, with half credit for exact ties and
   opening-family macro averages. Report anatomical subpopulations
   separately. An increased count of varying motors is not itself success;
   head-only improvement does not establish a better representation.
4. **Representation validation.** If the pilot is viable, register independent
   training seeds, a rate screen, a ranking-only control, fixed-probe tests
   and a frozen-embedding policy/value probe. Check performance on general
   positions and after fine-tuning. Then fit functional groups on discovery
   families and validate them on disjoint families and seeds.
5. **Fair Go comparison.** Count pretraining exposures, tuning and extra
   training work. Give the CNN the same information, labels and exposure
   budget, and match measured prediction FLOPs for the resulting engine.
   TPU optimization follows CPU qualification for this objective; prior TPU
   qualification does not automatically cover the new interface or loss.

The Rust interface accepts small external embedding/linear-score cotangents and
checks the parameter revision before applying them. It recomputes the forward
tape instead of exporting all CNS states. This deliberately simple prototype
uses **two forwards and one backward per update**; that extra work must be
reported. A retained-tape or fused objective is a later performance change.

## Completed CPU pilot

The final bank contains **540 training pairs from 39 families** and **94 probe
pairs from 12 other families**. Thirteen families were reserved; the visual
novelty filter removed every pair from one. Correlation measurements use the
93 unique, unaugmented visual positions among those probe endpoints. These
early-game branches span plies 21–65 and are a narrow probe distribution.
They are held out from embedding training, not the original validation set.

Both arms completed their declared 64 updates and 2,048 view exposures on
24 pinned physical cores each, concurrently on w2 and w3. The trainable arm
took 115.8 seconds; the frozen-circuit arm 120.6 seconds. The latter still
computes the full backward pass and masks core updates. These timings are
pipeline observations, not a matched playing-strength or inference benchmark.

| Probe measure | Initial circuit | Trained circuit + score | Frozen circuit + trained score |
|---|---:|---:|---:|
| Branch-ranking accuracy | 46.81% | 50.00% | 50.00% |
| Opening-family macro accuracy | 44.91% | 46.92% | 46.92% |
| Contrastive loss | 1.0986122882 | 1.0986122882 | 1.0986122882 |
| Ranking loss | .6931471743 | .6931471769 | .6931471785 |
| Raw response variation, RMS over cells | 9.9986 × 10⁻⁷ | 1.00016 × 10⁻⁶ | 9.9986 × 10⁻⁷ |
| Outputs with standard deviation > 10⁻⁶ | 73 | 73 | 73 |
| Covariance participation rank | 4.0117 | 4.0196 | 4.0117 |

The varying candidates are descending neurons; no head or nerve-cord motor
candidate crosses that fixed threshold on this bank. The frozen circuit's
raw responses remain bit-identical. Score gaps are only about 10⁻⁹; their
signs are not evidence of a reliable quality ordering. Counts cannot be
compared directly with the earlier 256-position overlay pilot because both
the stimulus distribution and input operator changed.

**No representation benefit was observed.** Only **1,759 of 15,270,273 edge
parameters (0.0115%)** changed at all in FP32; their maximum absolute change
was 1.43 × 10⁻⁴ and RMS change 8.19 × 10⁻⁸. Sensory gains, type biases/leaks
and score weights also moved, but output variation barely changed. Both
losses remain at their equal-prediction values, ln(3) and ln(2). Raw motor
RMS rate is about .0236, vastly larger than variation across these inputs.
This supports investigating baseline-dominated normalization and very small
learning signals. It does **not** establish that the biological motor cells
cannot become useful, or that this circuit has received substantial training.

Source **a64f480502a5b26d3fb7** passes **48 Rust and 84 Python tests**.
The full-CNS CPU gate checks states, individual readouts, loss, every gradient,
and all parameters/moments over three independently evolving Adam updates at
B32/K8. No tolerance was relaxed. Each pilot checkpoint also restores every
array and the sampler exactly and reproduces the next update in a fresh model
in the same process. The two additional diagnostic updates per arm account
for 64 extra view exposures and do not alter the published step-64 checkpoint.
This is not a fresh-process restoration or TPU qualification claim.

All eleven files per pilot, including checkpoints, responses and correlations,
have verified copies on w0 and their original worker (w2 learned, w3 frozen),
about 269 MB per run. They remain in volatile shared memory. These checkpoints
contain a dummy one-action head and an ordinal score; they cannot be used as
a Go policy/value engine without a separately trained task head.

Receipts: [pilot and parameter changes](results/embedding-pilot-v1.json),
[full-CNS CPU parity](results/embedding-cpu-parity-v1.json),
[tests and retained engineering revisions](results/embedding-engineering-v1.json).

## Next gate: useful learning signals

Do not fit functional groups from this pilot or extend its flat loss blindly.
First compare neutral-light and training-position responses and gradients by
anatomical group. Then qualify a fixed, **training-only** centering/scaling
transform for the objective, with bounded amplification and raw rates retained
for interpretation. A physiological neutral-adaptation intervention is a
separate factor, not silently part of the same comparison. Preserve a frozen
circuit control and make any loss-scale/optimizer change explicit.

The qualification should establish finite, reproducible and appreciable core
updates before a longer pretraining run. Do not center each better/worse pair
around its own mean: this can turn pairwise contrast into an antipodal encoding
without learning a useful representation. Once the learning path is viable,
use multiple training seeds, general-position probes and downstream policy/value
evaluation before deciding motor groups. The original validation and final test
sets remain outside the representation-discovery fit.

## Small interfaces

```text
python/flygo/vision.py          spherical sampling and individual motor ports
python/flygo/data/branches.py   causal pair selection and family sampling
python/flygo/contrastive.py     loss, analytic cotangents, JAX loss reference
scripts/prepare_embeddings.py  immutable geometry and replay-pair bank
scripts/qualify_embedding.py   full-CNS objective and Adam parity gate
scripts/train_embeddings.py    bounded learned/frozen CPU pilot and probes
tests/test_embedding.py        meaningful interface and derivative checks
```

With the qualified package installed, the command sequence is:

```bash
python scripts/prepare_embeddings.py --output /dev/shm/flygo/runs/spherical-pairs-new
python scripts/qualify_embedding.py \
  --bank /dev/shm/flygo/runs/spherical-pairs-new \
  --batch-size 32 --epsilon 1e-4 \
  --output /dev/shm/flygo/runs/embedding-qualification-new
python scripts/train_embeddings.py \
  --bank /dev/shm/flygo/runs/spherical-pairs-new \
  --qualification /dev/shm/flygo/runs/embedding-qualification-new/result.json \
  --variant learned --output /dev/shm/flygo/runs/embedding-learned-new
```

Run the frozen arm with `--variant frozen` and its own output directory, on
another admitted research lane or host. Output directories must be new.
Training refuses a different bank, native library, model/objective, relevant
Python sources or insufficient B32 qualification. Existing immutable runs use
`spherical-pairs-v5`, `embedding-qualification-b32-v1` and
`embedding-{learned,frozen}-v1`. The commands reserve shared memory and pin
their declared CPU lanes; remote launch also requires the verified runtime
and bank bundle. Changing an experiment requires a new contract and receipt.

The existing graph, Rust recurrence, optimizer, Go replay engine and checkpoint
arrays are reused. Biological group fitting remains downstream of trained
responses. Evidence that task optimization can make connectome-constrained
visual models informative motivates this order; it does not establish a Go
advantage. [Connectome-constrained visual models](https://www.nature.com/articles/s41586-024-07939-3).
