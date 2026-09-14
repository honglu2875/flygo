# Biological interfaces and response-guided readouts

Revised 2026-09-14 after the user's proposal for spatial bilateral history and
correlation-based action groups. This becomes the first application of
[G1–G3](group-study.md), before another broad architecture sweep. The fixed
MaleCNS neurons and edges remain the substrate; strengths remain learnable.
The engineering gates and completed study records are preserved.

**Question:** does respecting sensory organization, signal propagation and
functional readout structure improve learning at the same prediction FLOPs
and labeled-position horizon? Anatomical plausibility, sharper policies and
stable correlations are intermediate measurements, not the answer.

## What the current interfaces actually do

The ordinary initializer distributes 972 features over 15,912 sensory cells
and distributes 149,210 other cells over 656 output pools. Pooling uses
inverse-square-root counts and learned neuron gains. The spatial overlay
reassigns board positions for 3,490 photoreceptors, **retaining their original
random feature channels** and every other sensory assignment. It is not a
coherent retinal image or a montage of historical boards. The earlier small
spatial benefit therefore leaves the new hypothesis open.

Changing `FlyConfig.seed` changes ports and heads. Initial recurrent strengths,
leaks and biases are otherwise identical. A multi-initialization response
study must explicitly vary physiological parameters or strengths, keeping
input coordinates fixed. Head-only changes cannot reveal new internal tuning.

The current feature contract contains four boards: own/opponent occupancy
from the current player's perspective, followed by black-to-play, signed
komi/area, consecutive passes/2 and point legality. The Go history engine
already supports depths 1–64; the corpus cache, learner and GTP profile are
currently fixed to depth 4. Stored complete action sequences permit a new,
causal history encoding without regenerating expert games.

## Biological evidence and its limits

Eye geometry is not a spherical fit to neuron somata. Optical axes describe
viewing directions; optic-lobe column coordinates describe another space.
Recent microCT work registers these spaces using anatomical landmarks. It
finds highest angular acuity near the frontal equator, nonuniform sampling
and less than 20° binocular overlap; the optic chiasm also changes orientation.
Its female eye measurements are a potential template, not measured optical
axes for our MaleCNS individual. The accompanying data/code are public.
[Eye structure study](https://www.nature.com/articles/s41586-025-09276-5),
[eye-map resources](https://github.com/reiserlab/eyemap_T4).

Binocular temporal mechanisms have been studied experimentally in flies.
Alternating historical Go positions between eyes remains our engineered
encoding, rather than an established biological interpretation of those
mechanisms. [Binocular microsaccadic sampling](https://pubmed.ncbi.nlm.nih.gov/35298337/).

The male visual-system reconstruction documents incomplete lamina coverage
and difficult photoreceptor segmentation. R1–R6 are undercounted, and some
R7/R8 identities are uncertain. Missing cells and the left/right imbalance
must not be interpreted as true retinal density. R7/R8 subtype assignments
use anatomical/connectivity evidence rather than direct rhodopsin measurement.
[Visual-system inventory](https://www.nature.com/articles/s41586-025-08746-0).

Photoreceptor classes should not receive unrelated random channels at the
same visual location. R7/R8 pathways have distinct color and polarization
specializations; an arbitrary Go plane is not a measured receptor stimulus.
Start with a declared luminance rendering, then consider receptor-specific
spectral encodings as a separate hypothesis.
[Photoreceptor target circuits](https://pmc.ncbi.nlm.nih.gov/articles/PMC8789284/).

Descending neurons relay brain signals toward the nerve cord; motor-neuron
annotations identify outputs more meaningfully than soma proximity to a leg
or wing. Keep descending, head-motor and nerve-cord motor populations separate
before combining them. Comparative datasets inform the interpretation, but
their body IDs cannot be substituted for MaleCNS IDs.
[Descending/ascending circuit study](https://www.nature.com/articles/s41586-025-08925-z).

A close methodological precedent is flyvis: spatially organized input,
connectome-constrained recurrent dynamics, type-shared physiological
parameters and task optimization yielded predictions checked against neural
recordings. Its synaptic parameter sharing, temporal input and activity
regularization offer concrete hypotheses. This evidence concerns visual
computation, not Go or a generic connectome advantage.
[Lappalainen et al.](https://www.nature.com/articles/s41586-024-07939-3).

## Measured interface atlas

The [CPU pilot](results/bio-ports-pilot-v1.json) preserves canonical body IDs,
source hashes and candidate membership. Runtime arrays are in
`runs/bio-ports-pilot-v1/` under the shared-memory root.

| Retained sampling | Left | Right |
|---|---:|---:|
| Qualified photoreceptors | 1,328 | 2,162 |
| Distinct inferred visual columns | 652 | 730 |
| Mean columns per region with 12 disjoint regions/eye | 54.3 | 60.8 |
| Spatial samples needed for twelve separate 9×9 boards | 972 | 972 |

Qualification here means a unique same-side downstream-column majority,
confidence at least .5 and eligible synapse mass at least 4. These columns
are inferred, not optical measurements. Photoreceptor counts are not counts
of independent spatial pixels. Twelve equally sized disjoint regions cannot
give each board point its own column in this retained mapping. Overlapping
sampling can encode more inputs, but its conditioning and distinguishability
must be measured rather than assumed.

The output candidates are 1,314 descending neurons, 708 nerve-cord motor
neurons and 107 head/central-brain motor neurons: **2,129 cells**. The atlas
retains subclass, type, side, exit nerve and soma neuromere. `receptorType`
in this annotation table contains putative sensory receptor labels; it is
not a postsynaptic neurotransmitter-receptor map. Transmitter predictions
are stored separately, and the model's sign conversion remains an approximate
physiological prior.

## Input design

Represent a qualified viewing direction by `u_i ∈ S²`. For board point p in
history patch j, let `v_jp ∈ S²` be its rendered direction. A bounded local
sampling operator A produces sensory drive:

```
I_i(t) = g_i ∑_{j,p,c} A[i,j,p,c] X[t − lag(eye_i,j), p, c] + context_i(t)
A[i,j,p,c] ∝ receptor_response(i,c) × kernel(acos(u_i · v_jp))
```

Use a compact neighborhood and a declared normalization, preserving local
contrast and recording total drive per eye/type. Each stone influences nearby
receptors coherently. Empty board, own stone and opponent stone can first be
rendered as three luminance levels about a neutral background. Color identity
and contrast choices are fixed engineering decisions, with matched controls.
Do not equate a photoreceptor with an ON/OFF detector downstream.

First register the retained hexagonal column lattice and its orientation.
Use a flat local chart honestly if spherical registration is not yet
qualified. Check visual-field flips, mirrored left/right views, neighborhood
distortion, missing columns, boundaries and input distinguishability. Never
infer acuity from missing reconstructions or affine-normalize missing regions
into a fictitious complete eye. Local pooling of photoreceptors must also
respect neural superposition rather than assuming co-facet cells share a
viewing axis.

The user's proposed montage is parameterized by P patches per eye:

```
left  patch j: board t − 2j
right patch j: board t − (2j + 1),       j = 0, …, P−1
```

Thus P=12 means 24 board states; twelve boards total would mean P=6. Keep
this explicit in each contract. The first comparison uses the existing four
boards (P=2), isolating organization from adding information. Increase history
only in a separate experiment with the same history available to the CNN.

Place newer patches in the better-resolved frontal-equatorial region **after
registration**, with larger patches if needed. Compare equal allocation and
reversed recency while preserving the sampling budget. Both eyes use the
current player's color perspective; every pass counts as a ply. Early-game
missing history needs an explicit validity convention. Apply the same Go D4
transform to every historical board before rendering, and transform targets
consistently. Current legality/pass/komi have a separate documented context
interface; no future state or label is broadcast.

Also test **temporal presentation**: place each history board at the same
retinal coordinates and present the sequence oldest to newest across internal
passes, with declared dwell time and neutral adaptation. This preserves spatial
resolution and gives temporal circuits an appropriate input. Compare it with
the montage at the same total recurrent work. A static montage does not itself
recreate temporal motion detection. Do not assign seconds to model passes
without a physiological calibration.

## Correlation protocol and readout design

For each declared input/weight configuration, record outgoing firing rates
`R[s,i]` over probes s, alongside voltages and evoked amplitudes where needed.
Compute signed Pearson correlations after centering across probes:

```
C[i,j] = ⟨R[:,i] − mean_i, R[:,j] − mean_j⟩ / (norm_i × norm_j)
```

Silent or numerically negligible cells remain missing. Record amplitudes,
participation rank, dominant common components and the number of independent
probes. Large correlation on tiny responses is not a useful information path.
Show both raw correlations and correlations after regressing declared shared
inputs such as stone density, turn and game phase. Removing those effects
can also remove useful signal; it is a diagnostic, not an automatic layer.

The next probe bank combines legal training positions with controlled local
contrasts, single-stone perturbations, flashes and moving boundaries. Fit
groups only on a discovery subset of training opening families, check them on
disjoint training families and stimulus families, and assess seed stability.
Separate input-map seed, strength/physiology seed, probe seed and optimizer
seed. Include untrained priors and independently trained models; random-model
correlations alone do not establish biological significance.

Use an anatomical candidate population first, then a sparse graph of reliable
positive correlations. Report opponent/negative channels separately instead
of grouping by absolute correlation. Compare functional groups against
type/side/subclass groups and population-, activity- and size-matched shuffled
groups. Use shrinkage or cross-validated low-rank responses when probes are
scarce. A correlation graph is an analysis artifact, not added neural edges.

Consensus co-membership across seeds and probe splits can yield a versioned
table. Store amplitudes, confidence, response protocol, neuron IDs, uncertainty
and source/checkpoint hashes with it. Revalidate after an interface or dynamics
change. Do not call it a permanent biological atlas from one simulation family.

Initially use groups as features:

```
z_g = ∑_{i ∈ group_g} a_i rate(h_i) / sqrt(|group_g|)
policy_logits = B z + b
value = tanh(vᵀ z_value + b_value)
```

An 82-group candidate is reasonable, with a learned small map B to 81 board
points plus pass. Anatomy does not assign cluster 0 to A1. Compare with 82
random groups from the **same cells and pool sizes**; separately test group
count and candidate population. Keep the value readout independently
specified. Pooling correlated cells reduces redundancy; it does not guarantee
that every action has useful distinguishing features. Local receptive fields
and action sensitivity are additional measurements.

Full-neuron FP32 correlation would require about 109 GB just for the matrix.
The present candidate matrix is about 18 MB. Use bounded candidate matrices,
response sketches or nearest-neighbor graphs for larger populations.

## First response pilot: completed, descriptive

`scripts/probe_biological_ports.py` used 256 positions from distinct V0 training
opening families, 24 pinned CPU cores, the original strengths and two explicit
log-magnitude perturbations (standard deviation .1). Perturbations preserve
each destination's incoming absolute mass and every sign/edge. Input/head seed
was fixed. The completed seed-1 1,048,576-exposure checkpoint was also probed.
No optimizer update or grouping fit occurred. Six cases completed in 42.9 s.
All 18 probe/analysis artifacts and source files have a verified second copy
on w1 (about 256 MB total). Five targeted diagnostic unit tests pass, including
silent cells, opponent responses, common-input removal and fixed incoming mass.

Reproduce with the retained release/checkpoint and a new immutable output name:

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 \
  /dev/shm/flygo/venv/bin/python -B scripts/probe_biological_ports.py \
  --output /dev/shm/flygo/runs/bio-ports-pilot-reproduction-v1 \
  --checkpoint /dev/shm/flygo/runs/tpu-confirm-v0-fly-s1-learner/checkpoints/step-00000512.npz
```

| Initialization/interface | Passes | Varying descending cells / 1,314 | Varying nerve-cord motors / 708 | Varying head motors / 107 |
|---|---:|---:|---:|---:|
| Original all-sensory input | 4 | 1,314 | 708 | 107 |
| Existing overlay, visual-only drive | 4 | 108 | 0 | 1 |
| Existing overlay, visual-only drive | 8 | 334 | 8 | 11 |
| Trained original interface | 4 | 121 | 21 | 1 |

“Varying” means sample standard deviation above 1e-6 model rate units. The
visual-only overlay still uses random feature channels and drives all 4,107
visual cells, including 617 cells without a qualified reassignment. It is a
diagnostic removal of nonvisual drive, not the proposed biological renderer.
The input populations differ, so this table is not a fair learning comparison.

At initialization, RMS candidate variation drops from .00222 with all sensory
ports to .00000207 with overlay/visual-only drive at K4. Short structural
paths therefore do not ensure substantial signal. The independent route
audit already permits visual influence at K4 on 1,299 descending, 543
nerve-cord motor and 95 head-motor cells; the rate responses are much weaker.

One modeling choice to investigate is the current initialization
`w_e = sign(pre) × log(1+n_e) / sum_incoming log(1+n)` with update fraction .5
and bias .02. This is an engineering normalization, not measured physiology.
The pilot does not isolate it as the cause. B2 should measure input-response
Jacobians and attenuation by type/path before choosing between longer settling,
type-pair synaptic scales and activity stabilization, each with a gain-matched
control. Preserving topology alone does not preserve biological dynamics.

The two small strength perturbations give K8 correlation-table agreement of
.9982–.9985 raw and .9949–.9953 after conditioning, on common varying cells.
This is local robustness to small perturbations with the same input map, not
evidence for a universal or physiological partition. Split-half results and
all missingness/thresholds are retained in the JSON.

The trained K4 policy has mean maximum probability .140 and mean entropy
.883 times legal-uniform entropy on these probes; the initial values are
.049 and .997. Its 656-dimensional pooled representation has covariance
participation rank 5.64 on this sample. This rank is a variance summary, not
a mutual-information estimate or proof of information loss. Extending that
K4-trained model to K8 sharpens the policy while concentrating pooled variance
further. A [post-hoc check of the same training probes](results/bio-ports-sharpness-v1.json)
shows why sharpness is insufficient:

| Same checkpoint, no retraining | Mean top-move probability | Teacher policy KL | Value MSE |
|---|---:|---:|---:|
| K4, as trained | .140 | 1.278 | .575 |
| K8, extrapolated | .693 | 13.920 | 1.247 |

Targets were consulted only after recording the correlation responses; they
did not enter grouping or probe selection. This is not a trained-K4 versus
trained-K8 architecture comparison or a validation estimate. The old 2,000-step
HTML example is not a diagnosis of the current checkpoint.

## Ordered milestones and acceptance criteria

| Step | Deliverable | Gate |
|---|---|---|
| B0: interface/response audit | Canonical candidate atlas, raw/conditioned response matrices, missingness and sharpness measurements | **Pilot complete**; provisional correlations do not become production groups |
| B1: coherent retinal input | Versioned local sampling map, orientation/coverage report, four-board bilateral and matched shuffled adapters | Check local distinguishability, perspective, history, drive scale and missingness; qualify Rust/JAX forward/gradient/update parity for any new operator |
| B2: propagation and response atlas | Biological stimulus probes, per-group amplitude/delay/adaptation/gradient measurements; initial and trained ensembles | Distinguish attenuated, silent and redundant paths; assess held-out probe and seed stability |
| B3: readout partition | Consensus candidates, anatomical and matched random controls, explicit pass/value interfaces | Group tables fit only on training discovery probes; verify on disjoint training families; qualify both backends |
| B4: learnability screen | Input-only, readout-only, then combined comparison; one added factor at a time | Same contents, sample stream, exposures and tuning opportunity; compare learning curves, validation KL/value and paired games |
| B5: confirmation | Three-seed comparison against the matched CNN and interface controls | Recount all prediction work, including rendering and passes; equal information and labeled horizon; report all seeds and tuning cost |

B1–B3 may reveal that a modest physiology change is necessary for useful
motor responses. Then isolate one intervention: type-shared synaptic scales,
activity stabilization, initial neutral adaptation, or a physiological time
constant prior. The baseline already shares leak/bias by type. Do not silently
combine new input, extra passes, homeostasis and readout groups and attribute
the result to retinotopy. G5 specifies the rule and its matched control.
Any auxiliary visual pretraining must have its examples, optimization and
compute reported, with an equivalent opportunity for the CNN in that arm.

Use compact histories or on-demand replay for longer windows, not a silent
expansion of the existing full-corpus FP32 cache. Keep all storage reservations
and shared-memory floors. CPU owns atlas construction and correlation work;
use the four-host JAX path for sufficiently large qualified training batches.
New history/sequence operators require CPU parity and actual TPU qualification.

Keep the implementation small: the atlas/probe command and immutable artifacts
are outside the learner; adapters belong in the existing port boundary; neuron
rules remain in the Rust/JAX recurrent backends. Add a reusable module only
when a second real consumer needs it. Record each gate and any hypothesis
change in `PROGRESS.md`, without rewriting earlier negative results.
