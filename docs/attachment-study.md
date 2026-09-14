# Spherical input ablation: supervised screen

Registered 2026-09-14 before scientific training. This starts stages A/B of the
[temporal vision design](temporal-vision-study.md). One-board-per-eye geometry,
persistent state, output regrouping and optimizer interventions remain separate.

**Question:** does a coherent historical or current-board image improve Go
policy/value learning beyond the shared nonvisual context? The earlier
contrastive pilot neither trained a Go policy nor supplied this context.
Its results are not a matched baseline for this experiment.

| Arm | Qualified photoreceptor inputs | Other sensory inputs |
|---|---|---|
| history | Existing spherical L: lags 0/2, R: lags 1/3 | Shared current context |
| current | Lag 0 in every existing patch | Identical current context |
| neutral | Constant .5 at every visual input | Identical current context |

All arms use the same 3,490 photoreceptors, chart, patch coordinates, compact
sampling weights, neutral level and initial input gains. No retinal area is
reallocated here. Current-versus-history is intentionally a history ablation.
Neutral is the context-only control, and is trained for the same final horizon.

The 84 context features are point legality in board order, black to play,
current-player signed komi/81, and consecutive passes/2. Select from the existing
sensory inventory, excluding all optic sensory cells and R-prefixed types, and
require a retained outgoing edge. Use one balanced seeded assignment (918421),
independent of head/learner initialization. This is an explicit engineering
map, with no claim that these neurons biologically encode Go context. There is
no zero-in-degree selection, extra history, capture feature or direct head bypass.
The immutable attachment receipt records all node IDs and feature multiplicities.

## Fixed learning contract

- Fixed MaleCNS graph `183e28b8d990ed5d3ffca091f0468d25dcca3dd1ccd98183e14615a09cda35b4`;
  no neuron/edge changes or transmitter-sign interventions.
- Existing hard leaky-rate equations, K=8, fixed initial state reset for every
  position. Initial input gain .2 and the existing signed, incoming-normalized
  strength prior. All nine parameter groups learn, with bias update multiplier .01.
- Each of the same 2,129 motor/descending candidates is an individual readout
  feature. Ordinary learned policy and tanh-value heads; ordinary supervised
  teacher policy cross-entropy plus value MSE. No contrastive objective.
- V0 dataset `c8862e9e998b05231b053220fcecf797630ac1c1c3be5a5d676a0c028bedf8b3`,
  original opening-family split. Uniform training-position draws with replacement
  and common whole-board D4 augmentation. Learner/sampler seed 1 in every arm.
- **1,000 optimizer updates × batch 32 = 32,000 labeled training exposures**
  per arm. Fixed final horizon. Adam peak rate .03, linear warmup 100 updates,
  then constant rate; global clip 1, bias multiplier .01 and all other rate
  multipliers 1. The initial epsilon 1e-8 fails the numerical gate described
  below. The peak rate and bias multiplier come from the supervised optimizer
  studies; no rate/epsilon selection uses this screen's validation.
- Evaluate at 0/250/500/750/1,000 updates on the existing fixed-seed 2,048-position
  train, validation and D4-novel validation slices. At 0 and 1,000, additionally
  evaluate each checkpoint under both other visual modes. These perturbations
  are diagnostics and do not replace its normal validation result.
  The shared novelty slice is defined on the original four-board feature cache;
  it does not prove novelty after current-only or neutral input reduction.
  Report this limitation and add input-specific novelty checks before interpreting
  generalization under the reduced encodings.
- Record group gradients/parameter changes, clipping and activity every 250
  updates on the actual training batch. Account for these additional forward
  and backward evaluations separately. Keep initial/final checkpoints and
  replicate them with verified hashes; no old study artifacts are overwritten.

## Qualification and interpretation

**Pre-training numerical revision:** runtime `861175bf88bf1ba6b4f5` passes all
87 Python tests. The first B32/K8 history gate passes states, losses and all
gradients, then fails one policy-weight parameter at its first Adam update
(difference 4.10e-5; unchanged rtol .003 / atol 5e-6). The failed attempt remains
in `runs/attachment-qualification-v1`. A read-only training-batch diagnosis in
`runs/attachment-precision-probe-v1` finds the relevant gradient is about 8e-10.
Independently computing both heads in FP64 still leaves a 1.64e-5 update
difference because their FP32 recurrent responses differ. Head summation alone
does not resolve this conditioning issue.

For a first unclipped Adam update, the sensitivity of `rate*g/(abs(g)+epsilon)`
to g is `rate*epsilon/(abs(g)+epsilon)^2`. An epsilon of 1e-8 can amplify tiny
absolute gradient errors by nearly three million at rate .03. Before scientific
training, test epsilons **1e-6, 1e-5, 1e-4** in that order and use the first to
pass all three modes and all three free-running updates. This bounded ladder
uses numerical agreement on training batches only. Keep that one epsilon fixed
across the scientific arms, report its effect on small gradients, and retain
every failed gate. No tolerance, rate, topology, readout or dataset changes.
This numerical choice is not evidence of an optimizer improvement; optimization
and the capacity of weak visual paths still need their own later study.

**Gate completed:** epsilon **1e-6** passes all three modes, including all states,
losses, gradients and three free-running parameter/moment updates. The higher
ladder values were not tested. All three scientific arms launched with that
fixed epsilon on workers 1–3, CPU cores 32–55. Their actual initial checkpoints
have identical parameters, moments, ports and sampler states; direct versus
Go-interface predictions match exactly at B1 and B32. Initial checkpoints have
verified second copies on w0. [Engineering evidence](results/attachment-engineering-v1.json).

All three arms completed their fixed **32,000 exposures**. On the registered
2,048-position validation slice:

| Trained input | Policy KL ↓ | Value MSE ↓ | Top-1 agreement |
|---|---:|---:|---:|
| History | 1.68728 | .65678 | 17.82% |
| Current | 1.67900 | .86844 | 18.60% |
| Neutral | 1.67716 | .83000 | 19.24% |

Blanking the trained history/current model's eyes increases KL by .00950/.00916
and value MSE by .18831/.22290. Replacing history with current imagery in the
history-trained model changes KL by -.00028. The models use visual input, but
the separately trained neutral control remains competitive. Blanking is also
a distribution shift; these perturbations alone do not establish generalizable
visual benefit.

Full validation completes under [the frozen endpoint plan](../configs/attachment-validation-v1.json):
**70,425 positions from 254 opening families**, using the fixed final checkpoints.

| Trained input | Full-validation KL ↓ | Full-validation MSE ↓ | Top-1 agreement |
|---|---:|---:|---:|
| History | 1.66743 | .62807 | 19.13% |
| Current | 1.65923 | .84937 | 20.44% |
| Neutral | 1.65202 | .81902 | 21.33% |

History-minus-neutral value MSE is **-.19095**, with a paired opening-family
bootstrap interval **[-.29620, -.14491]**. Their policy KL difference is +.01541,
interval [-.00017, +.02334]. Current-minus-neutral KL is +.00720, interval
[-.00212, +.01130]; its value difference is unresolved. Thus this seed gives a
value benefit for the history-trained model, with no policy benefit established
for either visual arm. These intervals condition on the three trained weights;
they do not measure variation across training seeds. On the common 61,541-position
reduced-source-novel subset, the history value difference is -.22683, interval
[-.34158, -.17294]. [Complete analysis and provenance](results/attachment-screen-v1.json).

About **98.4–98.9%** of edge parameters change, unlike the earlier contrastive
pilot. The learned sensory multipliers are signed and unconstrained; some
invert their input contrast. This is an external learned encoding, not a fully
physiological photoreceptor response model. Each arm additionally evaluates
38,912 training-time validation/perturbation positions, 256 diagnostic forwards,
128 diagnostic backwards and 70,425 full-validation positions. These are
reported separately from the 32,000 labeled optimizer exposures.

A separate input-only audit finds **62,984/70,425** validation sources unseen
under D4 for current-board-plus-context, and **61,541/70,425** for neutral
vision plus context. Current keys describe pre-render source features; absence
of FP32 retinal collisions is not certified. Neutral keys describe its actual
constant-vision input. The audit examines neither labels nor final-test inputs.

On 256 positions from distinct training families, the history/current models
have 415/621 motors with visual-minus-neutral standard deviation above 1e-6.
Almost all raw visual-response variance, **99.977%/99.962%**, is concentrated in
one annotated descending neuron, **DNg30, body ID 10123**. Unit-variance scaling
of the varying cells yields effective covariance ranks **7.13/5.02**, so the
other signals are not literally one identical feature. This is model-dependent
response concentration, not a claim about that neuron's biological function.
Raw responses and absolute amplitudes are retained; no action groups are fitted.

Counted warm B1 arithmetic on 64 common validation positions is approximately
**176.1M / 181.6M / 173.3M FLOPs** for history/current/neutral. B32 counts are
182.1M / 187.9M / 179.7M per position because row skipping is batch-wide.
The unchanged no-skipping convention gives about 252.7M. These models cannot
inherit the older K4 model's 8.3–8.9M warm comparison with the small CNN.
Counts include the external renderer and attached context; mask/integer work,
memory traffic, training and teacher work remain separate. Trace timings are
not production inference latency.

Pure Python attachments precede the existing Rust/JAX equations. Unit checks
verify coherent bilateral current input, independence from older planes, context
separation and disjoint sensory/readout identities. Freeze code and map hashes,
then qualify the full graph at actual B32/K8 for every arm: states, losses,
every gradient and three free-running Adam parameter/moment updates. Keep the
existing tolerances. A numerical failure is retained and prevents that arm's
scientific launch; it is not a poor learning result or permission to change the
optimizer inside this contract. Actual TPU qualification remains a later gate.

Training verifies the qualified runtime, map/mode, model dimensions, optimizer
and batch. Checkpoints record the external input encoding; resume rejects a
change of visual mode even though the underlying ports have the same shape.

Use paired learning curves and final validation metrics. If all three arms
perform similarly, investigate whether visual responses reach the readout and
receive effective updates before scaling training. If either image arm improves
over neutral, verify visual sensitivity and repeat with additional seeds under
a separately frozen confirmation horizon. Neither outcome alone establishes a
biological or equi-FLOP advantage. Recount prediction work and provide an
information-matched conventional control before making that claim. Final test
labels remain closed.

CPU execution uses 24 physical cores in a research lane per host, disjoint from
the 64 production cores. Storage admission preserves the 64 GiB free-SHM floor,
96 GiB available-RAM floor and 100 GiB own-file cap, including reservations.
The user has reassigned TPU availability to other work. Pause and coordinate
with the user before any TPU use; the current study uses no TPU devices.

## Next decisions

First confirm the history arm's value finding with paired learner/sampler
seeds 2 and 3, keeping all three inputs and the same 32,000-exposure horizon.
Register that confirmation before launch and place verified checkpoint copies
within the existing host budgets. It must not become a longer or retuned run.

Then keep the existing order: qualify one larger current board per eye as a
separate input allocation test, followed by output attachment. The concentrated visual
response motivates that output study; it does not justify freezing action groups
from this one short seed. Divisive gain control, tonic activity, transmitter
exceptions and optimizer conditioning remain separate rule/optimization factors.
State carried across plies remains unimplemented.

An engineering-only dependency audit finds that a reset K8 motor prediction
needs 83.53M of the 106.89M post-cache edge evaluations before zero skipping.
Most saving is in the final two passes. This suggests an exact inference
optimization to qualify later, with the entire logical graph retained; it is
not a measured speedup and does not apply to returning a complete carried state.
