# Controlled results

The current fly baseline is substantially weaker than the residual CNN at the
registered nominal prediction cost and training horizon. This is a result for
these adapters, dynamics and training settings. It does not establish the best
performance attainable with the fixed fly topology.

These are FLOP/exposure comparisons. Checkpoints contain 15,514,402 stored
learned scalars for the fly, 786,819 for the large CNN and 51,955 for the
small CNN. Optimizer moments and fixed graph/port arrays are excluded; the
count includes stored parameters on routes with zero gradients.
[Exact checkpoint counts](results/control-parameter-counts-v1.json).

The later [spherical-input confirmation](attachment-confirmation.md) evaluates
history/current/neutral imagery at a separate 32k-exposure, K8 contract.
Both visual arms improve value over neutral in the two new seeds, while policy
loss worsens. Motor-response concentration persists with seed-dependent cell
identity. These runs have different features and arithmetic and do not inherit
the CNN comparison below. [Paired evidence](results/attachment-confirmation-v1.json).

The complete [readout screen](readout-roles-study.md) retains current spherical
inputs and compares four external heads across three paired 32k-exposure seeds.
Soma-side minus dense mean policy KL is **−.01401**, value MSE **−.06375**, with
both improving in every seed at essentially the same mean counted prediction
FLOPs. Teacher top-move agreement falls .200 percentage points. Soma-side policy
KL also improves over the frozen shuffled split in every seed; value gains are
shared by that generic split. The five-cell value restriction has a mixed
tradeoff: KL −.03468 / MSE +.03730. Dense remains the reference pending registered
fresh-seed confirmation. This short screen has a different prediction budget
and does not inherit the CNN comparison below.

The [converged motor-decoder diagnostic](motor-convergence-study.md) completes
27 fits on three frozen 32k-exposure cores. At ridge .01, a gated linear residual
improves subset validation KL **1.68393 → 1.54872**, versus **1.65879** for its
bias-only control. Weak regularization fits mean training KL **.68778** but
worsens validation to **2.07265**. All numerical optimization-error bounds pass;
all solver outcomes and 97.42M additional fit exposures are reported. This
separates optimization and overfitting on a fixed bank; it is not a new
full-validation or matched-CNN result.

## Matched neural control, three seeds

Both families receive 1,048,576 labeled-position exposures: 512 Adam updates at
batch 2,048, from initialization, on the same frozen V0 release and deterministic
augmented sampling stream per seed. Each family first receives the same
three-rate screen; its minimum validation KL+MSE selects CNN rate 0.003 and fly
rate 0.01. Earlier fly-only tuning is separate project compute. All final
checkpoints are evaluated; there is no best-step or best-seed selection.

The fly uses all 165,122 neurons and 15,270,273 edges, four recurrent passes,
656 random readout pools and random sensory ports. The CNN uses 64 channels,
ten residual blocks and a 3×3 policy-hidden layer. Both have the same Go input,
legal policy loss and teacher value target. Their nominal arithmetic is
126,697,550 and 126,778,131 FLOPs per prediction, respectively: a 0.064% gap.
Parameter counts, memory traffic and accelerator utilization differ. The CPU's
data-dependent zero skipping and constant-state cache require a separate
executed-work ledger; this study does not match actual CPU instructions.

Full validation contains 70,425 positions from 800 complete games. Lower KL
and MSE are better; higher agreement is better.

| Model / seed | Policy KL | Value MSE | Teacher top-1 agreement |
|---|---:|---:|---:|
| Fly 1 | 1.504230 | 0.556951 | 22.67% |
| Fly 2 | 1.503745 | 0.547262 | 22.40% |
| Fly 3 | 1.505421 | 0.563244 | 22.59% |
| CNN 1 | 0.740054 | 0.379831 | 48.93% |
| CNN 2 | 0.750133 | 0.393475 | 49.62% |
| CNN 3 | 0.755611 | 0.432116 | 49.41% |

The mean paired CNN-minus-fly difference is −0.755866 KL and −0.154012 MSE.
Game-bootstrap 95% intervals, conditional on these trained checkpoints, are
[−0.772081, −0.739411] and [−0.183534, −0.126155]. Paired-seed difference
sample standard deviations are 0.007444 and 0.022997. Three training seeds do
not give a precise confidence interval over all possible training seeds.
Natural, D4-novel and other registered slices are retained in the
[complete report](results/matched-confirm-v1.json).

Each seed plays the same 16 paired openings against registry opponent 0,
`kata1-b6c96-s938496-d1208807`, at 16 KataGo visits. This is an early small
checkpoint, not the strongest KataGo model. The student receives either one
prior evaluation per move or a 16-simulation search budget. Every actual neural
evaluation, including search work, is counted.

| Student | Prior wins | PUCT wins | Gumbel wins | Gumbel capped |
|---|---:|---:|---:|---:|
| Fly, all three seeds | 0 / 96 | 3 / 96 | 0 / 95 | 1 |
| CNN, all three seeds | 94 / 96 | 93 / 96 | 92 / 95 | 1 |

Capped games are outside the completed-game denominator. These are raw panel
counts, not an Elo estimate. The small differences among the CNN's search modes
do not establish that search hurts. The main robust observation is the large
gap between the current network priors.

The experiment contracts are [rate screen](../configs/matched-control-v1.json),
[confirmation](../configs/matched-confirm-v1.json) and
[CPU followups](../configs/matched-confirm-followup-v1.json). Raw checkpoints,
aligned position metrics, games and frozen environments live under
`/dev/shm/flygo/runs`; hashes are preserved in the report. Final test labels
remain outside model selection.

## Depth and readout width on the earlier prototype release

Eight runs compare K=4/8, G=656/2,624 and seeds 1/2 at 320,000 exposures.
This release differs from V0; compare these runs within their own study.
Doubling recurrent passes does not consistently improve validation. At G=656,
the paired mean K8-minus-K4 change is +0.002910 KL and +0.003697 MSE; at
G=2,624 it is +0.000008 KL and +0.018119 MSE. Wider readouts improve policy KL
by 0.020626 at K4 and 0.023528 at K8, while mean MSE worsens by 0.005907 and
0.020329. Retain the individual seeds and their disagreement on value quality.

Evidence: `prototype-followup-v1/{depth-g656,depth-g2624,width-k4,width-k8}-report-v1.json`.
These results motivate testing informative spatial adapters and optimization
before simply increasing recurrent depth.

## Optimizer confirmation on V0

All nine runs completed 4,000 updates at B32: 128,000 exposures, K4, G656,
random ports and paired seeds 1/2/3. Each final checkpoint received the same
full validation. These CPU runs use one immutable source and compare within
this smaller-batch protocol.

| Global rate / bias multiplier | Mean policy KL | Mean value MSE | Mean teacher agreement |
|---|---:|---:|---:|
| .003 / 1 | 1.594022 | .529932 | 19.49% |
| .01 / 1 | 1.538219 | .564571 | 20.89% |
| .03 / .01 | 1.517230 | .565524 | 22.84% |

Both larger rates improve policy in all three seeds, but worsen value MSE
relative to .003. The paired .01-minus-.003 differences are −.055804 KL
[−.059186, −.052607] and +.034640 MSE [.028237, .040670]. Bias-scaled
.03-minus-.003 gives −.076793 KL [−.080588, −.072895] and +.035593 MSE
[.024479, .046810]. These intervals bootstrap complete games conditional on
the trained checkpoints; corresponding paired-seed standard deviations are
.007589/.012436 and .016940/.008737 for KL/MSE.

Compared with .01, bias-scaled .03 improves KL by .020989
[.019169, .022894], with no resolved mean value difference: candidate-minus-
reference MSE +.000953 [−.010527, .013319]. Policy and value therefore remain
separate selection criteria, and playing strength requires its own panel.
All individual seeds, slices, source hashes and contrasts are in
[the complete optimizer report](results/optimizer-confirm-v1.json).
The schedule and smooth-rate confirmations test whether these tradeoffs can
be improved without changing the fixed topology.

## Schedule screen on V0

All four seed-1 schedule trials use K4, G656, B32 and 128,000 position
exposures. Warmup reaches the declared peak at update 500; cosine alone reaches
0.1 times peak at update 4,000. This is a one-seed screen, separate from the
large-batch matched-control study.

| Peak rate / bias multiplier | Schedule | Full policy KL | Full value MSE |
|---|---|---:|---:|
| .01 / 1 | Constant | 1.546345 | .558619 |
| .01 / 1 | Warmup | 1.528106 | .555978 |
| .01 / 1 | Cosine | 1.536014 | .567042 |
| .03 / .01 | Constant | 1.534333 | .542177 |
| .03 / .01 | Warmup | 1.518023 | .521903 |
| .03 / .01 | Cosine | 1.507110 | .534657 |

For the bias-scaled family, warmup-minus-constant changes are −.016311 KL
and −.020274 MSE; conditional game-bootstrap intervals are [−.019363,
−.013529] and [−.032393, −.007823]. Cosine changes are −.027223 KL and
−.007520 MSE; intervals are [−.029696, −.024830] and [−.017281, .001782].
The value interval for cosine includes zero. Both candidates advance to seeds
2/3 because they trade policy and value quality; neither is declared a general
winner from this seed. Plans:
[schedule screen](../configs/optimizer-schedules-v1.json),
[confirmation](../configs/optimizer-schedules-confirm-v1.json).
Evidence: `optimizer-schedules-validation-v1/*-report-v1.json`.

## CPU cost and implementation

On 256 predeclared validation inputs, the three matched fly checkpoints use
8.340M, 8.312M and 8.874M counted warm B1 operations after exact zero skipping
and caching the constant initial message. At B32, the corresponding per-position
counts are 9.969M, 10.174M and 10.510M because a source row executes when any
batch element is active. The initial network is much denser; learning changes
execution cost. The ledger counts arithmetic under stated conventions, rather
than hardware instructions, and excludes memory traffic and nonlinear costs.
An 8.362M-FLOP CNN shape was selected from cost before its first training trial;
its rate screen and confirmation will test a closer cost scale. See
[the registered smaller control](../configs/small-control-screen-v1.json).

The optimized CPU keeps the same node/edge arrays and reduction order. A
constant-state message cache costs 660,488 bytes; transpose indices cost
122.2 MB, and packed weights use 61.1 MB during backward. Independent rate
entries and readout groups now execute in parallel. At 24 cores and B32,
softness .01 inference/gradient time changes from .558/1.711 s to .239/.916 s;
softness .05 changes from .673/1.945 s to .242/.925 s. These are warmed,
full-graph six-update fixtures, not long-trained playing-strength results.
Every output, loss and gradient byte matches, as do three fresh scheduled,
scaled continuation updates. Evidence: `cpu-smooth-lane-v1` and
`cpu-smooth-lane-s{01,05}-{serial,parallel}-v1`.

## Visual readout screen

The input-only reference and both visual-output variants use the same spatial
input, K4, G656, B32, seed 1 and 128,000 exposures. Visual variants pool 23,565
annotated cells at 60 board points; all recurrent neurons and edges still
execute. The shuffled control preserves cell populations and group counts.

| Readout | Policy KL | Value MSE |
|---|---:|---:|
| Broad original readout | 1.544321 | .553408 |
| Spatial visual readout | 1.670303 | .579456 |
| Shuffled visual readout | 1.693978 | .628023 |

Spatial-minus-shuffled changes favor spatial ordering: −.023675 KL with
conditional game-bootstrap interval [−.026308, −.020890], and −.048567 MSE
[−.060980, −.035817]. However, spatial-minus-broad readout is worse by .125982
KL [.120460, .131583] and .026048 MSE [.008390, .043387]. Restricting output to
these visual cells loses useful information under this protocol. This one-seed
result motivates preserving broader output access in future adapter studies;
it does not establish a general spatial benefit across training seeds.

Changing only the input from random to this spatial overlay changes KL by
−.002024 [−.003426, −.000703] and MSE by −.005211 [−.013780, .003497]. The
three-seed large-batch input study remains independent. Evidence:
`visual-readout-validation-v1/{spatial-readout,readout-order,input-only}-report-v1.json`.

## Smooth firing-rate screen

The hard rate is ReLU. The smooth alternative is
`r_s(h) = max(h,0) + s log(1 + exp(-abs(h)/s))`, with derivative
`sigmoid(h/s)`. It changes recurrent transmission and final readout together,
without adding parameters or edges. All cases use V0, K4, G656, B32, seed 1
and 32,000 training exposures; validation uses all 70,425 held-out positions.

| Global rate / bias multiplier | Rate softness | Policy KL | Value MSE |
|---|---:|---:|---:|
| .01 / 1 | Hard | 1.655283 | .608872 |
| .01 / 1 | .01 | 1.656904 | .685284 |
| .01 / 1 | .05 | 1.727510 | .615426 |
| .03 / .01 | Hard | 1.649547 | .571104 |
| .03 / .01 | .01 | 1.623348 | .570713 |
| .03 / .01 | .05 | 1.639406 | .571395 |

Smoothness helps only with the bias-scaled optimizer in this screen. At
softness .01 and global rate .03, candidate-minus-hard KL is −.026199,
with conditional game-bootstrap 95% interval [−.028442, −.024079]. Value MSE
changes by −.000391 [−.008248, .007407], and teacher agreement improves by
1.532 percentage points [1.241, 1.824]. Softness .05 gives a smaller KL
improvement, −.010140 [−.012578, −.007591], with no resolved value benefit.
At global .01, softness .01 instead worsens MSE by .076411
[.062115, .089928]; smooth rates are not an unconditional improvement.

The best screen candidate advances to paired seeds 1/2/3 at 128,000 exposures
from initialization, against the existing hard-rate controls. Screen exposures
remain separate from confirmation exposures. This one-seed observation does
not establish a benefit over training seeds or playing strength. The original
hard screen uses a verified compatibility view of its legacy result layout;
no metrics, checkpoints or selection were recomputed. Evidence:
[complete paired reports](results/smooth-rate-screen-v1.json),
[confirmation contract](../configs/smooth-rate-confirm-v1.json).

The subsequent parameter-validation optimization preserves every output/loss/
gradient byte and three fresh scheduled continuation updates. On 24 pinned
cores, the trained hard fixture's median warm B1 inference falls from 54.6 to
35.2 ms, and B32 from 130.7 to 105.6 ms. Gradient times are 200.8 to 180.0 ms
and 557.2 to 545.8 ms. These are three-repeat implementation measurements
under concurrent fleet work, not training outcomes or universal speedups;
initial-fixture gradient timing is slightly worse. Evidence:
`cpu-validation-lane-v1` and its four named profile directories.

## Schedule confirmation, three seeds

All six scheduled cases and three adopted constant controls now have full
70,425-position validation at 128,000 training exposures. K4/G656/B32 and
global rate .03 with bias multiplier .01 are fixed. Each contrast pairs the
same seeds and sampled training positions.

| Schedule | Mean KL | Mean value MSE | Paired KL change vs constant | Paired MSE change |
|---|---:|---:|---:|---:|
| Constant | 1.517230 | .565524 | — | — |
| Cosine to .1 of peak | 1.510981 | .561142 | −.006248 | −.004382 |
| 500-update warmup | 1.506564 | .538036 | −.010666 | −.027489 |

Warmup's conditional game-bootstrap 95% intervals are [−.012368, −.008975]
for KL and [−.035881, −.019969] for MSE. Cosine's MSE interval includes zero.
Warmup improves both losses on average, but its seed-3 policy loss is slightly
worse than the paired constant control. The intervals condition on these
checkpoints; the [paired report](results/schedule-confirm-v1.json) separately
reports variation across the three seeds. These are optimizer findings, not
an established playing-strength improvement.

## Output-weight rate screen

At 32,000 exposures and seed 1, multiplying only value-head weights' learning
rate by 1/sqrt(656) improves the hard-rate control from KL 1.649547 / MSE .571104
to 1.626361 / .549680. Scaling both policy and value weights gives 1.639104 /
.571380. Bias multipliers remain unchanged. The value-only paired differences
are −.023186 KL and −.021424 MSE, with conditional game-bootstrap intervals
[−.025757, −.020519] and [−.030861, −.013168].

The same changes do not transfer to smooth .01 firing: value-only scaling
worsens KL by .003832 and MSE by .012894; scaling both heads worsens KL by
.020725 with no resolved MSE change. Keep the factors isolated. These are
single-seed screens requiring a separately registered confirmation before
adoption. [All four comparisons](results/head-rate-screen-v1.json).

## Small CNN control: screen completed

The 17-channel, nine-block CNN passes the actual four-host TPU shape gate:
states, losses, gradients and three checkpoint-aligned updates meet the
original tolerances. A free-running CPU/TPU trajectory match is not claimed.
The declared 256-update/B2048 rate screen gives final fixed-slice KL+MSE
2.024164 at .001, 1.667881 at .003, and 1.557839 at .01. Thus .01 advances to
three new 512-update initializations under the original selection rule.
[Screen records](results/small-control-screen-v1.json).

The 8,361,890 nominal FLOP architecture was chosen before training against
the fly seed-1 counted warm CPU mean of 8,339,945 operations. Its count is
0.263% higher. This compares expected arithmetic under the declared rules,
not exact per-position work, hardware instructions, or accelerator padding.
Both controls receive 1,048,576 confirmation exposures; final full validation
and the same 32-game prior/PUCT/Gumbel panels are complete for all three seeds.
Operational zero-update failed attempts are retained separately.

| Seed | Small CNN KL | Small CNN MSE | Prior wins | PUCT wins | Gumbel wins |
|---|---:|---:|---:|---:|---:|
| 1 | .892922 | .447026 | 30/32 | 27/32 | 22/32 |
| 2 | .903868 | .477781 | 31/32 | 31/32 | 31/32 |
| 3 | .919386 | .433978 | 28/32 | 29/32 | 31/32 |

Mean small-CNN minus fly differences are **−.599074 KL** and **−.102891 MSE**.
Conditional game-bootstrap 95% intervals are [−.612615, −.584832] and
[−.128282, −.077328]; paired-seed sample SDs are .012656 and .030507.
The smaller CNN wins 89/96 direct-prior games, 87/96 PUCT and 84/96 Gumbel,
with no caps, compared with fly 0/96, 3/96 and 0/95 plus one cap respectively.
The same early KataGo checkpoint, 16 teacher visits and paired openings are
used throughout. These panels are not Elo and do not establish that search
always improves a given learned prior. [Complete paired evidence](results/small-control-confirm-v1.json).

## Spatial input, three seeds

At 1,048,576 exposures, preserving the audited visual attachment improves
policy KL by .003015 versus the original ports and .005344 versus the
matched shuffled attachment. The latter's conditional game-bootstrap interval
is [−.006180, −.004445]. Value differences remain unresolved: spatial minus
shuffle is +.001262 MSE with interval [−.002509, +.005230]. The spatial prior
still wins 0/96 games; PUCT wins 1/96 and Gumbel 3/96.

This supports a small policy effect of the specific retained spatial ordering.
It does not close the CNN gap, establish general retinotopic Go transfer, or
validate all inferred retinal geometry. The complete report includes all
baseline/spatial/shuffled seeds, weighted matched position differences and
raw common-opening panels. [Evidence](results/spatial-input-confirm-v1.json).

## CPU prediction timing and memory

These measurements use the same 24 pinned physical cores on host 0, the same
seeded training-only inputs, and final seed-1 checkpoints. Each batch size
has one fixed input batch and ten warm repetitions after cold preparation.
The fleet continues other work on its assigned cores.

| Model | B1 median latency | B32 median batch time |
|---|---:|---:|
| Rust fly | 23.735 ms | 76.234 ms |
| JAX CPU large CNN | 2.425 ms | 10.134 ms |
| JAX CPU small CNN | 1.298 ms | 5.423 ms |

Latency includes each implementation's kernel launches, indexing, validation
and memory effects; Go feature construction and search are outside the timed
call. It is separate from theoretical FLOPs and does not characterize all Go
positions. [Inputs, repetitions, cold times and RSS](results/prediction-latency-v1.json).
Streaming prediction reduces long-unroll memory without a resolved speedup:
the K32/B128 initial-model probe uses 1.435 GB peak RSS versus 4.005 GB for
the traced path. [Memory comparison](results/streaming-memory-v1.json).

A source-driven sparse-multiply candidate also preserves every output bit,
but takes 13.42 ms versus 6.36 ms on the trained hard-rate B1 fixture and
25.12 ms versus 10.06 ms at B32. Initial and smooth fixtures also regress.
These are individual recurrent-kernel timings, not complete predictions.
The candidate visits active sources in order and partitions destinations
between threads; its traversal and gathers outweigh the saved index scans
in these measurements. It was never enabled in production and has been removed
from the engine. The immutable source, patch and all repetitions are retained.
[Negative profiling result](results/sparse-source-profile-v1.json).

## Adam epsilon screen

All four seed-1 cases have full 70,425-position validation after 32,000 training
exposures, with the same K4/G656/B32, global rate .03 and bias multiplier .01.
Each is compared with its hard or smooth-.01 epsilon-1e-8 control.

| Firing rule | Epsilon | Policy KL | Value MSE |
|---|---:|---:|---:|
| Hard control | 1e-8 | 1.649547 | .571104 |
| Hard | 1e-6 | 1.659809 | .567759 |
| Hard | 1e-4 | 1.741155 | .619257 |
| Smooth .01 control | 1e-8 | 1.623348 | .570713 |
| Smooth .01 | 1e-6 | 1.707159 | .580606 |
| Smooth .01 | 1e-4 | 1.716843 | .590718 |

Every larger-epsilon case worsens policy KL. Hard 1e-6 has no resolved value
change: candidate-minus-control MSE is −.003345 with conditional game-bootstrap
95% interval [−.011485, +.004130]. Its top-1 agreement improves by 1.360
percentage points, showing that agreement and distribution loss can diverge.
The other three cases worsen both losses. Keep the default epsilon 1e-8;
this single-seed screen does not justify a larger-epsilon confirmation.
The hard-1e-6 operational retry consumed no extra optimizer updates before
its successful restart. [All records and paired comparisons](results/epsilon-screen-v1.json).

## Smooth-rate confirmation, three seeds

All three selected softness-.01 runs complete 4,000 updates / 128,000
exposures at B32, global rate .03 and bias multiplier .01. The hard controls
use the same seeds, inputs, exposure horizon and optimizer. Every final
checkpoint has full 70,425-position validation; no final test labels are used.

| Seed | Hard KL | Smooth KL | Hard MSE | Smooth MSE |
|---|---:|---:|---:|---:|
| 1 | 1.534333 | 1.493042 | .542177 | .552616 |
| 2 | 1.513853 | 1.486574 | .576473 | .558474 |
| 3 | 1.503502 | 1.491410 | .577922 | .571197 |
| Mean | 1.517230 | 1.490342 | .565524 | .560762 |

Policy KL improves in all three observed seeds. The mean paired difference
is −.026887, with conditional game-bootstrap 95% interval [−.028444, −.025192]
and paired-seed sample SD .014604. Mean value-MSE difference is −.004762
[−.011764, +.001348], with sample SD .014321; it remains unresolved and
seed 1 worsens. Teacher top-1 agreement increases by .796 percentage points
on average. These bootstrap intervals condition on the three checkpoints,
not a population of possible training seeds.

This confirms a policy-loss benefit at equal exposures for this optimizer
setting. It is not an equi-FLOP or playing-strength result. Smooth rates can
keep more rows active, and the hard-only zero-skipping ledger must be extended
and qualified before counting their effective work. Actual smooth-rate TPU
qualification and fresh match panels remain separate gates. Do not combine
this result with warmup or head-rate scaling without a new controlled study.
[All seeds and paired evidence](results/smooth-rate-confirm-v1.json).
