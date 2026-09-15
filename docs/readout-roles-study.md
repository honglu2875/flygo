# Learnable policy and value readouts

The user's readout proposal treats Go actions as combinations of motor signals.
The current baseline already has a jointly learned dense projection from all
2,129 individual motor features to 82 policy logits, including pass, and a
separate learned projection to a scalar before tanh. Each motor's readout gain
also learns. There is no requirement that one biological motor represent one
Go action. Keep dense learned mixing as the baseline.
For 9×9, the policy matrix has shape 82×2,129. Both heads learn jointly with
the circuit through the existing policy/value losses and Adam updates.

For raw motor rates r, diagonal learned gains D, policy weights P, value
weights v and biases:

```
policy_logits = P D r + b_policy
value         = tanh(v^T D r + b_value)
```

Raw activation variance alone does not establish whether these features are
adequate for either task. A high-variance neuron could supply a global value
signal while smaller, distinct signals support policy. The user proposes
deliberately assigning such a neuron, or a small group, to the value readout.
That is an output-attachment experiment, separate from changing retinal inputs,
neuronal equations or optimization.

## Proposed factors

1. **Selective value readout.** Restrict the trainable value projection to a
   declared candidate group, while retaining the dense trainable policy head.
   Candidate discovery uses training-position responses only. Freeze the
   selection and initialization rule before scientific training; do not pick
   an endpoint or cell using its validation score. The dominant identity varies
   across the existing seeds, so a universal single-cell role is unproven.
   If discovery uses a pretrained circuit, report its exposures and computation;
   a matched comparison cannot treat that pretraining as free.
2. **Side-aware policy readout.** Keep learned coefficients within anatomically
   defined left/right blocks. Left-associated motors supply board columns 0–3;
   right-associated motors supply columns 5–8. Column 4 and pass use both sides;
   unassigned cells remain shared. Audit anatomical side fields and their
   uncertainty first. Soma side is an explicit proxy, not proof of receptive
   field or motor output laterality. A shuffled-side control can distinguish
   anatomical organization from a generic projection constraint.

The [annotation audit](results/retinal-allocation-engineering-v1.json) finds
no populated `rootSide` field among these 2,129 motors. `somaSide` is populated:

| Motor superclass | Left soma | Right soma | Midline |
|---|---:|---:|---:|
| Descending | 656 | 648 | 10 |
| Central-brain motor | 54 | 53 | 0 |
| Ventral-nerve-cord motor | 355 | 353 | 0 |
| Total | 1,065 | 1,054 | 10 |

A soma-side policy experiment would name that proxy explicitly and retain the
ten midline cells as shared features. These counts do not establish functional
laterality or require restricting the dense decoder.

Test these factors separately against the dense baseline on the same frozen
input mapping, K, recurrence, optimizer and exposure horizon. Only consider a
combined head after the individual comparisons. All allowed projection weights
and biases remain learnable. Preserve every CNS neuron and edge; head masks
would constrain the external decoder, not recurrent connectivity.

An exploratory diagnostic may reconstruct the existing learned heads from
motor responses and retained coefficients, examine value versus relative-action
contributions, and temporarily remove a neuron's readout contribution. A common
shift to all legal action logits leaves policy probabilities unchanged. Such
fixed-checkpoint diagnostics help define candidates; they do not establish that
a newly trained selective decoder is better.

Full state/loss/all-gradient/optimizer parity, checkpoint continuation and actual
decoder arithmetic must be qualified before training a new head. Count its
work in the complete prediction budget. Compare fixed-horizon validation and
later controlled play; anatomical resemblance alone is not the acceptance
criterion. TPU remains paused. No selective or side-aware head has been trained.

## What the existing learned projections use

The [three-seed decoder diagnostic](results/readout-contributions-v1.json)
uses the existing B current-input checkpoints and the same 256 training-family
positions as the motor probes. It reads retained responses and coefficients;
no neural updates or new target labels are used. Reconstructing the native head
in its canonical FP32 summation order reproduces every policy logit exactly;
value differences are at most 1.2e-7 from the tanh implementation.

For each seed, select the motor with the largest across-position variance in
`r_current - r_neutral`. At the decoder only, replace that motor's current
response with its neutral-eye response at the same context. Leave other motor
features unchanged. This intervention gives:

| Seed | Leading motor | Value RMS change | Policy top-choice changes | Mean policy KL |
|---|---|---:|---:|---:|
| 1 | DNg30, body 10123 | .538 | 7.81% | .01961 |
| 2 | DNg30, body 10237 | .525 | 4.69% | .01562 |
| 3 | DNpe018, body 69173 | .597 | 15.23% | .02309 |

Values lie in [-1,1]. Policy KL compares the original distribution with the
intervened distribution over the same legal moves. These results demonstrate
substantial value influence and measurable policy influence. They do not score
either prediction against the teacher or establish a biological value function.

The diagnostic also centers each motor's visual response across positions and
projects that variation through the learned heads. For policy it removes each
position's mean legal logit, since a common shift has no policy effect. Removing
the leading motor leaves relative-action signal-energy ratios of 6.97e-6,
2.33e-7 and .00356, and pre-tanh value-energy ratios of 4.97e-9, 1.78e-9 and
.000129. The smaller motor signals have not been substantially amplified by
these fitted linear heads. Ratios describe the residual of a sum; they are not
additive variance allocations and can exceed one when contributions cancel.

The [diagnostic plan](../configs/readout-contributions-v1.json) declares top-one
and top-eight response-ranked subsets; both are retained in the report. Tests
cover invariance to common legal-logit shifts and cancellation. Raw responses,
source checkpoints, diagnostic code and report identities remain linked. This
supports testing deliberate value routing while retaining the dense policy
decoder as its paired control. It does not justify discarding the dense baseline.

## Next implementation gate

Use binary masks on the external policy and value matrices. A disabled
coefficient contributes neither to the forward result nor to any gradient,
optimizer moment, or clipping norm. Preserve the original recurrent neuron and
edge identities. An all-enabled mask must reproduce the original dense model's
states, losses, gradients and updates. The mask and its provenance belong in the
checkpoint contract; recovering with a different mask must fail.

The three proposed alternatives remain separate:

- **Value candidate group:** allow only the complete annotated DNg30 and DNpe018
  types to feed value; keep policy dense. The current motor inventory contains
  five such cells: DNg30 bodies 10123/10237 and DNpe018 bodies
  69173/113166/165031. This includes every member of the two types, rather than
  choosing one seed's winning body ID. It is a response-motivated candidate,
  not an established physiological value circuit.
- **Soma-side policy:** the previously defined left/right/midline blocks, with
  dense value. It keeps 98,294 of the 174,578 policy coefficients enabled.
- **Shuffled-side policy:** permute left/right assignments within each motor
  superclass using one frozen seed, preserving each class's side counts and
  the ten shared midline identities. It has the same per-action fan-in as the
  soma-side mask. This tests whether the anatomical assignment adds value
  beyond the block constraint itself.

Use a newly trained dense control and new paired head/sampler seeds 4/5/6 for
the screen. Seeds 1/2/3 informed candidate discovery. Record those earlier
exposures as development cost. Keep all enabled initial coefficients at their
original values, with no fan-in gain correction. Use one frozen current-only
input map for every arm, chosen and documented after closing the retinal study;
do not change inputs inside a head comparison. Initial core strengths remain
identical across seeds. Fix K8, B32, 1,000 updates, the same losses and the
qualified epsilon/rate schedule unless a separate numerical gate prevents it.

Implementation, immutable mask preparation, actual CPU/JAX derivative/update
and recovery gates, and a mask-aware arithmetic ledger precede a launch plan.
No masked-head model is implemented or trained yet. Subsequent persistent-state
and optimization experiments retain their own comparison contracts.

The contribution audit also motivates a later conditioning experiment. For an
invertible diagonal scale S, `policy = W S (r - mean) + bias` has the same affine
function class as the dense decoder. It could make weak motor variations easier
to optimize, but supplies no new information. Statistics must come from training
positions, with a declared variance floor and fixed fitting cost. Treat this as
an optimization/parameterization factor; do not silently combine it with the
value-group or side-mask trials. The earlier mean-conditioning qualification
failure remains part of the numerical history.
