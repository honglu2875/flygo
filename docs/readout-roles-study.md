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
