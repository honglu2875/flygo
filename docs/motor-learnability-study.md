# Can the motor features support a better policy decoder?

The [36-case screen](results/motor-learnability-v1.json) finishes all declared
endpoints. A raw linear residual improves mean held-out-subset policy KL from
**1.68393 to 1.58499** at rate .03, with improvement in all three seeds.
Aggressive rates and competitive gating can make policies very sharp, but the
tested settings worsen validation. This does not close the CNN gap.

The [registered plan](../configs/motor-learnability-v1.json) freezes three dense
B current-board/K8 checkpoints (seeds 4/5/6), each previously trained for 32,000
labeled exposures. Each supplies 2,129 raw motor rates **before learned readout
gains**, together with its original logits. The core and value predictions stay
fixed. All residual output weights/biases start at zero, so every case starts
with exactly the same predictions as its source checkpoint.

The common bank has 16,384 training positions from 4,158 opening families and
8,192 validation positions from 254 separate families, drawn from V0. Training
has one fixed independently sampled D4 orientation per position; validation
uses its natural orientation. Each head gets 1,024 B128 updates / 131,072 labeled
exposures, sampled with replacement from that bank. This is additional training
beyond the core's 32k exposures; all 36 fits consume 4,718,592 head exposures.
These are frozen-subset diagnostics, not full-validation, full-network,
playing-strength or matched-CNN experiments.

Four external decoders are fitted with FP64 Adam, epsilon 1e-8, clipping at 1,
and the three constant rates below. Raw linear uses the rates directly;
standardized linear uses training-only mean and standard deviation (floor
1e-6). Competitive gating retains at most the 128 largest positive standardized
deviations per position. The MLP adds 256 ReLU hidden units to standardized
features. No recurrent firing rule changes in this screen.

| Decoder | Rate | Mean train KL | Mean validation KL | Seed 4 val KL | Seed 5 val KL | Seed 6 val KL |
|---|---:|---:|---:|---:|---:|---:|
| raw-linear | 0.003 | 1.4058 | 1.6095 | 1.6009 | 1.6098 | 1.6178 |
| raw-linear | 0.03 | 1.3443 | 1.5850 | 1.5745 | 1.5706 | 1.6098 |
| raw-linear | 0.3 | 1.4393 | 1.8246 | 1.7979 | 1.7856 | 1.8904 |
| scaled-linear | 0.003 | 1.3620 | 2.0879 | 2.0589 | 2.0878 | 2.1169 |
| scaled-linear | 0.03 | 7.7811 | 9.8932 | 9.4765 | 10.1229 | 10.0801 |
| scaled-linear | 0.3 | 110.3586 | 129.9584 | 117.3393 | 144.0271 | 128.5088 |
| gated-linear | 0.003 | 1.0092 | 2.0042 | 1.9691 | 2.0172 | 2.0262 |
| gated-linear | 0.03 | 5.6044 | 8.5152 | 7.7248 | 9.0215 | 8.7992 |
| gated-linear | 0.3 | 78.7179 | 106.8476 | 101.9896 | 110.0558 | 108.4974 |
| scaled-mlp | 0.003 | 1.2212 | 1.9301 | 1.9760 | 1.9221 | 1.8922 |
| scaled-mlp | 0.03 | 15.3599 | 17.6200 | 19.4020 | 20.4964 | 12.9617 |
| scaled-mlp | 0.3 | 2012.7684 | 2300.1421 | 2172.2309 | 2742.1889 | 1986.0064 |

At rate .003, gating lowers training KL to **1.0092** but validation is **2.0042**.
At rate .3, the standardized linear head's average largest move probability
reaches **.9739**, while validation KL reaches **129.9584**. Finite completion
is not successful learning; every bad endpoint is retained. Confidence alone
is therefore insufficient, and overfitting/optimization must be separated from
whether the motor representation contains useful information.

Raw and standardized linear heads have the same representable functions:

```
A ((r - mean) / scale) + b
    = (A diag(1 / scale)) r + (b - A diag(1 / scale) mean)
```

The positive scale floor and learned bias make this an invertible change of
coordinates. Different results between them cannot establish different
information content. The tested Adam rates are not equivalent function-space
steps under this change. A regularized decoder fitted with a convergence-aware
optimizer is the next useful diagnostic before claiming a motor-information
limit. That [27-fit convergence diagnostic](motor-convergence-study.md) is now
complete: ridge .01 reaches mean validation KL 1.56892 with standardized linear
features and 1.54872 with gating, while weaker regularization overfits. A
subsequent propagation study should hold that decoder fixed and
measure both signal and gradients through the circuit. The smaller soma-side
confirmation remains a secondary control.

The three banks have 1,029 / 1,259 / 1,216 motors with training standard deviation
above 1e-6. These are total responses, including changing nonvisual context;
they are not the earlier current-minus-neutral visual-response counts.

A linear residual stores 174,660 parameters and adds 349,156 dense-matmul FLOPs;
the MLP stores 566,354 and adds 1,132,032. These counts exclude feature
normalization, gating/sorting, bias additions and the original circuit/head.
They are not complete prediction costs or a latency comparison. Equivalent
linear heads could later be folded into the original decoder after separate
numerical qualification.

Four focused tests passed before fitting: equal initial predictions, training-only
normalization/gating, directional derivatives with legal masks and nonunit target
mass, and exact next-update continuation from copied parameters/moments. Analysis
verifies pairing, fixed core, source identities and every bank/checkpoint hash.
All raw banks, endpoint heads, curves and numerical diagnostics have checked
second copies on another worker. Host-specific launch metadata is preserved
under separate owner directories in `runs/cpu-learnability-archive-v1`.
