# Controlled results

The current fly baseline is substantially weaker than the residual CNN at the
registered nominal prediction cost and training horizon. This is a result for
these adapters, dynamics and training settings. It does not establish the best
performance attainable with the fixed fly topology.

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
