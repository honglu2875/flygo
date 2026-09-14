# Exact dependencies for CPU prediction

This is an optional implementation experiment. It changes neither topology nor
the recurrence, attachment or learned parameters. Existing training and full
traces retain their complete computation. Scientific confirmation runs continue
on their original immutable source.

Let R_K be all neurons attached to a readout after K passes. Working backwards,
R_(t-1) = R_t union predecessors(R_t). The first term includes the learned leak's
self-dependency. Compute only rows in R_t at pass t, retaining every incoming
edge and its original reduction order for those rows. The constant initial
message remains cached. A plan depends only on the graph, readouts and horizon;
parameter updates invalidate the existing weight cache but not the plan.

For the present K8 spherical adapter, the independent topology audit predicts
83,534,872 post-cache incoming-edge evaluations instead of 106,891,911, before
activity-dependent zero skipping. This is about 22% fewer edges, not a measured
latency or complete FLOP reduction. Both paths still validate all parameter and
input arrays, and evaluate firing rates across their allocated state buffers.

The candidate API is `model.infer(features, prune=True)`; the default remains
unchanged. Pruning cannot return a full-state trace or a persistent neural state.
It checks finiteness in computed rows; a divergent branch outside the output's
finite-horizon dependencies is not evaluated. Use the complete path when global
state-divergence diagnostics are required. On finite complete trajectories,
policy logits and value predictions must match bit for bit.

Before adoption, require fixtures with cycles, disconnected components, leak
dependencies, K1/K2/K4/K8/K32, B1/B3/B32, hard/smooth rates and conditioned
readouts, including parameter updates and restores. Then check the actual
seed-1 initial/final spherical checkpoints against the frozen baseline on fixed
training inputs, and measure paired warm latency on the same CPU lane. Keep
cold preparation, dependency construction and memory overhead separate. A
slower candidate is retained as evidence or removed; graph counts alone cannot
justify adoption. TPU remains outside this experiment.

## Qualification and measured latency

The candidate is available as an opt-in after **49 Rust and 95 Python tests**
pass. The unchanged, separately invoked million-transition Go qualification is
ignored in this routine suite. The full-CNS gate covers all six seed-1
initial/final spherical checkpoints at B1/B32. Complete states and predictions
match the frozen implementation; pruned predictions match the complete path.
The history checkpoints additionally pass three in-memory updates at both
batch sizes: every gradient, loss, parameter and Adam array matches exactly.
The largest separate FP64 diagnostic-norm difference is 1.78e-15.

The balanced timing follow-up uses fresh processes, three rounds, seven warm
measurements per process, and the same 24 physical cores. Each checkpoint/batch
cycles through all three implementation orders. All 54 processes return the
same prediction bytes for their matched inputs.

| Input | Batch | Candidate full, ms | Candidate pruned, ms |
|---|---:|---:|---:|
| History | 1 | 93.64 | 78.85 |
| Current | 1 | 92.83 | 77.51 |
| Neutral | 1 | 89.32 | 79.29 |
| History | 32 | 309.36 | 275.81 |
| Current | 32 | 297.16 | 261.48 |
| Neutral | 32 | 318.94 | 279.10 |

Entries are medians of the three process medians, in milliseconds per batch.
Across the nine matched round/case ratios, median warm latency falls **14.7% at
B1** and **12.6% at B32**, compared with pruning disabled in the same binary.
The observed reductions range from 10.7–18.3% and 9.8–15.2%, respectively.
These are measurements under concurrent host workloads, not a general speedup
guarantee. The older frozen binary is retained as a separate comparator.

Cold calls cost more: across cases/rounds their median is 197.9 versus 126.5 ms
at B1, and 394.0 versus 347.0 ms at B32. Cold time includes the dependency plan
and parameter-cache preparation. Whole-process peak RSS is about 1.32/1.34 GB;
it includes loading and cannot isolate the plan's incremental memory cost.
The first timing run accidentally repeated each case's order across rounds;
its records remain alongside the corrected follow-up.

The [complete evidence](results/prediction-dependencies-v1.json) identifies
candidate environment `8ec37ee2188a52129f9c`, frozen references, native/source
hashes, tests and all comparisons. A complete pruned arithmetic ledger remains
pending. Existing scientific studies retain their unpruned work counts and
original immutable runtimes; this result establishes no learnability or
matched-FLOP advantage over a CNN.
