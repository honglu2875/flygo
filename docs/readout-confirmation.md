# Fresh-seed readout confirmation

**The soma-side policy benefit did not replicate. Keep the dense learned
projection as the reference.** Soma-side improves value MSE in all three
fresh seeds, but worsens policy KL in all three and fails the registered
confirmation rule.

[Frozen analysis](../configs/readout-confirmation-analysis-v2.json) ·
[All endpoints, contrasts and diagnostics](results/readout-confirmation-v2.json) ·
[Discovery and qualification history](readout-roles-study.md).

## Complete fixed-horizon results

Dense, soma-side and the same frozen shuffled-side mask use fresh head/sampler
seeds 7/8/9, the same initial circuit strengths, current spherical input B,
eight hard-rate passes and 32,000 labeled-position exposures per endpoint.
Learning rate, warmup, epsilon, loss and numerical runtime are shared. All
nine endpoints were evaluated on all 70,425 validation positions from 800
games / 254 opening families. Final test data was not accessed.

| Seed | Head | Policy KL | Value MSE | Teacher top-1 |
|---|---|---:|---:|---:|
| 7 | Dense | 1.66057 | .63139 | 19.06% |
| 7 | Soma-side | 1.70402 | .61860 | 19.10% |
| 7 | Shuffled-side | 1.67979 | .63819 | 19.78% |
| 8 | Dense | 1.68379 | .62287 | 18.79% |
| 8 | Soma-side | 1.68457 | .61186 | 18.43% |
| 8 | Shuffled-side | 1.67430 | .60135 | 20.44% |
| 9 | Dense | 1.65533 | .65813 | 19.31% |
| 9 | Soma-side | 1.66730 | .60366 | 18.67% |
| 9 | Shuffled-side | 1.67887 | .79361 | 17.84% |
| Mean | Dense | **1.66656** | **.63746** | **19.05%** |
| Mean | Soma-side | **1.68530** | **.61138** | **18.73%** |
| Mean | Shuffled-side | **1.67766** | **.67771** | **19.35%** |

All declared contrasts are retained. Differences are candidate minus reference;
lower KL/MSE are better. Agreement differences use percentage points.

| Contrast | Policy KL difference [conditional family 95% interval] | Value MSE difference [interval] | Agreement difference |
|---|---:|---:|---:|
| Soma-side − dense | +.01874 [.01018, .02298] | −.02609 [−.04162, .00753] | −.320 pp |
| Soma-side − shuffled | +.00764 [.00145, .01055] | −.06634 [−.08474, −.05682] | −.621 pp |
| Shuffled − dense | +.01109 [.00709, .01265] | +.04025 [.02147, .08338] | +.301 pp |

The paired soma-minus-dense policy differences are **+.04345 / +.000785 /
+.01197**, with sample SD **.02212**. Its policy differences from shuffled
are **+.02423 / +.01027 / −.01158**. Thus neither the dense comparison nor
the anatomical-versus-shuffled condition replicates the discovery result.
The value improvement over dense is consistent in these fitted seeds, while
its conditional family interval includes zero.

On the predeclared 62,984-position current-source-novel slice, mean policy
differences remain unfavorable: soma-minus-dense **+.00605**, soma-minus-shuffled
**+.00346**, shuffled-minus-dense **+.00259**. All per-seed results, value and
agreement metrics, and both slices are in the complete report. Source novelty
precedes retinal rendering; it does not certify absence of rendered collisions.

The opening-family bootstrap uses 1,000 resamples and seed 709. Its intervals
condition on fitted weights; paired-seed SD is reported separately. Discovery
and confirmation reuse the same validation population, so this is a training-seed
confirmation, not an untouched-data test. Reusing one shuffled partition does
not establish robustness to arbitrary partitions.

## Arithmetic, qualification and decision

Mean complete, unpruned prediction arithmetic on the registered warm B1 sample
is **179.58M / 174.33M / 177.75M FLOPs** for dense/soma/shuffled; warm B32 gives
**185.66M / 180.84M / 183.79M** per position. Renderer and masked-head work are
included. Zero skipping makes work depend on trained activity. The small
arithmetic reduction for soma-side does not repair its fixed-horizon
confirmation result, and is not an independently measured latency advantage.
This study does not inherit the earlier matched-CNN budget.

All nine initial/final checkpoint pairs have verified second-worker copies.
All endpoint identity, disabled-mask, response, decoder and arithmetic checks
pass. Decoder policy logits reconstruct exactly; value errors are at most
1.2e-7. All nine registered full validations, 256-position training signal
probes and 64-position arithmetic probes are complete. The analysis and
evidence have checked copies; owner processes have exited. Scientific training
cost is **288,000 labeled-position exposures** across nine runs, with
qualification and candidate discovery accounted for separately.

The final shuffled launch required an operational recovery. Renaming the
queue entry fixed its standard-library import collision, but its next attempt
failed CPU discovery because discovery inherited housekeeping affinity. Direct
deployment used the identical frozen helpers and plan with the original
discovery affinity; the deployer pinned housekeeping work afterward. No
learner was started by the failed discovery attempt. Failed records remain
intact; masks, source, optimizer and horizons were unchanged.

Keep dense for the next isolated study. The [signal-flow audit](signal-flow-study.md)
provides a stronger mechanism lead: typical visual responses remain weak,
type-bias gradients dominate, and most saved edge Adam denominators are
epsilon dominated. A [global-versus-per-group clipping comparison](../configs/group-clipping-study-v1.json)
is registered, with implementation and qualification still pending. No new
propagation rule, final-test result, playing-strength gain or biological
advantage is established. TPU remains paused.
