# Local regrowth: a proposed extension

The user's September 14 question introduces a useful research direction:
allow new neuron-to-neuron connections within selected groups. This would
relax the original fixed-topology constraint. The experiments already running
retain that constraint; no new connection has been introduced or trained.

## What is dense in our actual graph?

The [audit](../scripts/audit_groups.py) counts our exact Traced MaleCNS v1.0
graph, with at least two confidence-0.5 synapses per retained directed pair.
For a group S, density is the number of distinct internal directed edges,
excluding self edges, divided by |S|(|S|-1). Synapse counts are reported
separately. Whole-graph density is **0.056006%**.

| Annotated group | Neurons per side | Internal directed density |
|---|---:|---:|
| Delta7, central-complex interneurons | 21 | 98.81–99.76% |
| PFNa, a central-complex type | 29 | 98.89–99.38% |
| KCab-p, a Kenyon-cell subtype | 64–65 | 83.49–83.93% |
| ALLN, antennal-lobe local neurons | 208–210 | 23.40–24.66% |
| MBON, mushroom-body output neurons | 48–49 | 13.22–15.03% |
| ALPN, antennal-lobe projection neurons | 343 | 4.87–5.38% |
| CX, central-complex class | 1,471–1,479 | 3.40–3.42% |
| All Kenyon cells | 2,019–2,045 | 3.05–3.10% |

Here, "side" partitions neurons by somaSide, falling back to rootSide. It
does **not** locate each synapse in one hemisphere: some neurons arborize
bilaterally. Delta7 left-to-right and right-to-left blocks are also about
99.5–99.8% dense. EPG-to-Delta7 blocks are 74.1–77.4% dense. Useful dense
blocks can therefore join *different* cell types and cross soma-side groups.

Direct visual-column annotations cover 23,720 neurons in 1,771 groups.
These groups have 3–19 annotated neurons (median 12); median within-group
density is 37.62%, and pooled density is 37.24%. Their 111,160 existing
internal edges occupy 298,518 possible pairs. Adding every missing pair
would add 187,358 edges, **1.23%** of the current 15,270,273 edges.
This is an illustrative upper bound for this incomplete annotated subset,
not a proposal to make complete columns or a claim about all visual neurons.

[Compact numerical evidence](results/group-density-v1.json) includes the
source annotation hash, exact counts, incoming/outgoing fractions, script
hash, and full runtime-report locator. For example, MBON internal edges are
only about 1% of their incoming edges despite the relatively high internal
density. Group size, degree, threshold and incomplete annotations matter;
raw density enrichment over the whole brain is not a statistical test.

## What biology suggests

The visual system contains repeated columnar organization and structured
cell-type coverage. Some types tile a layer with little overlap; others
overlap many columns. The relevant locality is therefore overlap between
presynaptic and postsynaptic arbors in a compatible layer, with cell type
and visual position providing additional constraints. Soma distance alone
is a poor substitute. [Nern et al., Nature 2025](https://www.nature.com/articles/s41586-025-08746-0).

The central complex contains recurrent heading and navigation circuits,
including EPG/PEN/PEG pathways and recurrent Delta7 interactions. Their
structured feedback makes them candidates for dynamics experiments;
already nearly complete Delta7 blocks offer little scope for adding edges.
[Hulse et al., eLife 2021](https://elifesciences.org/articles/66039).

Mushroom-body circuits combine sparse sensory coding, distinct Kenyon-cell
subtypes, compartment-specific KC-to-MBON plasticity, dopamine and recurrent
feedback. Sparse *activity* does not imply absent KC-to-KC connectivity;
our subtype counts make that distinction concrete. Correlating similar
neurons more strongly could also harm the decorrelation useful to learning.
[Li et al., eLife](https://elifesciences.org/articles/62576).

Structural plasticity has direct experimental support: appetitive long-term
olfactory memory was associated with an input-specific increase in functional
microglomeruli in the adult mushroom-body calyx. That supports investigating
activity-dependent structural change; it does not establish that arbitrary
graph densification or any particular growth algorithm is biologically correct.
[Baltruschat et al., Cell Reports 2021](https://doi.org/10.1016/j.celrep.2021.108871).

## Proposed computational contract

Write one recurrent message as

`m = (A0 * W0 + M(t) * V) r(h)`.

Here `*` is elementwise multiplication, A0 is the immutable observed edge
mask, and M(t) is a learned structural mask drawn from a bounded, predefined
candidate set C. W0 remains learnable. New edges have separate parameters V,
provenance and optimizer state. Grouping candidates using annotation and
training-only statistics avoids validation-dependent topology selection.

The first candidate set should use same-side visual-column neighborhoods,
compatible source/target types, and eventually synaptic arbor/layer overlap
when that metadata is available. Include small adjacent-column offsets;
complete isolation within columns would prevent broader spatial integration.
Same-type, functionally similar cells alone are insufficient: circuits often
need complementary excitatory, inhibitory and modulatory types.

Parameter sharing of the form
`V_ij = kernel[type(i), type(j), column(i)-column(j)]`
would impose a convolution-like rule across columns. Restricting locality
without sharing only gives a locally connected network. Hexagonal retinal
coordinates do not automatically produce square-board translations or Go's
D4 symmetries, so the board adapter and symmetry tests remain necessary.

For a small initial growth budget, periodically score a bounded sample of
absent candidate edges, grow useful candidates and retire the same number
of *added* edges. Candidate gradient scores sum the postsynaptic error times
presynaptic activity over the unrolled passes and training batch. Keep the
source-sign constraint, cap incoming magnitudes, initialize new states
explicitly, and save all masks, indices, random state and optimizer moments.
An all-pairs absent-edge gradient would be prohibitively wasteful here.
Magnitude pruning and infrequent gradient-based growth have precedent in
[RigL](https://proceedings.mlr.press/v119/evci20a.html); this local,
sign-constrained adaptation would need independent qualification.

## Experiments and implementation order

1. **Execute the existing graph as sparse plus masked dense blocks.** All
   absent entries remain zero; outputs and gradients must pass the current
   tolerances. Measure complete-model latency and padding cost before claiming
   a speedup. A tiny dense block can be biologically interesting while covering
   too little total work to improve throughput.
2. **Compare a fixed local addition budget with random additions.** Use, for
   example, 0.1% and 1% extra edges, identical initialization scales, exposures,
   seeds and optimizer tuning. Keep both static additions and adaptive growth
   controls, so a gain from extra capacity is distinguishable from useful
   biological grouping or structural learning. These are proposed budgets,
   not registered or started trials.
3. **Test a fixed-total-edge reallocation separately.** Pruning observed edges
   to fund growth changes the biological backbone more substantially. It needs
   an explicit relaxed-topology contract and a degree-matched random rewiring
   control. Equal edge counts still do not guarantee equal executed FLOPs:
   firing sparsity, passes and padding must be recounted.
4. **Test shared local kernels and dopamine-modulated growth separately.**
   Establish the simpler gradient-based result first. Count topology-search
   overhead in training compute; dopamine is not a substitute for a specified
   credit-assignment rule.

For TPU, use fixed-capacity index/value arrays and update them only at agreed
training boundaries. Rebuild and verify the CPU transpose mapping, remap or
reset moments for changed edges, broadcast one canonical graph version to all
SPMD workers, and preserve static tensor shapes to avoid recompilation per
growth event. Freeze the final topology for inference. This remains a design;
the current engine and checkpoints continue to use the observed fixed graph.
