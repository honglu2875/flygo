# Next phase: understand neuron groups before changing their rules

Status: planned after the current engineering gates and registered studies.
The user will also investigate the biology independently. Keep this phase
deliberate: characterize the circuit, derive a small number of hypotheses,
then test them. The initial [density audit](structural-plasticity.md) is a
starting measurement, not a grouping algorithm or an architectural winner.

The target remains prediction efficiency at equal counted FLOPs and labeled
training exposures. Report parameter counts, training/tuning cost, hardware
padding, latency and search evaluations separately. The existing evidence
does not establish a fly advantage.

## Milestones and deliverables

| Milestone | Deliverable | Gate before advancing |
|---|---|---|
| G1: annotation atlas | Immutable neuron-to-group tables with source hashes, missingness and overlapping labels | Distinguish observed labels from inferred group membership; stable canonical neuron IDs |
| G2: circuit organization | Directed, weighted block maps, recurrence, boundary traffic, degree-controlled and spatial comparisons | Robustness to group size and synapse thresholds; inspect representative cells rather than trusting density alone |
| G3: dynamics in FlyGo | Per-group activity, input sensitivity, timescales and gradient flow over recurrent passes | Same fixed training-only probe inputs and checkpoints; separate initial from trained behavior |
| G4: execution layout | Sparse-plus-masked-dense block prototype with complete-model CPU/TPU profiles | Unchanged graph and model, state/gradient/update parity, realistic batch and padding costs |
| G5: group-specific rules | One motivated neuron-rule change with a matched control | Written equations/state lifetimes, qualified derivatives/recovery, counted cost and multi-seed validation |

## G1: keep several descriptions of a neuron

Avoid forcing every neuron into one universal partition. Retain cell type,
superclass/class, annotated side, visual column, lineage and anatomical
membership as separate axes. A cell can participate in multiple neuropils;
its processes can cross the midline despite its soma label. Connectivity
communities are another inferred view, and should not overwrite annotations.

The local tables already include type, superclass/class, soma/root side,
assigned optic-lobe coordinates, soma locations, developmental annotations,
and transmitter predictions. Audit coverage and meanings before use. In
particular, a field called `receptorType` must not be assumed to specify
postsynaptic neurotransmitter receptors. Soma coordinates do not describe
axon/dendrite overlap, and birth class is not synapse age.

Local chemical environment, receptor abundance, glial regulation, release
probability, time-varying neuromodulator concentrations and physiological time
constants are generally not fully determined by the retained graph. Record
these as unknown or as external physiological priors with explicit provenance.

## G2: determine which groupings explain connectivity

Measure internal and between-group edge density, synapse-count distributions,
reciprocity, incoming/outgoing traffic, sign composition and distance of
sensory input to readout. Report counts and denominators. A small dense group
may account for little computation or depend mostly on external inputs.

Compare type-only, location-only and combined groupings. Include directed
degree-preserving rewiring and spatially constrained nulls; ordinary random
graphs confound degree and locality. Repeat descriptive comparisons at
several declared synapse thresholds while keeping the production graph fixed.
Inspect anatomical examples and contradictory cases. Treat groups inferred
from common input/output partners as candidates, not proven shared functions.

Start with visual neighborhoods, central-complex types, antennal-lobe local
circuits and mushroom-body compartments/subtypes. Their very different sizes
and functions make a useful cross-check against a single universal rule.

## G3: ask what the current model actually uses

The baseline already shares learned **leak and bias by cell type**. Adding
type-specific time constants again is not a new intervention. First inspect
the learned values, firing fractions, state ranges, saturation, correlations,
signal attenuation and gradients within each proposed group. Estimate response
to controlled input perturbations and persistence after input removal; a graph
cycle alone does not establish useful memory.

Use fixed probes from the training partition, including simple legal board
patterns and paired perturbations. Record the baseline initialization and
each declared trained seed. No validation labels should decide group membership
or fit a physiological explanation. Groupwise Go responses describe our
adapted model, not necessarily the living fly's sensory tuning.

## G4: optimize the implementation without changing the hypothesis

Maintain one canonical edge/parameter order and a separate execution layout.
Pack genuinely dense type-to-type or neighborhood blocks, keep missing entries
masked to zero, and process the remainder sparsely. Measure what fraction of
all edge work each packing captures, matrix padding, indexing cost and memory
traffic. Very small blocks may need batching across equivalent neighborhoods
to be useful on TPU; batching is possible without sharing their weights.

Preserve the existing deterministic reference path. Qualify the proposed
layout on full states, losses, gradients, updates and fresh restore, using the
existing tolerances and reporting reduction-order effects. Compare end-to-end
latency at batch sizes relevant to prior inference, search and training.

## G5: derive restrained interventions

Candidate rules include hierarchical type/region parameter sharing, bounded
adaptation, local inhibitory normalization, transmitter-dependent filtering,
or compartment-local modulatory gates. Choose one from the G2/G3 evidence.
Specify whether it adds representational capacity, changes optimization, or
imposes regularization; include a same-capacity shuffled-group control.

For example, a fast rate state plus a slow adaptation state adds an N-by-B
state and a few local operations. That is a concrete dynamics hypothesis;
assigning an arbitrary label such as "dopamine" to an unconstrained learned
gate is not sufficient. Any episode-local state needs explicit reset and
search-branch cloning semantics.

Shared kernels across cell types and visual offsets, or structural regrowth,
are later hypotheses described in the [regrowth note](structural-plasticity.md).
New edges would require a separately identified relaxed-topology experiment.
Current registered models retain the original topology. Do not launch a large
cross-product of groupings and neuron rules before the earlier gates resolve.

## Keep the repository small

Use one atlas artifact under `graphs/<graph_id>/groups/<version>/`, one focused
audit command, and a compact report with interactive block maps when useful.
Add a reusable package module only when two real consumers need it. Keep
anatomical data, execution-layout generation and neuron dynamics in separate
modules; avoid making the trainer aware of biological labels or backend packing.
