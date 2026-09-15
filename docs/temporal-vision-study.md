# Current-board vision and memory across moves

Design revision, 2026-09-14. The user proposes
giving both eyes the current board and letting recurrent state carry history.
This follows the [Doomfly source review](doomfly-review.md). Existing runs,
including the negative spherical embedding pilot, keep their original contracts.
The first [supervised attachment screen](attachment-study.md) completes A/B and
a neutral visual control at 32,000 exposures each, including full validation.
Visual drive becomes measurable but concentrates heavily in one descending
neuron. The [C map](retinal-allocation-study.md) now passes geometry, numerical
and recovery gates; three paired CPU trials are running. D remains planned;
no persistent model is being trained yet. Dense jointly learned policy/value
heads remain fixed within the input comparison; their structural alternatives
have a [separate design](readout-roles-study.md).

The broader research order is attachment, execution, then optimization. Within
attachment, vary inputs before outputs. A change to a loss, optimizer, synaptic
sign, firing rule or recurrent depth must not silently accompany an input map.
Signal and gradient measurements belong in each gate; interventions suggested
by those measurements belong in their own comparison.

## Two different kinds of recurrence

Let t count actual Go plies and k count internal circuit passes. With board x_t,
context c_t, sensory map E, and a fixed-topology recurrent operator F_theta:

```
h[t, k+1] = F_theta(h[t, k], E(x_t, c_t)),  k = 0, ..., K-1
policy_t, value_t = readout(h[t, K], legal_t)

reset model:       h[t, 0] = h_initial
persistent model:  h[t, 0] = h[t-1, K]
```

The current Rust/JAX models implement the reset model. Increasing K adds work
within one prediction; it does not supply previous game positions. Persistent
state is an additional execution capability. It might retain useful history,
forget it, or accumulate irrelevant activity. Recurrence alone guarantees none
of those outcomes. Do not label an internal pass with biological milliseconds.

## Ordered comparisons

These are design stages, not completed run registrations. Before each launch,
freeze source and data hashes, exact maps, initialization, optimizer, seeds,
batch/sequence lengths, fixed final exposure horizon and validation positions.
Use ordinary supervised policy/value targets for the Go learnability study;
the earlier contrastive objective remains a separate representation experiment.

| Stage | Visual input | State at a new ply | Question |
|---|---|---|---|
| A | Four-board spherical montage, L: 0/2, R: 1/3 | Reset | Matched supervised reference |
| B | Current board copied into the same four patch locations | Reset | Does removing older imagery help with geometry held fixed? |
| C | One current-board image per eye | Reset | Does allocating more retinal area to one board help? |
| D | Same spatial map as C, with a paired color-stable reset control | Carry from the preceding ply | Does learned memory improve this exact interface? |

A/B use identical sample positions and all nonvisual context, ports, neuronal
equations, K and optimization. They intentionally differ in historical
information; describe the result as a feature ablation. B/C isolate the spatial
allocation change while holding the information fixed. If C fails the sampling
or numerical gate, retain the failure and use B as the explicitly named input
for both reset and persistent comparisons. Do not select maps using test labels.
Keep the anatomical motor candidates/readout fixed in this sequence; output
selection and pooling receive their own later attachment comparison.

Both eyes view the same two-dimensional board. This supplies bilateral visual
input, without inventing stereo depth or a temporal offset between eyes. Use
the established common spherical chart and its uncertainty, with one fixed
orientation convention. Qualify coverage, sampling rank, local perturbation
responses and total drive by eye/type before training. A coherent image does
not establish that every receptor subtype's response model is physiological.

Current turn, komi, consecutive passes and engine-computed legality remain
explicit context. They are not all recoverable from a stone image. Keep the
context map identical across A-D after its own preparation/qualification;
none of these arms should silently omit it as the contrastive pilot did.
The engine retains exact superko history and enforces legality. The network's
learned memory does not replace rule enforcement.

The proposed route through zero-in-degree nonvisual sensory cells is a separate
attachment factor. Audit source edges as well as the fixed filtered graph,
exclude cells without outgoing connections, and retain annotation uncertainty.
The two-contact filter can create apparent source nodes. Older moves and
captures through that route remain a later alternative, not additions to D.

## A consistent trajectory interface

For D, prefer absolute black/white stone encoding plus side to move, applied to
its reset control too. Otherwise the existing own/opponent encoding reverses
the apparent identity of every stone each ply while the hidden state persists.
Qualify that perspective change separately before the carry/reset comparison;
do not attribute its effect to memory. Teacher value targets remain explicitly
in the current player's frame, with unit checks for any conversion.
The primary D comparison is against a newly trained reset control with that
same absolute-color encoding, rather than the earlier current-player C arm.

Advance the state once for every observed ply, including opponent moves and
passes. Predict from the board before the action label is applied. Reset at
game boundaries; never carry between unrelated sampled positions or opening
families. Use one spatial D4 transform throughout a trajectory segment. A
midgame evaluation must state how its hidden state was reconstructed.

Train ordered replay segments with truncated backpropagation through time,
with declared burn-in and loss masks. Maintain state within a segment; detach
only at registered truncation boundaries. Reconstruct a segment's initial state
from a causal prefix using the current parameters, or explicitly register a
streaming-state approximation and its staleness. Do not silently load states
computed using unrelated or future checkpoints. Independent random-position
minibatches do not train memory between moves.

State must be explicit in the Rust/Python/JAX interfaces, not a hidden mutable
singleton. Prediction returns next state. Training exposes the gradient with
respect to initial state. The cached first sparse message currently assumes
the fixed h_initial; it cannot be reused for arbitrary carried states. Keep
the existing reset API backward compatible and qualify both paths.

Before a scientific run, test chunked versus uninterrupted inference, game
reset, pass handling, trajectory augmentation, checkpoint/sampler continuation,
and Rust/JAX states, losses, all gradients and free-running optimizer updates.
For small graphs, check the initial-state gradient independently. Preserve
existing numerical tolerances. CPU qualification precedes an actual TPU gate
at the intended sequence/batch shapes.

## Evidence and fair cost accounting

Measure policy KL, value MSE, calibration and learning curves on the existing
family-separated validation set. Probe local spatial sensitivity and motor
amplitude before interpreting response correlations. For D, compare intact,
reset and deliberately mismatched past states on held-out trajectories. Keep
these diagnostic interventions separate from the primary validation score;
a disrupted state can cause distribution shift as well as remove memory.

Report the loss-bearing labeled exposures and every burn-in/unlabeled prefix
observation, unique games, optimizer steps and total training work. Do not
hide repeated prefix computation by counting only the final labeled board.
Count all K passes per observed ply, opponent updates, reconstruction work and
state copying. Recount actual sparse work because persistence changes activity
and invalidates assumptions about a shared reset-state first pass.

A persistent fly can access more history than a stateless four-frame CNN.
That comparison alone cannot establish an architectural sample-efficiency
advantage. For the memory phase, provide a conventional CNN plus recurrent
state with the same available trajectory, or separately register a common
bounded history accessible to both models. Match prediction FLOPs and labeled
exposures, report tuning/training costs, and confirm promising results with
three seeds before gameplay panels. Final test labels remain closed.

Prior-only evaluation comes first. Later search must branch the hidden state
with the game state, account for each neural update, and avoid merging nodes
solely by board hash when their neural histories differ. This is necessary
before interpreting a persistent network through PUCT or Gumbel search.
