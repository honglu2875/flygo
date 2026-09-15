# Explicit recurrent state

The research API can carry all neuron states across calls and differentiate
through their boundaries. It is an engineering interface; a trained persistent
Go model and an actual full-CNS numerical qualification remain pending.
The existing `infer`, `loss_and_grad` and `train_step` continue to reset for
each prediction. The registered optimization trials use their frozen runtime.

The [CPU fixture gate passes](results/recurrent-state-interface-v1.json):
56 Rust and 186 Python tests, including independent dense finite differences,
two-ply JAX gradients and three continued updates. Twelve old/new reset
protocols match all 1,602 checked arrays exactly. The initial test bundle
omitted a report fixture and PyArrow; its failure and the successful corrected
bundle are retained. The native binary and numerical source did not change
between those test attempts. Both evidence archives have verified peer copies.

## Prediction

```python
state = model.initial_state(batch_size)       # float32 [neurons, batch], 0.01
out = model.infer_state(features, state)      # features remain batch-major
policy_logits, value = out['logits'], out['value']
state = out['next_state']                    # complete state for the next ply
```

State belongs to the caller. Interleaved games require separate states; reset
at each new game and advance on every ply, including opponent moves and passes.
`trace=True` additionally returns the initial state and all internal passes.
Streaming inference retains only a boundary state. It always evaluates the
full graph: readout pruning would discard neurons needed by a later move.
Learned weight transforms remain cached, but the cached first message for the
constant reset state is bypassed.

For a memory study, both reset and carry arms must use the same absolute
black/white input frame. Today's own/opponent channels change perspective
every ply, including passes. The [temporal study contract](temporal-vision-study.md)
defines the separate adapter and causal replay gates before such a study.

## Gradients across chunks

Write one chunk as
`(logits_t, value_t, h_next) = F_theta(features_t, h_t)`.
`state_vjp` evaluates the vector-Jacobian product with three caller-supplied
cotangents and returns `(parameter_gradients, d_initial_state)`:

```python
# Keep the forward inputs and initial states for a fixed-parameter window.
# dlogits/dvalue already include the loss reduction over that entire window.
gradients = {name: np.zeros_like(p) for name, p in model.parameters().items()}
dh = None
for features, initial, out, dlogits, dvalue in reversed(window):
    grad, dh = model.state_vjp(
        features, initial, revision=out['revision'],
        dlogits=dlogits, dvalue=dvalue, dstate=dh,
    )
    for name in gradients:
        gradients[name] += grad[name]
model.apply_gradients(gradients, revision=window[0][2]['revision'],
                      rate=0.003, clip=1.0, epsilon=1e-6)
```

This is a literal VJP, like `embedding_loss_and_grad`. The model's supervised
`value_core_scale` does not alter it. The final-state cotangent is added after
the heads; it is neither rescaled nor divided by batch size. Omitting it at a
window boundary explicitly truncates the gradient. No optimizer update occurs
inside a VJP. The separate update applies the configured clipping mode once,
after accumulating the window gradient, and reports clipping statistics.

Native code recomputes each chunk's tape during backward to avoid exporting a
full graph tape to Python. Features and initial states must remain unchanged
between forward and backward. The returned revision rejects a VJP or update
after intervening parameter updates or checkpoint restores. Caller-owned
boundary states are not checkpointed automatically. Training must reconstruct
a causal prefix at current parameters, or separately register and test a state
staleness approximation. This API does not implement a sequence sampler.

The independent JAX `forward(..., initial_state=h)` accepts the same node-major
state. Its returned `states[-1]` can feed another call; JAX autodiff then
propagates through that boundary. JAX inputs must be finite, and the new path
has only a CPU fixture gate at this stage. Existing TPU learner entry points
have not been changed.

Implementation lives in `crates/fly-core/src/model.rs`, the small native
`crates/flygo-python/src/fly/explicit_state.rs` boundary, and `RustFly` in
`python/flygo/fly.py`. The [engineering contract](../configs/recurrent-state-interface-engineering-v1.json)
records independent finite-difference, JAX, recovery and update checks.
