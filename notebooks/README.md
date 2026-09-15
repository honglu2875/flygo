# Research walkthrough

Open [flygo_research_walkthrough.ipynb](flygo_research_walkthrough.ipynb) and run
the cells in order. It reconstructs a small Go sample, instantiates the full Rust
network, explores weights and annotations, plots both spherical eye inputs and
signal propagation, trains all permitted parameters, and computes validation KL.

The defaults use 512 training positions, 256 held-out positions and 64 B32 updates.
Up to eight positions per game provide more opening-family coverage than a few
complete games. This is an exploratory example. Its result is not the V0
full-validation 1.4903 smooth-rate result or a comparison with a CNN. TPU use is
disabled.

The model uses the confirmed spherical baseline: current-board eye inputs,
eight internal passes, and dense learned projections from all 2,129 motors.
Adam clips each parameter group's gradient separately at norm 1. Both policy
and value fully train the circuit. The short default fit finishes during the
100-update learning-rate warmup; increase `UPDATES` to explore a longer fit.
See the [clipping confirmation](../docs/clipping-confirmation-study.md) for the
optimizer evidence, and keep attachment or neuron-rule changes separate.

The checked run includes its actual plots and outputs. It completed in 301 s on
four pinned CPU cores, with policy KL **2.03807 → 1.90477** on its 256-position
validation slice (12 opening families). Value MSE was **.88994 → .90901**.
The 512 training positions cover 55 families. About 2.3 MB of game payloads were
fetched, in addition to the indices. See the
[execution record](../docs/results/research-notebook-hf-v2.json) for exact data,
native source, environment, sampling and schedule identities.

The [earlier execution](../docs/results/research-notebook-hf-v1.json) is preserved
with its older runtime and global clipping (KL 1.90137). The new default follows
the full V0 clipping confirmation; this tiny walkthrough does not establish an
optimizer advantage. Each execution adds 2,048 public-sample training exposures.

Install Python 3.12, Rust, and the package with its notebook extra:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[notebook]'
.venv/bin/python -m ipykernel install --user --name flygo --display-name 'FlyGo (CPU)'
.venv/bin/jupyter lab notebooks/flygo_research_walkthrough.ipynb
```

On the existing research fleet, the notebook environment can instead import the
qualified package under `/dev/shm/flygo/environments/9471407cdd0c68b40179/site-packages`.
Keep its native module and Python package together. The notebook records their
hashes. Run on available pinned CPUs, for example `56–59` for a small exploration;
do not start another numerical-library kernel on an occupied research lane.

The first configuration cell exposes the data, graph and adapter locations.
Go data comes from [quintic/go9x9](https://huggingface.co/datasets/quintic/go9x9),
pinned to commit `4f579b0a21c456f3bb568174d384056351124cd5`. The dataset has
207,138 games and 19,598,695 labeled positions. The notebook downloads about
28 MB of compressed indices and uses HTTP byte ranges for individual NPZ
payloads. It verifies the manifest, indices and selected games; complete tar
shards are not downloaded or extracted. The published opening-family splits
and raw teacher policy/value targets are preserved.

The teacher labels both players' turns. `raw_policy` has 82 probabilities
(81 intersections and pass); `raw_value` is signed winrate from the player to
move's perspective. Played actions are used to reconstruct history. They are
not the policy target, and terminal wins/losses are not the value target.

The existing artifact tree supplies the model assets and optional offline data:

- `releases/v0-1m/manifest.json` and its referenced per-game NPZ records;
- `graphs/183e28b8d990ed5d3ffca091f0468d25dcca3dd1ccd98183e14615a09cda35b4/`,
  with its manifest, canonical arrays and neuron metadata;
- `ports/spherical-context-v1/attachment.npz` and the matching JSON receipt.

These artifacts are not stored in git. On another workstation, obtain the same
prepared artifact bundle and set `FLYGO_ROOT`, or configure the individual paths.
[prepare_graph.py](../scripts/prepare_graph.py) documents how the canonical graph
is reproduced from the MaleCNS source tables. The full graph remains in the demo;
only the Go sample and training horizon are small.

Set `FLYGO_DATA_SOURCE=local` for the original offline V0 release. It is a
separate one-million-position subset, so its scores should not be interchanged
with this Hub sample. `FLYGO_HF_DATASET` and `FLYGO_HF_REVISION` can select
another export with the same indexed-NPZ schema. Arbitrary SGF, parquet or KataGo
training tensors need an explicit target and feature adapter.

The default dataset is public and needs no token. No credentials are embedded
in the notebook.

Optional environment controls for a smaller execution check are
`FLYGO_NOTEBOOK_TRAIN`, `FLYGO_NOTEBOOK_VAL`, and `FLYGO_NOTEBOOK_UPDATES`.
The notebook writes a compact result record under the configured artifact root,
preserves the fleet’s existing RAM buffers, and never replaces a trained checkpoint.
