Offline expert data and architecture study, 2026-09-13.

The first scientific experiment is supervised distillation from a fixed strong
KataGo teacher on a diverse 9×9 state distribution. This comes before sustained
fly self-play RL. Generate data as soon as the Go engine is qualified, while
developing the fly backend independently. Start ablations on a frozen initial
corpus while a larger version is being produced.

The core distinction is between the behavior that reaches a position and the
teacher that labels that position. Weak opponents supply useful mistakes and
recovery positions; their selected moves are not expert policy targets.

**Teacher and opponent selection.**

Interpret the fixed expert as a strong KataGo checkpoint qualified for the
chosen board size. The already cached `kata9x9-b18c384nbt-20231025` is the first
9×9 candidate. KataGo's official [extra-network page](https://katagotraining.org/extra_networks/)
still describes that specialist as one of its strongest 9×9 nets, including
relative to newer nets; it is not suitable for other board sizes. This supports
starting there, not assuming an unconditional current best.

Compare it with a small shortlist of current strong general networks at the
intended rules and search budget. Include playing results, tactical positions,
raw value behavior and CPU throughput. As inspected on 2026-09-13, the
[main network index](https://katagotraining.org/networks/) names
`kata1-tf3-b11c768-s11001M-d5973M` as strongest confidently rated. Those ratings
are not a 9×9 calibration. Recheck availability at implementation time, then
freeze the selected teacher's file hash and settings for the entire corpus
version. A later teacher change creates a new label version.

The expert's model and its search budget are separate choices. First compare
reasonable generation budgets, for example 16/64/128 visits, for quality and
positions per second. A teacher with a strong model and a tiny search budget
is not the same player as that model with extensive search. Pin both.

Select roughly 8–16 historical checkpoints spanning several measured 9×9
strength bands. The old repository already has four pinned early checkpoints
and qualification records that can seed the weak end of this ladder. Extend
with intermediate and strong opponents from official downloads. Calibrate
bands under fixed rules/budgets using nearby opponents and common anchors;
global published Elo or training step count is only a selection aid.

The implemented first corpus uses uniform checkpoint strata: a resident worker
owns one opponent, and frozen releases select equal complete-game counts from
every checkpoint/expert-color pair. The calibrated lower/middle/upper bands
contain 3/3/2 checkpoints, so this is not equal sampling of the three bands.
Band-balanced sampling is a later, separately recorded data experiment.
Randomly assign the
expert Black or White with a 50/50 probability, balanced over work batches.
Record the chosen checkpoint, color, seeds, search budget and exploration
settings. Randomize opening play in a declared manner; neither the teacher
identity nor opponent identity changes halfway through a game.

The qualified first teacher remains `kata9x9-b18c384nbt-20231025`: its small
16-game panel against the current general candidate was 8–8 at 16 visits.
Seven adjacent-opponent panels supplied 112 complete 9×9 games for a coarse
local ordering. This is enough for a screening ladder, not precise ratings.
The exact teacher/opponent hashes and data contract are recorded in
`configs/katago-corpus-v1.json`. Current production uses 16 visits for behavior;
raw teacher labels on the opponent's turns use a one-visit analysis request.

Disable resignation initially so endgames, passing and actual scores are
represented. Record move caps and process failures. Keep complete histories
and explicit truncation status. States from a valid capped prefix can still
receive teacher labels; they have no invented terminal outcome or ownership.
This differs intentionally from terminal-target self-play replay.

Pin 9×9, komi 7.5, positional superko, suicide policy and the qualified
pass-alive area scoring profile. Use full rule objects when interacting with
KataGo, and reject a checkpoint/backend that silently substitutes rules.
The Rust engine remains authoritative for the training environment. Validate
every committed move and board; a tolerant GTP `play` acknowledgment alone
does not establish legality.

**Two reusable data stages.**

Generation emits game histories and available player-analysis records.
Labeling replays each pre-action state and evaluates it with the fixed expert,
including positions where the historical opponent is to play. Neither stage
needs the fly model. Stable game/position IDs permit retries, incremental
labeling and later teacher upgrades without regenerating games.

Use the existing `~/go` infrastructure as a bootstrap: its pinned official
KataGo CPU build, model receipts, GTP transport, parallel match scheduling,
native history replay, scoring audits and dataset validation are useful.
Existing internal-teacher datasets can exercise loading, but are not the new
KataGo corpus and must not be relabeled by name. The old evaluation-only model
receipts describe old experiments; this user's new teacher-data use gets its
own manifest, leaving those historical claims intact.

The existing match runner can produce the first pilot. For bulk generation
and labeling, add a small adapter to KataGo's asynchronous
[analysis engine](https://github.com/lightvector/KataGo/blob/92ee95c0a4b25fec214da00951ab69e97e207729/docs/Analysis_Engine.md),
which accepts concurrent positions and whole-game turn selections. Reuse
long-lived processes by model and NUMA allocation, with bounded outstanding
requests and response IDs. Confirm complete final responses, ordering and
position identity; do not depend on arrival order. Keep full move prefixes
for history-sensitive evaluations. Truncate requests to the requested causal
prefix for the first implementation and qualify any whole-game batching path.

Keep a thin bootstrap boundary to `~/go`: record its source/build identities,
and import its outputs through FlyGo's corpus schema. The FlyGo trainer must
not import the old recipe framework. Migrate the useful producer pieces into
`python/flygo/data` as the small permanent interface is built.

**Labels and value semantics.**

Each valid nonterminal position has a raw teacher value. Label before playing
the target action. Store the played action separately, along with its player.
Use the fixed expert for both current-player colors.

| Field | Meaning | Initial use |
|---|---|---|
| `behavior_action` and actor ID | Actual action and model that played it | History/provenance, not automatically the policy label |
| `teacher_policy_raw` | Strong teacher's neural policy without search | Cheap distillation baseline and diagnostic |
| `teacher_policy_search` | A declared distribution from strong-teacher root search | Main search-distillation policy target where available |
| `teacher_value_raw` | Strong teacher's neural expected win/loss value | Primary offline value target |
| `teacher_value_search` | Strong teacher's root search value | Separate diagnostic or later target variant |
| Teacher ownership/score | Predicted continuation ownership/score | Optional auxiliary labels, clearly identified by kind |
| `game_outcome`, final ownership/score | Actual behavior-game results | Audit and separate experiments, not the primary teacher-value target |
| Label metadata | Model hash, query/config, budget, symmetry, actual work and label-kind mask | Reproducibility and valid comparisons |

KataGo distinguishes raw root predictions from search results. In the pinned
analysis interface, `rootInfo.rawWinrate` is the raw value field,
`rootInfo.winrate` is a search aggregate, and `policy` is the raw prior.
`moveInfos` supplies root-edge statistics. Record the perspective configuration
explicitly. These fields are documented in the
[analysis protocol](https://github.com/lightvector/KataGo/blob/92ee95c0a4b25fec214da00951ab69e97e207729/docs/Analysis_Engine.md).

One unambiguous value convention is to request White-perspective outputs,
then store

\[
v_T(s)=\sigma(s)\left(p_T(\text{White win})-p_T(\text{Black win})\right),
\quad \sigma(s)=+1\text{ for White to move, }-1\text{ for Black}.
\]

For that configured analysis perspective this is
`sigma * (2 * rootInfo.rawWinrate - 1)`: the pinned
[result-construction code](https://github.com/lightvector/KataGo/blob/92ee95c0a4b25fec214da00951ab69e97e207729/cpp/search/searchresults.cpp)
constructs its raw winrate from the win-minus-loss prediction. Retain no-result probability when
available rather than assuming it is identically zero. Independently qualify
the conversion against the explicit White win/loss outputs from
[`kata-raw-nn`](https://github.com/lightvector/KataGo/blob/92ee95c0a4b25fec214da00951ab69e97e207729/docs/GTP_Extensions.md).
The scalar represents the teacher's continuation estimate, not a measured
probability of winning against the particular sampled weak opponent.

In particular, a final win by the expert against a weak opponent need not
validate the teacher's earlier balanced-play value. Do not merge behavior-game
outcomes into the primary value labels. The initial offline value loss is MSE
against `teacher_value_raw`; alternative scalar/probability losses get their
own controlled comparison. Search-value and terminal-outcome targets become
separate declared objectives later.

Define a search policy explicitly, for example
`q(a) ∝ edgeVisits(a)^(1 / temperature)` on legal actions, initially temperature
1. Disable root symmetry pruning for this first labeling mode, request all
searched root moves, and use root-edge rather than transposition-inflated child
counts. Retain counts and the transformation settings. This is our declared
distillation target, not a claim to reproduce every detail of KataGo's own
training-data target construction. Audit the mapping, especially pass,
unvisited moves, symmetry handling and zero-count responses.

Separate cheap and expensive labels. Every stored valid position gets the
teacher's raw value and raw policy. Search labeling can first cover a smaller,
representative subset, then grow. A corpus used for the main search-policy
ablation must have search labels for all of its selected policy-training rows.
Missing search labels carry a mask; there is no silent fallback to a different
target. Reuse generation-time expert analysis only when its label contract
matches. Label the other side independently.

Start a raw-policy control immediately, and qualify a search-policy corpus
with, for example, 64/128 labeling visits. These budgets are pilot candidates.
Reanalyze a stratified audit subset at a larger budget to measure target
stability. Fix the label budget, temperature and loss mix within each
architecture comparison. Changing teacher quality halfway through a run is a
data intervention, not an architecture effect.

**Corpus format.**

Store compact full game histories and immutable position/label shards. A
manifest contains source/config/model identities, game counts, distinct-state
counts, label coverage, split assignments, sizes and file hashes. Position
records retain game/ply identity, compact board or reconstructable prefix,
current player, legal mask and label arrays. Keep scoring/rules/komi and
history semantics recoverable. Materialized model input planes are rebuildable
caches; no final board or future move appears in a pre-action observation.
Opponent/checkpoint IDs and expert-role flags are metadata, not inputs to the
baseline student. Opponent-conditioned prediction would be a separate task.

Use compressed archives for the authoritative corpus and uncompressed typed
arrays for memory-mapped training caches, all under `/dev/shm/flygo`. Initial
shard sizes can be 100k–1M positions, chosen after measuring compression and
replay cost. Keep authoritative histories and manifests when evicting decoded
caches. Both representations occupy RAM; memory mapping shares buffers between
processes but does not make their storage free.

The first current-data research snapshot is `prototype-20260913`: 15,632
complete games and 1,481,078 positions, cloned independently from production.
It preserves the existing family assignments: 1,322,078 train, 98,256 validation
and 60,744 reserved test positions. Its unequal opponent mix is recorded in the
manifest. It enables immediate prototyping while the separate balanced V0
release waits for its slowest stratum; it does not replace that acceptance gate.
The feature cache is versioned, atomically published, hash-verified, and mapped
read-only by the trials. The current loader bound is two million positions.

At 9×9, one FP32 82-action policy costs 328 bytes per position. A single-policy
corpus with compact state/value/metadata is roughly 0.4–0.5 KB per position
before compression: 100 million positions already means roughly 40–50 GB.
A second dense policy adds 32.8 GB at that scale, and FP16 ownership adds
16.2 GB. Histories, replicas, checkpoints and scratch space add more. Measure
real shard bytes; do not extrapolate compression guarantees. Two copies of a
40–50 GB corpus consume 80–100 GB across the hosts before caches and other
artifacts. Fit release sizes to the measured total footprint.

**Primary RAM storage and shared headroom.**

Use `storage_root = "/dev/shm/flygo"` on each host. Read-only inspection on
2026-09-13 found about 200 GiB of free tmpfs capacity per host and about
400 GiB of physical RAM per host. `/dev/shm` shares physical memory with
process heaps; it is not an extra 200 GiB of memory. Each host has its own
mount, shared with other tasks on that host. Accessing the same path on another
host does not access the same files. Recheck live capacity before launching.

```text
/dev/shm/flygo/
  artifacts/                  Pinned graph arrays and teacher/opponent models
  corpus/<version>/           Compact histories, label shards and manifests
  cache/                      Rebuildable decoded arrays and compilation caches
  runs/<run>/                 Resolved config, checkpoints, replay, logs, metrics
  staging/                    Incomplete shard/checkpoint publication and transfer
  tmp/                        Job-owned temporary files
  control/                    Host reservations, leases and artifact locations
```

The launcher directs temporary files and generated build/compilation caches
into this root as well. Source, small configuration files and design/progress
notes stay in the repository. There is no persistent-disk prerequisite.

| Initial per-host limit | Value | Accounting |
|---|---:|---|
| Total FlyGo storage cap | 100 GiB | All FlyGo jobs together, including replicas, caches, logs and staging |
| Minimum free `/dev/shm` | 64 GiB | Live filesystem free space after pending/admitted peak writes |
| Minimum available system RAM | 96 GiB | Live `MemAvailable` after pending/admitted additional storage and compute needs |

The cap permits at most 400 GiB of FlyGo files across four hosts initially;
replication and temporary copies count within that total. These are starting
runtime limits, to be revised from measured usage and other workloads. Respect
any tighter cgroup memory limit as well.

All FlyGo processes on a host share one small reservation ledger and lock.
Admission checks actual filesystem availability, `MemAvailable`, current owned
usage and outstanding reservations. Reserve expected peak additional storage
and heap needs before starting a shard, transfer, training allocation or
checkpoint. Include simultaneous old/new files, compression buffers and
checkpoint staging; count RAM-backed files once in the physical-memory budget.
Reservations are bookkeeping, not large filler files. Reserve checkpoint
headroom before admitting long learning segments, and release reservations
when work completes or a dead worker's owned resources are reconciled.

Monitor headroom during execution. If another task consumes the buffer, stop
admitting work and pause producers at safe boundaries. Evict only FlyGo's
rebuildable caches, abandoned temporary files and checkpoints allowed by the
explicit retention policy. Keep active corpus shards and the last verified
recovery checkpoint; pause if reclaiming expendable files is insufficient.
Never remove another task's files or change global cleanup/mount policy.
A headroom target controls our allocations; it cannot prevent another job
from abruptly consuming memory.

Shard the corpus across hosts and maintain two verified copies of frozen
shards and recent recovery checkpoints on distinct hosts. Replicate manifests
with content hashes and host/path/replica status. Avoid copying the full corpus
onto every host; fetch bounded training caches as needed. A new local artifact
can be marked pending replication, but a frozen release or checkpoint is
advertised as recoverable from single-host loss only after all required peer
copies verify. Keep the last two recovery checkpoints plus explicitly selected
or best checkpoints under declared retention. Do not evict the last verified
pair while its replacement is being published.

Stage and atomically rename complete files on the same mount. This provides
process-interruption consistency, not reboot durability. Surviving peer copies
support restart after one host loses its RAM files; a common cleanup or loss of
all hosts can lose the corpus and checkpoints. This volatility is accepted for
the initial all-RAM workflow. The previous repository's
[storage incident record](../go/research/studies/visual_katago/encoder_interruption_storage_check_20260913.json)
documents missing RAM files on all four hosts with the cause unestablished.
M2 must qualify session/logout survival using a small project-owned sentinel
and the intended job supervisor; M3 must demonstrate restore from a surviving
peer. Do not inherit a root-ownership workaround or alter system policy without
establishing the cause and scope of any observed cleanup.

**Splits and sampling.**

Assign approximately 90/5/5 train/validation/test by game/opening family, with
opponent-band and expert-color coverage. Color-paired continuations and D4
variants of an opening family share a split. Assign groups before generation,
and carry them through labeling, retries and derived input caches. Never split
individual plies independently. Keep selected opponent checkpoints and opening
families outside the training generator for a distribution-shift panel.

Whole-game splitting alone does not remove common opening states. Audit
exact-input overlap and symmetry-equivalent prefixes, and report a separate
novel-state validation subset excluding training duplicates. Do not combine
all games merely because they share the empty board. Deduplicate repeated
complete games and track state multiplicities instead of counting repetitions
as new coverage.

The teacher may use information not present in the fly's finite observation.
Audit disagreements among targets for identical available student inputs,
teacher search variation and symmetry conventions. The old repository has
already measured this kind of target ambiguity; reuse that diagnostic. Keep
the primary labels frozen so it does not become moving noise across ablations.

Weak-opponent games can contain many easy, already-decided states. Inspect
game phase, teacher value, entropy, score margin, opponent band and color.
Retain the natural distribution, but use a declared stratified training sampler
and report both natural and balanced validation metrics. Include substantial
near-even and strong-opponent coverage, plus endgame/pass examples. Sampling
weights are recorded; millions of repeated won positions are not an adequate
coverage goal.

**Use the four hosts from the data stage onward.**

Refresh the four-host inventory at implementation time. The current host has
120 physical cores and two NUMA nodes; all four hosts expose 240 logical CPUs
and about 400 GiB RAM each in the latest inspection. Check physical topology
on the remaining hosts before choosing affinity. Budget process memory and
`/dev/shm` together, using the shared limits above and the capacity left by
existing workloads.

| Phase | CPU use across all four hosts | RAM use | TPU when free |
|---|---|---|---|
| Engine/teacher pilot | Build/qualification on a bounded allocation; generator/labeler workers on the rest | Resident teacher/opponent weights and query caches | Not required |
| Corpus expansion | Independent game and labeling shards; balanced queues on every host | Decoded histories, label buffers and bounded caches | Optional qualified teacher inference |
| Architecture study | Independent Rust learner trials, data preparation and match workers | Shared graph mappings, dataset caches, tapes and optimizer state | JAX student training after qualification |
| Online refinement | Rust search/actors plus learner or evaluation roles | Replay and versioned model replicas | JAX learner and optionally fly inference |

Do not reserve an entire host for a mostly idle coordinator. Assign physical
cores and NUMA memory deliberately, then benchmark worker count, search threads,
neural batch size and library thread pools. The product of analyses and search
threads, plus backend threads, matters more than a single thread-count flag.
Use single-NUMA worker groups where useful and test SMT rather than equating
240 logical CPUs with 240 independent physical cores.

Keep teacher processes warm and group requests by checkpoint. For mixed
opponents, run bounded model-specific queues or schedule checkpoint blocks
without changing the registered sampling distribution. Use coarse transport
of game/position shards; computation stays local. Manage all jobs through a
simple SSH launcher with explicit worker IDs, leases, logs and owned process
groups. Retries must not duplicate published games or label rows.

Measure generated positions/s, teacher-labeled positions/s by label kind,
CPU core-hours, NN evaluations, batch occupancy, queue depth, memory, bytes per
row and shard publication time. Rebalance generation and labeling so neither
creates an unbounded backlog. CPU utilization alone is not the objective:
maximize accepted labeled positions and completed experimental work per hour.

KataGo CPU execution is the initial teacher path. A TPU teacher is a separate
conversion/validation effort. The existing JAX KataGo-related work in `~/go`
does not automatically load the selected specialist's complete policy/value
checkpoint. Any port must match features, policy, value, score conventions and
the exact weights. Initially use a free TPU for the qualified fly student;
only accelerate teacher labeling there after full teacher parity. Do not
replace the strongest teacher with a smaller model merely to fill the TPU.

**Scale in frozen releases, then run controlled architecture studies.**

| Release | Proposed size | Purpose and condition |
|---|---:|---|
| Pilot | About 10k labeled positions | Catch protocol, perspective, history, legality and label errors; measure cost |
| V0 | About 1M labeled positions | First real offline training and sampler/validation qualification |
| V1 | About 10M labeled positions | Main initial architecture screen, after throughput/storage qualification |
| V2 | 100M+ positions if useful | Larger-data confirmation and coverage expansion, with a measured budget |

These are position targets, not claims about generation time. Report complete
game counts, unique states and label coverage separately. Compute a production
estimate from the pilot: for P positions and sustained labeling rate r, label
time alone is P/r; generation, transfer, validation and training also consume
resources. Freeze completed manifests so experiments can start before the
largest release finishes. Append-only production never changes an experiment's
referenced dataset underneath it.

Use one learner update API for offline data and later online replay. Initial
loss is teacher-policy cross entropy plus MSE to the raw teacher value, with
optional separately weighted teacher-ownership/score losses:

\[
\mathcal L(\theta)=\mathbb E_{s\sim D}\left[
-\sum_{a\in\mathrm{legal}(s)}q_T(a\mid s)\log\pi_\theta(a\mid s)
+\lambda_v\left(v_\theta(s)-v_T(s)\right)^2
+\lambda_{\rm aux}\mathcal L_{\rm aux}(s)\right].
\]

The selected corpus defines `q_T` as raw or search policy explicitly. Each
head's masked loss is normalized over its own valid weighted examples.
Report policy KL
as cross entropy minus target entropy. Evaluate value error by game phase and
value band, along with raw/search teacher disagreement and policy quality on
tactical and pass/endgame slices. Calibration against behavior-game outcomes
does not directly assess the teacher's balanced-play value semantics.

For the first sweep, keep graph, data version, target kinds, minibatch draw
schedule, augmentation and adapter budgets fixed. Vary K, sensory mapping,
readout, parameter sharing and one local dynamics extension at a time. Give
each architecture the same small tuning budget for learning rate and essential
stability settings; forcing one unsuitable optimizer setting can misrank them.
Screen cheaply, then replicate promising results across at least three seeds.
Report both matched-data and matched-wall-time results, including CPU
core-hours/TPU chip-hours and all trainable adapter/core parameters.

Validation ranks candidates but does not substitute for playing strength.
Run periodic small panels for screened candidates and a larger confirmation
panel against multiple pinned KataGo checkpoints, with paired colors and fresh
openings. Include opponents outside the generator's pool. Measure both raw
prior play and fixed PUCT/Gumbel assistance; distinguish fixed search budgets
from fixed move-time comparisons. Record caps, timeouts and uncertainty.
Keep headline test positions and game openings out of architecture selection.

The old `state_expert_distillation` study showed that better aggregate value
MSE did not establish improved play, reinforcing the need for both measurements.
After choosing an architecture, use its own search to generate fresh states and
targets, optionally ask the strong teacher to label those states, and then
introduce online self-play refinement. This addresses the distribution shift
between expert-generated boards and boards reached by the learned fly engine.
