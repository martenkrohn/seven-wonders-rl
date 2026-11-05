# Environment and training contracts

Reviewed 29 October 2026. This reference describes supported modes; defaults
and the configuration of a particular experiment are distinct. For the latest
completed setup and its scientific interpretation, see the
[research overview](../research/OVERVIEW.md). Historical protocol links below
document implementation provenance, not instructions for a new experiment.

## Protocol v2

Each environment has its own working directory ending in `io/`. C++ publishes JSON using a temporary file and atomic rename; Python publishes all three player commands before the ready signal. Commands carry the expected turn. Status includes `protocol_version=2`, `game.engine_seed`, `game.phase` and `last_actions` acknowledgements (`applied`, `skipped`, `fallback_discard`). The wrapper requires a matching seed and exactly the next turn. It re-reads once after observing process exit to cover final publication between read and poll.

Statuses for turns 0 through 20 are action requests. The final status is turn 21, era 4, phase `finished`, after the last action, final battles and all guild copies. Terminal masks are empty. No further policy selection or value bootstrap is performed for a terminal state. There are 21 engine exchanges, generally 18 applied actions per ordinary player; seventh-card abilities can add actions.

Timeouts and engine/protocol failures are never fabricated transitions. The wrapper reports a cached diagnostic observation with `transition_valid=false` for failed waits; other engine errors raise `EngineError`. The trainer rejects the pending rollout, records diagnostics and closes every engine. It never assigns a terminal reward to a timeout. Startup failures close the process and stderr handle. Timeouts use monotonic clocks.

## Observations and GAE

Observation-memory architectures 4 (single head) and 5 (multiple heads),
enabled by default in the trainer (`--no-observation-memory` opts out),
add a seat-local explicit history and complete current own-hand information.
The packed input also contains current legal-action features/masks; one shared
information encoder augments the context used by actor and critic. New residual
outputs start at zero when migrating pretrained weights, preserving predictions
on identical inputs and keeping frozen prior inputs unchanged. Memory has a
separate schema hash. See the [memory guide](../guides/OBSERVATION_MEMORY.md).

Each environment/seat owns its history; cached checkpoint policies have no
mutable history. Observation payloads are immutable copies. Own action fields
are filled after engine acknowledgement, never retroactively in stored PPO
inputs. Invalid history slots are masked. Unplayable seventh-card exchanges
do not reveal a new private hand. PPO continues to use its existing rollout
and minibatch layout because each packed observation includes its full history.

Action indices remain `action_type * 7 + hand_slot`, with 28 slots. Global observations use relative seats `[self, left, right]`. Feature schema names/order and a separate semantic version are checked when loading a checkpoint. Era, round, coins, shields and selected-seat hand size use actual engine fields. Greedy choice-resource ties use a stable resource order rather than Python set iteration.

The base 102-dimensional global vector includes `table_prod` in resource order `[wood, ore, clay, stone, loom, glass, papyrus]` for all three relative seats. The engine adds each wonder's starting resource before the initial status, so it is already visible through production totals. Science symbols are derived from built cards for both policy and critic; separate science jokers and held trading discounts are included. See [feature semantics v3](../archive/selfplay/FEATURE_CORRECTIONS_2026-10-23.md). Optional public-state schema v1 appends public played-card identities and all three wonder identities/stage plans, yielding 645 global and 121 action inputs. Zero-initialized new encoder columns preserve warm-start behaviour; old priors retain their original raw input prefixes. Database/schema hashes protect this extension. The simulator edition is unchanged; see [the long-run setup](../archive/long-training/LONG_TRAINING_RUN_V1.md).

ActorCritic architecture v2 combines action and global embeddings through `Linear(2H,H) -> ReLU -> Linear(H,1)`. This permits state-dependent action preferences; a single linear head would cancel the shared global contribution in softmax. The value branch remains unchanged. Model architecture versions are recorded separately from feature semantics in the run manifest, configuration and checkpoints. Loading an old linear-head checkpoint requires `--allow-legacy` and reconstructs architecture v1 for historical inference, preserving its predictions; it is not a conversion or training resume.

Rollouts retain `[T,N,...]` axes. Every transition stores pre-reset next values; terminal next values are zero. GAE runs separately in each environment. Termination cuts bootstrap and trace; a valid external truncation bootstraps from the real final observation and cuts only the trace. Transport failures are rejected before GAE. Returns use raw advantages; only the policy advantages are normalized, using population standard deviation to handle one sample. Flattening happens immediately before PPO minibatches.

## Randomness and artifacts

Optional architecture v3 adds multiple actor/critic pairs over the existing
shared encoders. A per-environment head ID is selected once per episode, stored
in the rollout, and used for both PPO likelihoods and pre-reset value bootstrap.
Optional random prior networks are fully frozen and read raw features; their
outputs are added to actor logits before masking. Cached opponent models hold
no episode state: head selection belongs to each environment/seat. Version-2
checkpoints remain supported and can explicitly initialize all heads. Training
version 4 adds fixed weighted opponent groups and per-head evaluation. See
[the first multi-head experiment](../archive/exploration/MULTIHEAD_EXPLORATION.md) for its configuration.

`derive_seed(master, role, indices...)` uses NumPy SeedSequence with a versioned role map. Engine games use a local `mt19937_64`; the learner uses an explicit Torch generator; minibatches, opponents and evaluation have separate NumPy streams. Evaluation and checkpoint loading preserve the learner's global Torch RNG. Reproducibility is scoped to the same software/toolchain and CPU configuration.

The run manifest records source revision/dirty status, engine hash, dependency versions, seed version, feature semantics and configuration. Outputs are unique by default and an explicit nonempty output directory is rejected. Checkpoints are atomically replaced and include dimensions, hidden size, semantics and model/optimizer weights. Successful training always saves `latest.pt`. `evaluated.pt` stores the last fully evaluated self-play learner; `best.pt` is used only by fixed-bot mode. Milestone runs support explicit `--resume` of complete PPO updates, optimizer and local RNG states; unfinished engine games are recorded as abandoned and replaced with new episode seeds. This is not bit-exact engine continuation.

## Evaluation and timing

`--pair-pfsp` enables a separate own-snapshot league with joint selection of
ordered left/right opponent pairs. Discounted terminal training win shares,
neutral pseudo-counts and a uniform control component determine probabilities.
Only completed training games update the estimates. Learner and opponent heads
are marginalized as checkpoint-level mixtures. Durable checkpoints include the
pair statistics; resume restores them together with RNGs and abandons unfinished
games. `league_status.json` exposes the selection distribution. Separate external
evaluation populations can be used for milestones without supplying any fixed
training opponents. See the [fresh-run guide](../guides/PAIR_PFSP_TRAINING.md).

`--continuous` extends pair-PFSP milestone training to an unbounded budget and
an absolute `--eval-every` schedule. `run_fresh_pair_pfsp.py --continuous
--auto-resume` selects a dedicated run directory, 512-step saves and a
262,144-step evaluation interval. The stop task publishes a request checked
between complete PPO updates. Hard interruptions recover the last durable
update; interrupted reports are re-run, including the initial evaluation.
Only a truncated final JSONL append is repaired automatically. The main
continuous dashboard keeps the initial and twelve latest detailed reports,
while retaining the full curve and individual reports on disk. See the
[operating guide](../guides/CONTINUOUS_TRAINING.md).

Training version 5 optionally combines fixed opponent groups with a lagged recent
snapshot group and temporal archive, keeping group weights independent of member
counts. Its milestone mode evaluates one immutable reference panel at
0/25/50/75/100%, saves detailed reports per checkpoint and atomically refreshes a
combined dashboard during training. Evaluation uses separate RNG streams and
pauses collection. Legacy rolling evaluations remain the default outside this mode.

Training version 3 implements neural-only self-play. A fresh `checkpoint_0.pt` is saved before any game, and the initial snapshot remains eligible even with positive pool lag. No minimum pool size or heuristic/random fallback exists. Each completed game's opponents are fixed snapshots; start/completion logs carry their paths and SHA-256 hashes, the learner seat and engine seed. Learner seats are distributed across environments and episodes. Loading errors stop the run with the opponent path in diagnostics.

Self-play checkpoint evaluation refreshes its trained references at each measurement: the latest strictly older snapshot plus another near half the current step. Step zero and current/future snapshots are excluded; no pre-training evaluation runs. Every summary stores reference hashes/steps and a fixed-plan identity. Cross-checkpoint deltas are emitted only for matching plans. The last successful evaluation writes `evaluated.pt`; scores against changing opponents do not select a global best. Initial references remain available for optional separate baseline tests. Separate `scripts/eval_selfplay.py` holdout runs compare stages on an unchanged trained reference plan. The VP-margin reward and training opponent pool are unchanged.

Both central evaluators use `evaluation.runner.run_game`. Every attempt has completed/truncated/failed status; partial scores are null. Strict VP wins and VP ties are separate. Summary metrics only use completed games and report their denominator and failure counts. Empty completed sets yield null metrics. Greedy and seeded stochastic evaluation are explicit modes. Action records distinguish requested and applied/skipped actions; missing optional features never alias another column.

Persistent output is separate from exchange files. Training, fixed evaluation and detailed evaluation accept the I/O root and use isolated descendants. Owned exchange directories are removed after diagnostic/result persistence. Timing reports label overlapping inclusive scopes, use real wall time as denominator and report means and recent quantiles. The paired benchmark measures transport separately from full training and compares behavior fingerprints.

Self-play training and its holdout CLI automatically attempt RAM exchange under
`/dev/shm/sevenwonders`; missing/unwritable shared memory falls back to disk.
Explicit I/O roots propagate failures instead of changing storage silently.
Checkpoints, episode records and diagnostics remain persistent. The ordinary
rolling evaluation interval defaults to 32,768 transitions and waits for a
trained predecessor; it has no step-zero evaluation. Milestone mode instead
includes a step-zero baseline and fixed progress checkpoints.

Frozen neural opponent requests are grouped by cached policy identity across collector environments. Every observation is built from that environment's pre-turn status and each decision draws from its original seat-local NumPy RNG. The group performs one policy forward pass, then validated action overrides are injected into `send_action()` without resampling. Distinct policy objects are not combined; ordinary policies and legacy opponent samplers use the original serial selection. Standalone game steps can group the two opponents within one game. `--opponent-inference serial` disables grouping in training and evaluation for comparisons. Timing separates opponent preparation, inference groups and number of requests. Learner inference, PPO/GAE, reward and model architecture are unchanged.

## Runtime extensions, 25 October 2026

An environment owns one `ObservationContext` for its latest parsed status. Public
player data and absolute table features are prepared once; the learner and two
opponents lazily request their relative seat views. A new status replaces the
context. Protocol JSON and feature semantics remain unchanged.

Bootstrap uses the critic-only `forward_values()` path without computing actor
logits. Memory models also evaluate their shared information encoder, including
the action information already packed in the observation. Entirely terminal batches return zeros without a
forward pass. Mixed batches keep their original shape and mask terminal outputs
to zero, preserving CPU arithmetic for the nonterminal rows. External truncation
still bootstraps from the pre-reset final observation.

The unified trainer indexes its own snapshot publications and automatically sizes
the frozen policy cache to the eligible pool plus active seats. Inference snapshots
omit optimizer state; regular `latest.pt` checkpoints retain it. Other pool users
can keep automatic scanning or explicitly refresh an index.

Detailed CPU evaluation optionally uses independent spawned processes with one
Torch thread per worker. Each game keeps its original seed, seat and ID; output
is sorted back into plan order. Completed attempts survive another worker's
exception. Self-play and green-priority CLIs default to two workers, and in-training
evaluation stays serial unless `--eval-workers` is set. The throughput preset is
opt-in because eight environments with horizon 64 change GAE segments relative
to the unchanged default of two environments with horizon 256.

## Specialist training objectives and pretrained initialization

`RewardSpec` version 1 selects standard terminal VP margin or adds weighted final
science/net military points before applying the common reward scale. Defaults
preserve standard behavior; all nonterminal transitions remain zero and invalid
transitions never earn a bonus. Completed episode records carry separate reward
components. Configurations, manifests and checkpoints retain the reward contract.
The evaluators create standard-reward environments regardless of checkpoint
training objective and report actual scores and wins.

`--init-checkpoint` performs strict feature/architecture-compatible weight loading
(including the critic), uses the source hidden size and starts a fresh optimizer,
step counter, seed streams and league. Checkpoint provenance includes source path,
SHA-256, source step and training objective. It is weight initialization, not exact
resume. The step-zero snapshot is the pretrained starting policy; holdout labels
and dashboards identify it accordingly. Existing random-initialized experiments
retain their original default behavior. See [the specialist plan](../archive/specialists/SPECIALIST_REWARDS.md).

`--frozen-opponents PATH [PATH ...]` optionally replaces the neural growing league
with an explicit immutable set. Checkpoints are copied into `frozen_opponents/`,
compatibility-checked and recorded with source hashes. Seats sample independently
and uniformly from that set, with normal seat-local action RNGs. Learner snapshots
are still saved for analysis, but are never registered in the frozen pool.
In-training evaluation uses the same fixed neural opponents and separate evaluation
seeds, allowing paired progress on an unchanged plan. Specialist v2 commands use
one normal trained model for both opponent seats and other normal checkpoints for
the separate holdout.
