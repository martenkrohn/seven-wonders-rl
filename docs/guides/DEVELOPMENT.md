# Development guide

Current operational guide, reviewed 29 October 2026. See the
[documentation index](../README.md) for research and experiment status,
[architecture](../reference/ARCHITECTURE.md) for implementation contracts and
[known issues](../reference/KNOWN_ISSUES.md) for limitations. Historical experiment
commands are in the archive; they are not a prescribed next training run.

Models support nonlinear actor heads, optional multiple heads and fixed priors.
`--public-information` adds public card/wonder identities and stage information;
the simulator's rule edition is a separate concern. Checkpoint compatibility is
checked explicitly. New phases use new output directories; only a supported
resume continues an existing run.

The trainer enables complete own-hand and observation-history inputs shared by
actor and critic by default (`--no-observation-memory` disables them).
It also enables public information. Checkpoint
migration, technical checks and the distinction from inferred opponent hands are
described in the [memory guide](OBSERVATION_MEMORY.md).

## 1. Build and regression checks

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
cd python
.venv/bin/python -m pytest
```

The existing C++ assert-based tests retain assertions in Release builds. The final-status C++ regression constructs a controlled last turn and verifies the seventh card, age-III victory/defeat tokens, guild copying and skipped actions. Python tests cover mathematical GAE cases, reset/terminal/truncation boundaries, engine failures, final-publication races, replays, RNG isolation and short complete training.

## 2. Short training checks

Run from `python/`. Choose new empty output directories; defaults generate unique names. The following runs hold 128 transitions per update constant and each perform 16 updates:

```sh
.venv/bin/python scripts/ppo_training/train_unified.py --mode fixed --seed 7 --total-timesteps 2048 --n-envs 1 --rollout-horizon 128 --eval-every 1024 --eval-games-per-matchup 2 --out-dir runs/check-one
.venv/bin/python scripts/ppo_training/train_unified.py --mode fixed --seed 7 --total-timesteps 2048 --n-envs 2 --rollout-horizon 64 --eval-every 1024 --eval-games-per-matchup 2 --out-dir runs/check-two
```

These are technical development checks, not benchmarks of playing strength. One- and two-environment trajectories need not match. Repeat exactly the same configuration in another directory to test reproducibility. `updates.jsonl` includes trajectory hashes and numerical diagnostics; checkpoint weights can be compared directly.

Successful completion always saves `latest.pt`. `best.pt` is only updated when every scheduled evaluation attempt completes. Short runs before the first evaluation therefore still produce a loadable model. Existing experiments are never overwritten. `--save-every` controls periodic saving; the first update and successful completion are also saved.

## 3. Separate evaluation

```sh
.venv/bin/python scripts/ppo_training/eval_ppo_descriptives.py runs/check-two/latest.pt --games 20 --seed 2000 --action-mode greedy --out-dir runs/check-two-evaluation
.venv/bin/python scripts/ppo_training/analyze_eval.py runs/check-two/eval_2048.jsonl
```

The detailed evaluator writes `games.jsonl`, `actions.jsonl`, `summary.json` and a manifest. Its default is greedy; use `--action-mode stochastic` to match the unified trainer's current default. Both modes use explicit evaluation configuration, and stochastic actions use local seeded RNGs. Every attempt is retained; no failed seed is silently replaced. `completed + truncated + failed = attempted`. Only completed games contribute VP, win and tie metrics. No completed games means unavailable metrics, not zero performance. The neural self-play holdout tool is `scripts/eval_selfplay.py`, described in the separate prototype guide.

Missing optional wonder gains are documented in records. Action counters use applied actions, not ignored seventh-turn commands. Historical evaluations lacking completion markers are reported as unverified by the analyzer. Historical checkpoint inference requires `--allow-legacy`; no such override is needed or recommended for new runs.

## 4. Reproducibility and failures

The engine accepts `--seed UINT64`, `--poll-us 0..1000000` and `--quiet`. The wrapper passes each episode's effective seed. The manifest records the role-based seed derivation, source/engine fingerprints, versions, threads and observation semantics. Stable tie resolution in the trade estimator prevents Python hash randomization from altering features.

A failed exchange cannot supply a PPO sample. The trainer writes `failure.json`, last statuses and engine stderr to persistent output, closes processes and discards the pending rollout. It preserves already saved checkpoints and exits with failure. Unexpected programming exceptions propagate rather than being silently counted as ordinary engine failures.

There are three distinct ways to use previous work:

- `--init-checkpoint PATH`: initialize a new training phase from compatible
  weights, with a fresh optimizer and counters.
- `--resume`: continue a supported milestone run at complete PPO updates,
  restoring optimizer and local RNG states. Incomplete engine games are recorded
  as abandoned and replaced; this is not bit-exact game-state continuation.
- `--restart-incomplete`: archive a matching incomplete output and start again.
  It does not continue its learned weights.

Use the corresponding wrapper's `--help` and the saved configuration to determine
whether its resume mode is supported. Restart/skip and resume modes cannot be
combined freely. The long-run wrappers and their original configurations are
documented in the [archive](../archive/README.md).

## 5. Profile and compare I/O

Self-play now automatically prefers writable shared-memory exchange and groups cached neural opponent requests; see [SELFPLAY_POC.md](../archive/selfplay/SELFPLAY_POC.md). Use an explicit disk `--io-root` and `--opponent-inference serial` when measuring the reference configuration. The historical benchmark below uses fixed bot opponents, so it does not measure the new neural batching path.

Rollouts already live in RAM. Exchange files are the part moved by `--io-root`. Training and both central evaluators use isolated exchange directories; their owned temporary files are cleaned after results/diagnostics are persisted. Use a persistent output path for models.

```sh
.venv/bin/python scripts/benchmark_runtime.py --mode both --repeats 5 --games 20 --total-timesteps 2048 --n-envs 2 --rollout-horizon 64 --io-roots ../io_benchmark /dev/shm/sevenwonders --engine-poll-us 0 1000 --out-dir runs/runtime-comparison
```

This measures four configurations on two storage roots, with and without the C++ wait interval. Warm-up is excluded and configuration order alternates. The transport test runs fixed legal actions without PPO. The training test includes process startup and checkpoint output in external wall time, also reporting learner time and separate timing buckets. Evaluation is deliberately disabled in this benchmark and has its own timings in normal development runs.

Outputs contain filesystem identification, free space, CPU time, session RSS high-water marks, recent timing quantiles, medians, dispersion and paired speed ratios. Parent scopes contain child scopes and must not be added together. Kernel page cache is left intact. No physical SSD-throughput claim follows from file-call counts.

Behavior fingerprints must match within each paired repeat: complete status traces for transport, model weights for training. A mismatch rejects the comparison. Treat failures as failures, not faster games. Reduce `--games` or step budgets for a tool smoke check; use sufficiently long runs for performance decisions.

Use measurements before larger protocol changes. Pipes and direct bindings are not introduced without evidence that file transport remains the relevant bottleneck.

## Runtime optimizations, 25 October 2026

The shared observation context lasts for exactly one parsed engine status. It
prepares public player data and the table once, then builds/caches the three seat
views lazily. It never adds cache data to protocol JSON. The feature schema,
relative seat order and values of illegal/empty action rows remain unchanged.
Critic-only bootstrap retains the original batch shape for mixed terminal batches
to preserve CPU arithmetic; all-terminal batches skip the critic entirely.

The training pool uses explicit registration after an atomic snapshot save.
Other pool users retain automatic external-directory discovery; `refresh()` is
available for explicit updates. Inference snapshots omit optimizer state, while
`latest.pt` retains it. Snapshot storage alone is not sufficient for training
resume; the supported milestone-resume mechanism is described above.

The `--throughput-preset` option and evaluation worker flags are described in the
[script index](../../python/scripts/README.md#runtime-options-25-october-2026).
`benchmark_optimizations.py` compares frozen before/after trees sequentially,
keeping 512 samples per PPO update for both tested environment counts. Compare
matching environment configurations for semantic equivalence; eight versus two
environments are different training trajectories.
