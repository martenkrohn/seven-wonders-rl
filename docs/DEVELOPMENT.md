# Development guide: corrected training and runtime

Implemented scope: points 1–5 of the restart audit (23 October 2026), followed by the model correction and a neural-only self-play prototype. See the [self-play guide](SELFPLAY_POC.md) for short pilots, holdout comparisons and a moderate next run, [architecture](ARCHITECTURE.md) for contracts and [remaining limitations](KNOWN_ISSUES.md) for model/rule gaps.

Subsequent model correction: new runs use a nonlinear joint policy head (architecture v2). Use a fresh output/league directory; historical linear-head models are available only for explicit legacy inference. Wonder starting resources are already included in production inputs and covered by regressions. Future-stage costs, effects and newer-edition wonder rules remain deferred. The recorded runtime benchmark predates this architecture change.

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

No exact-resume command exists. Do not mistake saved optimizer state for complete serialized environment and RNG state.

## 5. Profile and compare I/O

Self-play now automatically prefers writable shared-memory exchange and groups cached neural opponent requests; see [SELFPLAY_POC.md](SELFPLAY_POC.md). Use an explicit disk `--io-root` and `--opponent-inference serial` when measuring the reference configuration. The historical benchmark below uses fixed bot opponents, so it does not measure the new neural batching path.

Rollouts already live in RAM. Exchange files are the part moved by `--io-root`. Training and both central evaluators use isolated exchange directories; their owned temporary files are cleaned after results/diagnostics are persisted. Use a persistent output path for models.

```sh
.venv/bin/python scripts/benchmark_runtime.py --mode both --repeats 5 --games 20 --total-timesteps 2048 --n-envs 2 --rollout-horizon 64 --io-roots ../io_benchmark /dev/shm/sevenwonders --engine-poll-us 0 1000 --out-dir runs/runtime-comparison
```

This measures four configurations on two storage roots, with and without the C++ wait interval. Warm-up is excluded and configuration order alternates. The transport test runs fixed legal actions without PPO. The training test includes process startup and checkpoint output in external wall time, also reporting learner time and separate timing buckets. Evaluation is deliberately disabled in this benchmark and has its own timings in normal development runs.

Outputs contain filesystem identification, free space, CPU time, session RSS high-water marks, recent timing quantiles, medians, dispersion and paired speed ratios. Parent scopes contain child scopes and must not be added together. Kernel page cache is left intact. No physical SSD-throughput claim follows from file-call counts.

Behavior fingerprints must match within each paired repeat: complete status traces for transport, model weights for training. A mismatch rejects the comparison. Treat failures as failures, not faster games. Reduce `--games` or step budgets for a tool smoke check; use sufficiently long runs for performance decisions.

Use measurements before larger protocol changes. Pipes and direct bindings are not introduced without evidence that file transport remains the relevant bottleneck.
