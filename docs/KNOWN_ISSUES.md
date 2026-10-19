# Known issues and development priorities

The environment and regression tests are useful foundations, but they do not validate training quality. These are current technical limitations, not benchmark results.

## Multi-environment advantage calculation

In `python/scripts/ppo_training/train_unified.py`, rollout entries are appended by step and environment, then processed as one flat trajectory. The GAE loop uses `value_buf[t + 1]` and one shared `last_adv`, so adjacent entries can belong to different games. The rollout-end bootstrap also does not correctly preserve every environment's continuation. The older trainers use a related approach.

Before using substantial new training runs, preserve the environment axis through reward, value, terminal and bootstrap buffers, then compute advantages independently per environment. An analytical regression test should verify that changing one game's rewards cannot change another game's advantages, including staggered episode endings.

## Evaluation and reproducibility

- The fixed evaluator treats a truncated rollout as finished and can report an intermediate score as a result. Completed games, truncations and failures should be counted separately.
- The C++ engine uses a wall-clock random seed. Python seeds alone therefore do not reproduce identical engine games. A seedable engine entry point is needed for controlled comparisons.
- Feature schemas evolve. Before loading old checkpoints, compare their schema and feature ordering with the current code; matching vector dimensions alone is insufficient.
- Current tests cover selected simulator behavior and interfaces, not every board-game rule. The wrapper does not implement every possible extra-choice interaction.

## Throughput and opponents

The engine transport writes and polls files every turn and starts a new process for a game. Rollout buffers already live in memory. `--io-root` can place training exchange directories on a RAM filesystem; it does not replace the protocol, and the evaluation paths do not all share this setting. Profile the complete loop before choosing between filesystem changes, pipes or a direct binding.

The heuristic opponents are intentionally limited. A high win rate against them is not a general measure of playing strength. Self-play needs comparison against a fixed reference population, seat rotation and separately reported failures. The opponent pool initially falls back to heuristics until eligible snapshots exist.

## Development order

1. Correct and analytically test GAE and bootstrap handling.
2. Separate incomplete games in evaluation and add engine seed control.
3. Establish short, repeatable checks before longer training.
4. Profile transport and improve the opponent/evaluation population.

No long-run command or performance target is promised here. The current unified trainer uses CPU and has no exact-resume option.
