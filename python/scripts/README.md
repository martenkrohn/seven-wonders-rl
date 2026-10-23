# Script index

Run Python commands from the `python/` directory with the environment active.

| Area | Entry point |
| --- | --- |
| Main PPO training | `ppo_training/train_unified.py --help` |
| Neural self-play holdout | `eval_selfplay.py --help` |
| Exact simulator change inventory | `audit_simulator.py --check` |
| Detailed checkpoint evaluation | `ppo_training/eval_ppo_descriptives.py --help` |
| Evaluation analysis | `ppo_training/analyze_eval.py` |
| Heuristic and population comparisons | `roundrobin_eval.py`, `population_tournament.py`, `simple_roundrobin.py` |
| Trace/debug evaluation | `eval_policy.py`, `eval_with_traces.py` |
| Engine data extraction | `extract_card_db_from_cpp.py`, `extract_wonder_db_from_cpp.py` |
| Fixture and action diagnostics | `capture_fixtures.py`, `fixture_stats.py`, `inspect_fixture_actions.py`, `validate_action_features.py` |
| Paired runtime benchmark | `benchmark_runtime.py --help` |
| Historical, disabled trainers | `train_ppo.py`, `train_ppovppo.py` |

The unified trainer is the main development entry point. Earlier trainer source is retained for history; its executable entry points are disabled because they contain the old GAE and evaluation paths. Individual diagnostic scripts may assume local fixtures or checkpoints that are not distributed.

See the [development guide](../../docs/DEVELOPMENT.md) for short training/evaluation commands and the paired runtime benchmark. New output directories must be empty; generated defaults are unique.

Check the [known issues](../../docs/KNOWN_ISSUES.md) before starting new experiments. Recorded runs and checkpoint files stay outside version control.

## Neural self-play prototype

`ppo_training/train_unified.py --mode selfplay` trains from random neural weights
against frozen checkpoints from the same fresh run, with no bot bootstrap.
`eval_selfplay.py` evaluates a final model on separate seeds against fixed neural
references or chosen trained checkpoints, paired with its untrained baseline.
See [the experiment guide](../../docs/SELFPLAY_POC.md).

`audit_simulator.py` regenerates the exact C++ comparison against the pinned
upstream commit; `--check` detects simulator files changed without updating the
record. Every C++ correction also requires an entry in
[the simulator change log](../../docs/SIMULATOR_CHANGES.md).
