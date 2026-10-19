# Script index

Run Python commands from the `python/` directory with the environment active.

| Area | Entry point |
| --- | --- |
| Main PPO training | `ppo_training/train_unified.py --help` |
| Detailed checkpoint evaluation | `ppo_training/eval_ppo_descriptives.py --help` |
| Evaluation analysis | `ppo_training/analyze_eval.py` |
| Heuristic and population comparisons | `roundrobin_eval.py`, `population_tournament.py`, `simple_roundrobin.py` |
| Trace/debug evaluation | `eval_policy.py`, `eval_with_traces.py` |
| Engine data extraction | `extract_card_db_from_cpp.py`, `extract_wonder_db_from_cpp.py` |
| Fixture and action diagnostics | `capture_fixtures.py`, `fixture_stats.py`, `inspect_fixture_actions.py`, `validate_action_features.py` |
| Earlier trainer implementations | `train_ppo.py`, `train_ppovppo.py` |

The unified trainer is the main development entry point. Earlier trainers are retained as historical implementations, not recommended alternatives to the unresolved advantage-calculation issue. Individual diagnostic scripts may assume local fixtures or checkpoints that are not distributed.

Check the [known issues](../../docs/KNOWN_ISSUES.md) before starting new experiments. Recorded runs and checkpoint files stay outside version control.
