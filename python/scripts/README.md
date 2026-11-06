# Supported tools

Run from the repository root using `python/.venv/bin/python python/scripts/NAME`.

| Tool | Purpose |
| --- | --- |
| `train_decision_ppo.py` | Bounded or continuous PPO, own-snapshot league, stop and resume |
| `eval_decision.py` | Independent fixed-panel evaluation, reports and dashboard |
| `build_dashboard.py` | Render existing evaluation or prepare sanitized public demo |
| `open_dashboard.py` | Open latest successful evaluation dashboard |
| `benchmark_decision_performance.py` | Bounded throughput measurement of the current trainer |
| `check_engine.py` | Verify exact submodule/lock/engine contract |
| `export_bga_data.py` | Copy or verify neutral engine-generated resources |

Use `--help` on tools with arguments. [Training](../../TRAINING.md),
[evaluation](../../docs/guides/EVALUATION.md) and
[development](../../docs/guides/DEVELOPMENT.md) describe normal workflows.
Historical trainers, experiment wrappers and legacy exports are retired.
Their source remains in Git history; no protocol-2 runtime is maintained.
