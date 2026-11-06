# Training

Run these commands from the repository root after [setup](README.md#setup).
They use the same run directory as the VS Code tasks in `python/.vscode/tasks.json`.

```sh
python/.venv/bin/python -u python/scripts/train_decision_ppo.py \
  --continuous --auto-resume --workers 4 --seed 331 \
  --engine build/7Wonders \
  --output-dir python/runs/continuous-decision-pfsp-v1-seed331
```

A saved `checkpoint.pt` is resumed; otherwise a fresh learner starts. Four CPU
games feed one PPO learner. Own frozen snapshots form the opponent league;
seats rotate. A short fixed-panel evaluation runs every 5,000 completed games.
Terminal output is English. All learning counters count confirmed decisions or
completed games; evaluation games are recorded separately.

To stop cleanly, use a second terminal from the same directory:

```sh
python/.venv/bin/python python/scripts/train_decision_ppo.py --stop \
  --output-dir python/runs/continuous-decision-pfsp-v1-seed331
```

Wait for **CLEANLY STOPPED**. Ctrl+C and SIGTERM request the same boundary stop.
The current PPO update or intermediate evaluation finishes before saving.
Start with the first command again to continue. A hard interruption loses work
after the last saved update; unfinished games are abandoned without being scored.
Resume restores weights, optimizer, Torch RNG, opponent sampler and counters,
then starts fresh games with newly reserved seeds. It does not restore midgame state.

For a small technical run in a new directory:

```sh
python/.venv/bin/python python/scripts/train_decision_ppo.py \
  --steps 128 --workers 4 --output-dir python/runs/smoke
```

Use `--help` for rollout size, update epochs, snapshot interval and league settings.
The normal CLI default is four workers; resume inherits the saved width unless
explicitly overridden. One worker uses the same trainer with in-process collection.
Changing worker count changes the per-game rollout horizon and can affect learning.
Checkpoint rules, feature schema and league settings must remain compatible.

VS Code: open `seven-wonders.code-workspace`, then **Terminal → Run Task**.
Start/resume, stop, evaluate and open-dashboard tasks use the same current run.

`metrics.json` contains saved training configuration and provenance.
`training_status.json` reports the last saved state; the directory lock, not a stale
`running` string, determines whether a trainer is actually active. Never remove
league snapshot files while a checkpoint or evaluation still references them.
See [architecture](docs/reference/ARCHITECTURE.md) and
[evaluation](docs/guides/EVALUATION.md) for details.
