# Script index

The active BGA entry point is `train_decision_ppo.py`; neutral data are checked
with `export_bga_data.py --check`. See the [BGA guide](../../docs/guides/BGA_RL.md)
for protocol 3, offered actions and incompatible checkpoint semantics. The
research trainers and results below use the explicitly retained legacy API.

The [documentation index](../../docs/README.md) is the entry point for current
research, operating guides and technical limitations. The experiment-specific
sections below describe tools and historical wrappers, not a prescribed next
training run. Original protocols and results are grouped in the
[exploratory archive](../../docs/archive/README.md); new focused experiments use
the [protocol structure](../../docs/experiments/README.md).

Run Python commands from the `python/` directory with the environment active.

`ppo_training/train_unified.py` enables by default the explicit
observation history and complete own-hand information shared by actor and critic.
It automatically enables public inputs; old checkpoints can initialize a new
phase with a neutral residual branch. See the
[memory guide](../../docs/guides/OBSERVATION_MEMORY.md) for compatibility and a
short technical check. Use `--no-observation-memory` for comparisons or resuming
old runs without memory. This setting does not select a new opponent curriculum.

`ppo_training/run_fresh_pair_pfsp.py` starts a new random learner and its own
frozen-snapshot league with measured ordered-pair matchmaking. Old long-run
models are evaluation-only. The recipe supports `--dry-run` and `--resume`;
see the [start guide](../../docs/guides/PAIR_PFSP_TRAINING.md) and
[protocol](../../docs/experiments/001-fresh-pair-pfsp/PROTOCOL.md).

`ppo_training/run_fresh_pair_pfsp.py --continuous --auto-resume` runs without
an end budget, evaluates a fixed panel every 262,144 transitions and saves
every 512 transitions. `--continuous --stop` requests a clean stop.
The [continuous training guide](../../docs/guides/CONTINUOUS_TRAINING.md)
describes the three prepared VS Code tasks and crash recovery.

`open_dashboard.py` now uses `configs/dashboard.json`: one pinned final-model
evaluation plus the configured new training run. Historical experiments are
not auto-discovered. New prepared runs evaluate the fixed model once on their
current panel for paired comparisons; see the
[dashboard selection guide](../../docs/guides/EVALUATION.md#bereinigte-dashboard-auswahl).

| Area | Entry point |
| --- | --- |
| Main PPO training | `ppo_training/train_unified.py --help` |
| Science trajectory diagnostics | `ppo_training/analyze_science.py --help` |
| Print science experiment commands (does not train) | `ppo_training/science_experiment_commands.py --help` |
| Neural self-play holdout | `eval_selfplay.py --help` |
| Same model with/without forced green priority | `eval_green_priority.py --help` |
| Exact simulator change inventory | `audit_simulator.py --check` |
| Local visual comparison of cards and wonder editions | `build_edition_comparison.py --help`; [data, sources and image slots](../../docs/reference/edition-comparison/README.md) |
| Detailed checkpoint evaluation | `ppo_training/eval_ppo_descriptives.py --help` |
| Evaluation analysis | `ppo_training/analyze_eval.py` |
| Heuristic and population comparisons | `roundrobin_eval.py`, `population_tournament.py`, `simple_roundrobin.py` |
| Trace/debug evaluation | `eval_policy.py`, `eval_with_traces.py` |
| Engine data extraction | `extract_card_db_from_cpp.py --check`, `extract_wonder_db_from_cpp.py --check`; [legacy export workflow](../../docs/guides/LEGACY_DATA_EXPORT.md) |
| Fixture and action diagnostics | `capture_fixtures.py`, `fixture_stats.py`, `inspect_fixture_actions.py`, `validate_action_features.py` |
| Paired runtime benchmark | `benchmark_runtime.py --help` |
| Before/after self-play, evaluation and feature benchmark | `benchmark_optimizations.py --help` |
| Historical, disabled trainers | `train_ppo.py`, `train_ppovppo.py` |

The unified trainer is the retained legacy research entry point. Earlier trainer source is retained for history; its executable entry points are disabled because they contain the old GAE and evaluation paths. Individual diagnostic scripts may assume local fixtures or checkpoints that are not distributed.

To retry an interrupted run, add `--restart-incomplete --skip-completed` to the
same training command. Incomplete matching output is archived under `runs/archive/`
before a fresh start; matching finished runs are skipped. Generated science
experiment commands include these flags. Re-generate older shell scripts before
retrying. This does not resume from a checkpoint.

See the [development guide](../../docs/guides/DEVELOPMENT.md) for short training/evaluation commands and the paired runtime benchmark. New output directories must be empty; generated defaults are unique.

Check the [known issues](../../docs/reference/KNOWN_ISSUES.md) before starting new experiments. Recorded runs and checkpoint files stay outside version control.

## Neural self-play prototype

`ppo_training/train_unified.py --mode selfplay` trains from random neural weights
against frozen checkpoints from the same fresh run, with no bot bootstrap.
`eval_selfplay.py` evaluates a final model on separate seeds against fixed neural
references or chosen trained checkpoints, paired with its untrained baseline.
See [the experiment guide](../../docs/archive/selfplay/SELFPLAY_POC.md).

The [science diagnosis guide](../../docs/archive/science/SCIENCE_DIAGNOSTICS.md) describes passive
training telemetry and a four-condition exploration/credit-assignment experiment.
Enable it with `--science-diagnostics`; offline analysis never launches games.
It emits both a JSON summary and a standalone HTML dashboard. The dashboard
builder also accepts `--science-summary PATH` to load saved diagnostics alone or
alongside an existing `--evaluation-dir`.
The [green-priority comparison](../../docs/archive/science/GREEN_PRIORITY_EVALUATION.md) evaluates
one trained checkpoint freely and with legal green building forced, 480 games per
mode by default, with paired intervals and the existing console/dashboard.

`audit_simulator.py` regenerates the exact C++ comparison against the pinned
upstream commit; `--check` detects simulator files changed without updating the
record. Every C++ correction also requires an entry in
[the simulator change log](../../docs/SIMULATOR_CHANGES.md).
The audit delegates to the separate `engine/` submodule. `check_engine.py`
verifies its exact commit and contract against `engine.lock.json`; see the
[engine workflow](../../docs/guides/ENGINE.md) before updating it.

## Runtime options (25 October 2026)

The unified trainer automatically shares per-status observations, uses critic-only
bootstrap, indexes its own snapshot pool and caches the eligible opponents plus
active seats. Snapshots contain inference weights; `latest.pt` still includes the
optimizer. Use `--opponent-cache-size N` to override the automatic cache bound.

For a new run, `--throughput-preset` selects eight environments with horizon 64,
retaining 512 samples per PPO update. Ordinary defaults remain two environments
with horizon 256 so existing experiment commands retain their trajectory setup.
The preset changes the GAE horizon and seed/game distribution; compare playing
strength on the same holdout plan before adopting it for an experiment series.

`eval_selfplay.py` and `eval_green_priority.py` accept `--workers N` (default 2).
The unified trainer accepts `--eval-workers N` (default 1); process startup can
outweigh parallelism on tiny evaluations. Each CPU worker uses one Torch thread,
keeps its reference cache and writes isolated exchange files. Final records retain
the original plan order, game IDs and local seed streams.

```sh
.venv/bin/python scripts/ppo_training/train_unified.py --mode selfplay --throughput-preset --out-dir runs/new-throughput-run
.venv/bin/python scripts/eval_selfplay.py --run-dir runs/new-throughput-run --workers 2 --out-dir runs/new-throughput-holdout
.venv/bin/python scripts/benchmark_optimizations.py --help
```

The before/after benchmark requires a preserved source tree and engine binary,
uses warm-up and alternating trial order, and rejects changed trajectory/model
fingerprints. It also measures detailed evaluation against common checkpoints,
shared feature construction and complete engine-status transport fingerprints.

## Science and military specialists

The [specialist reward plan](../../docs/archive/specialists/SPECIALIST_REWARDS.md) defines terminal-only
category bonuses and the evaluation procedure. `train_unified.py` accepts
`--init-checkpoint`, `--reward-profile standard|science|military` and
`--specialist-weight` (default 1). Initial weights are loaded with a fresh optimizer,
new counters and a separate league. Existing source checkpoints are not changed.

`episodes.jsonl` records `reward_components`: base VP margin/reward, actual final
category points, added bonus and total training reward. Evaluation reports actual
VP and wins without the training bonus. Warm-start holdouts identify their initial
model as pretrained.

Print commands without starting training (run from `python/`):

```sh
.venv/bin/python scripts/ppo_training/specialist_experiment_commands.py
.venv/bin/python scripts/ppo_training/specialist_experiment_commands.py --phase evaluate
```

Prepared local scripts: `bash runs/train_specialists_v1.sh` trains science and
military sequentially, 524,288 additional transitions each. Then
`bash runs/evaluate_specialists_v1.sh` evaluates both on the same frozen ordinary
opponents. `runs/train_specialists_with_control_v1.sh` is an alternative experiment
series that additionally trains a standard-reward control in separate directories.
Do not run both training scripts as a single series; choose whether to include the
control. Interrupted matching outputs are archived and restarted from the source,
not resumed. Finished matching runs are skipped.

The specialist opponent plan was subsequently changed to **fixed normal models**.
Use `bash runs/train_and_evaluate_specialists_v2.sh` from `python/`: it trains both
specialists against two frozen copies of the ordinary baseline `latest.pt`, then
evaluates both against the common older ordinary references. Own specialist
snapshots are saved but never admitted as training opponents. The older v1 scripts
use the previous growing self-play pool and are not the commands for this series.
The generator now defaults to the v2 prefix and accepts `--phase all` and
`--training-opponent PATH [PATH ...]` to print a complete experiment sequence.

## Discards by age (offline)

`ppo_training/analyze_discards.py EVALUATION_DIR [EVALUATION_DIR ...] --out-dir NEW_DIR`
reads existing detailed evaluation records and writes JSON plus `by_age.csv`.
It reports executed voluntary discards by age, round, available structure builds,
coins and card color, with actual-action denominators and engine-seed-block
intervals. Untrained models, constrained policies and duplicate source/plan
combinations are excluded. See [the 26 October analysis](../../docs/archive/science/DISCARD_ANALYSIS_2026-10-26.md)
for the seven completed long-run models and their available intermediate states.

## Multi-head exploration: one run, then evaluate

From `python/`, run:

```bash
.venv/bin/python scripts/ppo_training/run_multihead_experiment.py
```

Four heads, no priors initially, 262,144 additional transitions total, against a
fixed mixture of two standard models plus the science and military specialists.
Then evaluate the initial model, final per-game head selection and every head
separately. `--dry-run` prints commands without training. No additional experiment
starts automatically. See [MULTIHEAD_EXPLORATION.md](../../docs/archive/exploration/MULTIHEAD_EXPLORATION.md)
for budgets, population weights, diagnostic outputs and interpretation limits.

The prepared next experiment enables calibrated fixed priors while retaining
the original starting model, opponents and budget:

```bash
bash scripts/ppo_training/train_and_evaluate_priors_v2.sh
```

It trains once, evaluates, and compares against saved v1 results without extra
comparison games. See [MULTIHEAD_PRIORS_V2.md](../../docs/archive/exploration/MULTIHEAD_PRIORS_V2.md).
`calibrate_priors.py` measures policy perturbation without learning;
`compare_multihead_runs.py` matches saved game plans by opponent hashes.

Frische gepaarte Evaluation nur der fertigen v1/v2-Zufallskopf-Policies:

```bash
.venv/bin/python scripts/ppo_training/eval_multihead_fresh.py
```

Standard: 32 neue Seed-Blöcke, alle Gegnerpaare/Sitze, 1.536 Spiele pro Modell;
keine Einzelkopf-Evaluation und kein Training. Modellkopien, Seed-Prüfung und
Spielplan werden vor der Ausführung gespeichert. Ergebnisse des ersten Durchlaufs:
[MULTIHEAD_FRESH_EVALUATION_2026-10-26.md](../../docs/archive/exploration/MULTIHEAD_FRESH_EVALUATION_2026-10-26.md).

## Langer Lauf mit Wunderinformationen und Zwischen-Dashboards

```bash
.venv/bin/python scripts/ppo_training/run_long_training.py
```

Ein Lauf mit 2.097.152 zusätzlichen Übergängen, vier Köpfen/Priors aus v2,
öffentlichen Karten-/Wunderidentitäten und gemischter Snapshot-Liga. Evaluationen
bei 0/25/50/75/100% veröffentlichen das Dashboard automatisch während des Trainings.
Öffnen: `.venv/bin/python scripts/open_dashboard.py --run-dir runs/long-mixed-wonders-v1-seed131`.
Nach einer Unterbrechung denselben Trainingsbefehl mit `--resume` aufrufen.
Einstellungen, Informationsvertrag und Resume-Grenzen stehen in
[LONG_TRAINING_RUN_V1.md](../../docs/archive/long-training/LONG_TRAINING_RUN_V1.md).

Zweiter langer Lauf mit ausschließlich eigenen Trainingsgegnern (60% jüngere
Snapshots, 20% gefülltes historisches Archiv, 20% feste End-/25%-Anker):

```bash
.venv/bin/python scripts/ppo_training/run_long_training_v2.py
```

Weitere 2.097.152 Übergänge aus dem bisherigen Endmodell. Spezialisten und alte
Standardmodelle werden nur noch separat evaluiert. Details und Dashboardbefehl:
[LONG_TRAINING_RUN_V2.md](../../docs/archive/long-training/LONG_TRAINING_RUN_V2.md).
