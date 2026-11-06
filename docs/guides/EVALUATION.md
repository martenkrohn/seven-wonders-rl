# Evaluation and dashboard

The independent evaluator uses immutable active-agent snapshots. It leaves the
training learner, optimizer, sampler and journals untouched and may run while
training continues, sharing CPU resources.

```sh
python/.venv/bin/python -u python/scripts/eval_decision.py \
  --run-dir python/runs/continuous-decision-pfsp-v1-seed331 \
  --engine build/7Wonders --workers 4
```

By default the first three trained snapshots form the fixed reference panel.
They become available at 196,608 decisions with the normal snapshot interval.
Forty engine-seed blocks cover all nine ordered pairs and three learner seats:
1,080 games per model. Later evaluations compare the current committed snapshot
with the previous successfully evaluated model on the same plan. A new panel or
different seed/block configuration requires a different `--evaluation-dir`.

The panel is a development comparison against same-run policies. New game seeds
do not make these opponents unseen policies. General strength requires additional
opponents and independent training runs. Win share splits ties; primary progress
is VP margin against the strongest rival. Confidence intervals resample whole
engine-seed blocks, not individual games. Missing or failed games invalidate the
comparison and remain recorded.

Every session writes `summary.json`, `report.txt`, `dashboard.html`, game JSONL,
an immutable candidate copy and a provenance manifest. The HTML opens directly
without a server or network. It shows win share, VP margin with intervals, score
categories, seats and all opponent pairs. English/German, light/dark themes,
aggregate CSV export and local JSON loading are available. It does not invent
card-choice or age-by-age telemetry that the active evaluator does not record.

To render an existing session again without games or loading weights:

```sh
python/.venv/bin/python python/scripts/build_dashboard.py \
  --evaluation-dir python/runs/YOUR-RUN/evaluation/YOUR-SESSION
```

To open the latest report for the normal run, use the VS Code dashboard task or:

```sh
python/.venv/bin/python python/scripts/open_dashboard.py \
  --run-dir python/runs/continuous-decision-pfsp-v1-seed331
```

Automatic training controls use a separate panel under `intermediate_evaluation/`
every 5,000 completed games. They evaluate the just-saved learner, rather than
waiting for a league snapshot. The normal one-block plan is a quick diagnostic
with no meaningful block-bootstrap interval. It is not directly comparable to
the independent 40-block plan.

To prepare a public snapshot from a complete session:

```sh
python/.venv/bin/python python/scripts/build_dashboard.py \
  --evaluation-dir python/runs/YOUR-RUN/evaluation/YOUR-SESSION \
  --publish-dir demo
```

Only sanitized aggregate data enter `snapshot.json` and the standalone HTML.
Local paths, training manifests, raw games and weights are excluded. Review the
snapshot and commit it; the Pages workflow deploys only `index.html`,
`snapshot.json` and `.nojekyll` from `main`. This is a deliberate saved example,
not a live feed and not necessarily the latest training checkpoint.
