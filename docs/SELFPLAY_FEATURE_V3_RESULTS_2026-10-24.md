# Self-play assessment: feature semantics v3

Assessed on 24 October 2026. Run: `poc-moderate-features-v3-seed31-restart1`.

**Finding:** learning is clearly demonstrated on this test plan, with a modest further gain after the halfway checkpoint. Science appears more often at the end, but early science investment remains rare. This is evidence of learning with limited strategic diversity, not evidence that deep exploration has been solved.

## Experiment and data quality

- Training seed 31; 262,144 transitions; 512 PPO updates; 12,482 completed training games.
- Training completed normally in 2,690.97 seconds (44 minutes 51 seconds); all eight checkpoint evaluations completed their 144 games without failures.
- Source commit: `dd9c3fc2f57dd0b60fca03a7126b07a62da7cee3`, clean working tree at training start; feature semantics v3, model architecture v2.
- Separate evaluation seed 59001; 40 engine-seed blocks; 480 games for each of four models. Frozen opponents are the same run’s 65,536- and 131,072-transition checkpoints.
- Each block covers all three learner seats and four opponent combinations (both homogeneous pairs and both mixed orders). Every tested model uses the same plan.
- All 1,920 games completed: zero failures or truncations. Each has final states, three age snapshots and agent action traces. No score-sum mismatches, invalid age sequences or unknown built cards were found.
- Evaluation actions are stochastic. Reported 95% intervals use 2,000 bootstrap resamples of whole engine-seed blocks; seat and opponent rotations within a block are correlated.

## Playing strength

| Model | Win share | Own points, mean | Margin vs strongest opponent, mean |
|---|---:|---:|---:|
| Untrained (0) | 0.00% | 16.15 | -32.40 |
| 65,536 transitions | 28.44% | 36.84 | -5.85 |
| 131,072 transitions | 38.23% | 37.77 | -3.16 |
| Final (262,144) | 43.54% | 39.04 | -2.03 |

Final-model win-share interval: **39.06% to 48.23%**. Final margin interval: **-3.09 to -0.98 points**.

Win share gives tied wins fractional credit. The final model won outright in 203 games and shared a top score in 12. A negative mean margin can coexist with a win share above one third: large losses can outweigh smaller wins in the score mean. This does not imply that the agent is generally stronger than arbitrary opponents.

### Paired final-model gains

| Compared with | Margin gain | 95% interval | Win-share gain | 95% interval (percentage points) |
|---|---:|---:|---:|---:|
| Untrained (0) | +30.37 | [+28.898, +31.907] | +43.54 pp | [+39.06, +48.23] |
| 65,536 transitions | +3.83 | [+2.362, +5.236] | +15.10 pp | [+8.85, +21.46] |
| 131,072 transitions | +1.13 | [+0.008, +2.242] | +5.31 pp | [+0.83, +10.11] |

The large gain over the untrained policy is clear on this plan. Gains over 65,536 steps are also clear. The gain over 131,072 steps is smaller: its margin interval only barely excludes zero (lower bound approximately +0.008 points). Treat that late margin improvement cautiously rather than as a precise or robust general effect. The win-share interval for the late gain is positive on this plan. These selected checkpoints do not establish monotonic improvement at every update.

## Final scoring profile

| Category | Mean | P10–P90 across games | Maximum |
|---|---:|---:|---:|
| Civilian / blue | 12.58 | 6.0–21.0 | 34 |
| Military | 5.61 | -3.0–16.0 | 18 |
| Science | 1.03 | 0.0–4.0 | 16 |
| Commercial / yellow | 4.26 | 0.0–9.0 | 13 |
| Guild | 2.80 | 0.0–9.0 | 17 |
| Direct wonder points | 7.50 | 3.0–13.0 | 20 |
| Treasury | 5.24 | 2.0–9.0 | 15 |

Civilian points and wonders remain major scoring sources. Yellow cards contribute more late in training: their mean score rises from 1.99 at 65,536 steps to 2.94 at 131,072 and 4.26 at the end. These category differences describe the policy; they are not causal values of individual actions. P10–P90 describes game-to-game variation, not uncertainty in the mean.

## Science: some emergence, little early commitment

| Model | Science mean | Games with science points | Science maximum |
|---|---:|---:|---:|
| Untrained (0) | 1.77 | 63.3% | 16 |
| 65,536 transitions | 0.34 | 26.0% | 10 |
| 131,072 transitions | 0.30 | 24.4% | 4 |
| Final (262,144) | 1.03 | 42.3% | 16 |

The final model scores zero science points in 57.7% of games. Its median is zero, P10–P90 is 0–4 points, and its mean among games with positive science is 2.44. Only 10/480 games (2.1%) reach at least 10 science points; none reaches 20. This is not yet a frequently successful high-scoring science strategy.

Actual green-card choices by the final model:

| Age | Decisions with a legal green build | Green buildings chosen | Choice rate |
|---|---:|---:|---:|
| I | 429 | 22 | 5.13% |
| II | 1,178 | 98 | 8.32% |
| III | 1,100 | 124 | 11.27% |

Early science is not completely ignored, but is still rare. Age-I choices were Apothecary 10, Scriptorium 8 and Workshop 4. Rates count decision opportunities, including repeated availability, rather than unique cards dealt or counterfactual best actions. The final model averages only 0.046 green buildings in Age I, versus 0.204 in Age II and 0.258 in Age III.

Science points can also come from wonders and guild effects. A positive science score therefore does not by itself demonstrate a deliberate early green-card strategy. The untrained model actually has a higher science mean than the final model while losing every game; more green alone is not a sufficient success criterion.

## Production and investment behaviour

- The final model builds 1.78 brown buildings per game in Age I and 0.29 in Age II. Production investment is concentrated early rather than absent.
- Grey buildings are extremely rare: 20 across 480 games, including just one in Age I. When a grey building is legally available, it is selected in about 0.5% of decisions.
- This is not the same as having no manufactured-resource access. Forum is built 265 times, Caravansery 293 times; trade discounts and starting resources also matter. Choice production is not counted as multiple fixed resources in the economic tables.
- The policy builds 1.42 yellow cards per game in Age I and 1.64 in Age II. East Trading Post is built 379 times, West Trading Post 171 and Marketplace 133. This shows a particular economic preference, not proof that those choices are optimal.
- Discards decrease from 10.24 per game for the untrained model to 2.06 at the end; final Age-I discards average 0.32. That is evidence of more productive action use, without establishing optimal investment timing.
- The mean coin balance is 3.76 after Age I, 8.14 after II and 16.80 at the end. Retained coins may support trade or treasury points; these aggregates do not establish that spending more would improve results.

## Exploration and limits of the diagnosis

Mean policy entropy falls from 1.257 over the first 32 updates to 0.426 over the last 32. That is consistent with a more concentrated action policy, but changing states and legal masks also affect entropy. This trend and rare early science make exploration and delayed credit assignment reasonable next investigation targets; neither is established as the sole cause.

This experiment does not isolate the effect of feature corrections. The previous public example had different seeds, opponents and feature semantics; it is not a controlled baseline. Only one training seed and same-line opponents are tested. Wonder-stage migration remains deferred, and these results apply to the simulator variant in use.

A useful next experiment would keep the current model inputs, terminal reward and evaluation plan fixed, then compare exploration settings across independent training seeds. Predefine both playing-strength and strategy diagnostics, especially early legal science choices and the frequency of high-science games. A colour-specific reward bonus is not needed to perform that comparison. This is an experiment proposal, not a demonstrated remedy.

## Artifacts and reproduction

[Open the public dashboard](https://martenkrohn.github.io/seven-wonders-rl/). The public snapshot contains the aggregate statistics for all four models; select the model in the dashboard. Its metadata and the embedded statistics are versioned in `demo/`. Local raw games and checkpoints are not committed.

Local detailed directory: `python/runs/poc-moderate-features-v3-seed31-restart1-detailed/`. It contains `dashboard.html`, `report.txt`, `report.md`, `analysis.json`, `categories.csv`, `cards.csv`, per-model JSONL files and the combined `games.jsonl`.

From `python/`, the evaluation command is:

```sh
.venv/bin/python scripts/eval_selfplay.py \
  --run-dir runs/poc-moderate-features-v3-seed31-restart1 \
  --reference runs/poc-moderate-features-v3-seed31-restart1/checkpoint_65536.pt \
              runs/poc-moderate-features-v3-seed31-restart1/checkpoint_131072.pt \
  --compare runs/poc-moderate-features-v3-seed31-restart1/checkpoint_65536.pt \
            runs/poc-moderate-features-v3-seed31-restart1/checkpoint_131072.pt \
  --blocks 40 --seed 59001 \
  --out-dir runs/NEW-EMPTY-EVALUATION-DIRECTORY
```

An existing local evaluation can be re-rendered with `--report-only --out-dir EXISTING-DIRECTORY`, without new games. The public snapshot is exported using `scripts/prepare_dashboard_demo.py`; see [the update guide](DASHBOARD_DEMO.md).

### Identifiers

Source `analysis.json` SHA-256: `e27c47cea1d894e29a703c13b048902a8288bc3c191731b654cca9b086501604`.

| Model | Checkpoint SHA-256 |
|---|---|
| Untrained (0) | `d77c26c99c979428f6d067ba8c10f16e0430de9e687b9fa59ca2b183de9cf8fe` |
| 65,536 transitions | `6f063aff828ca6e2d1c4396e3806b02cf843be1ae003b0ada632c15eb20a9ac6` |
| 131,072 transitions | `dee025061c6bb9acdc6388669c701377f4ed1d3940ab37a70ef8f0cc00c8aef9` |
| Final (262,144) | `c1235b96fd44d52c045eca4b1f3bbb00164deca56d49d764ece4c0e011063847` |
