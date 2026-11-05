# Validation

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

## Public-state inputs and milestone training: 26 October 2026

Prepared a 2,097,152-transition phase with the trained v2 four-head policy,
preserved frozen priors, appended public card/wonder information and a mixed
recent/archive/fixed opponent league. Focused checks cover private-hand isolation,
seat rotation, all 14 wonder definitions against the C++ implementation, neutral
input migration, gradients on the appended columns, strict schema loading,
group probabilities, lag and fallback. Real-engine tests interrupt after 25%
and during the final evaluation, then resume with optimizer/RNG state and retain
all completed dashboard reports; unfinished games are explicitly replaced.

The actual-model validation run completed 4,096 transitions plus five fixed-panel
evaluations of 72 games each: 360 completed, no failures/truncations. New feature
weights changed and all prior weights remained byte-identical. Every report is
present in the combined dashboard. Approximate speed excluding evaluation was
139 transitions/s; short-run measurements are not evidence of learning quality.
The full 2.1M run has not been started. See [setup](../long-training/LONG_TRAINING_RUN_V1.md) and
[machine-readable validation](../long-training/LONG_TRAINING_VALIDATION_2026-10-26.json).

## Completed green-priority comparison: 25 October 2026

All 960 saved games were independently checked against identical model hashes,
unique seed/seat/opponent plans, role-local RNG seeds, complete age sequences,
final score totals and category sums. All 2,164 legal green opportunities in the
forced variant produced green buildings, with zero rule violations. Saved paired
results were reproduced; additional paired block-bootstrap intervals were
computed for science, own total points and strongest-opponent points. Findings
are in [GREEN_PRIORITY_RESULTS_2026-10-25.md](../science/GREEN_PRIORITY_RESULTS_2026-10-25.md).
The local dashboard was refreshed. No new games or training were run in this
analysis.

## Same-weight green-priority evaluation: 25 October 2026

Added an evaluation-only legal-action restriction and a paired CLI, defaulting to
480 games per mode (960 total) on test seed 69001. A short real-engine CLI test
played 12 games per mode, verified identical seed/seat/opponent/checkpoint
identities, zero green-priority violations, unchanged checkpoint hash, explicit
free/forced labels, complete reports and offline re-rendering after deleting the
test checkpoint. Component tests cover paid/free green actions, excluding
wonder/discard and illegal green actions, observation immutability, and unchanged
actions/RNG progression when no green is available. The first targeted 23-test
run passed; after dashboard/opener additions, the final 10-test subset passed.
The JavaScript DOM harness passed in V8, including both languages and the new
labels and rule statistics. CLI help and `git diff --check` passed. No full
960-game evaluation or new training was launched. Commands and interpretation
are in [GREEN_PRIORITY_EVALUATION.md](../science/GREEN_PRIORITY_EVALUATION.md).

## Common science holdout analysis: 25 October 2026

All 5,760 saved games from four variants and three checkpoints were verified
offline: matching unique seed/seat/opponent plans, 40 seed blocks with 12 games
each, complete actions and age sequences, consistent final totals and category
sums, and no reported failures or truncations. Direct comparisons use 2,000
paired bootstrap resamples of whole seed blocks. Results and limitations are in
[SCIENCE_HOLDOUT_RESULTS_2026-10-25.md](../science/SCIENCE_HOLDOUT_RESULTS_2026-10-25.md).
The workspace dashboard was refreshed and checked to embed four detailed
evaluations plus four training-diagnostic datasets. The optional four-run
replication script passed shell syntax checking; no new games or training were
started for this analysis.

## Dashboard navigation correction: 25 October 2026

Science datasets no longer override every requested sidebar view. Switching
between diagnostic and evaluation tabs selects the appropriate remembered dataset;
science-only files explain missing evaluation data while keeping navigation
functional. The workspace opener now builds a separate combined dashboard from
available completed holdout reports and science diagnostics, without waiting for
every variant or overwriting files the evaluator may still be producing.
Twelve targeted Python tests passed. The JavaScript DOM harness passed in V8,
including sidebar transitions to all six evaluation views, return to the prior
science dataset, missing-data views, both languages and mixed imports. Existing
local dashboard assets were refreshed without running games or training.

## Restarting interrupted training: 25 October 2026

The generated science experiment commands now archive incomplete matching runs
and skip completed matching runs. Twelve targeted tests passed across restart
decisions, science diagnostics and short real-engine training. An injected
failure was followed by a successful fresh restart, verification that the old
checkpoint remained in the archive, and a repeated command that skipped the
completed run without changing its checkpoint hash. Tests also cover missing
completion/configuration files, changed settings, external leagues, concurrent
directory locks and recognition of a legacy process without a lock. The final
six restart-specific tests passed again after test cleanup. No existing user run
was archived, deleted or restarted, and no long training was launched. This adds
fresh-restart support, not checkpoint resume.

## Passive science diagnosis: 25 October 2026

Dashboard integration was checked separately with 12 passing Python tests for
dashboard generation, public-demo compatibility, offline analysis and science
telemetry. Node is unavailable locally, so its pytest wrapper was skipped; the
dependency-free JavaScript DOM harness was executed successfully in a V8 isolate
instead. It covers seven views, both languages, science-only startup, multi-run
diagnostic imports, metric switching, charts, CSV export, invalid imports and
returning to existing evaluation comparisons. No live training process or public
demo snapshot was changed. The offline analyzer now emits a standalone HTML
dashboard alongside its JSON summary.

The targeted 26-test suite passed: `test_science_diagnostics.py`,
`test_collector.py`, `test_training_smoke.py`, `test_training_contracts.py`,
`test_gae.py` and `test_evaluation_reports.py`. Short real-engine training verifies
that enabling diagnostics preserves rollout hashes and final model weights
exactly, and that completed game totals and all three age boundaries match the
existing episode records. Tests cover paid/free green probability mass, skipped
actions, fallback discards, ties, empty denominators and the offline analysis CLI.
The command generator was checked in print-only mode; trainer help and
`git diff --check` passed. No long training or experiment holdout was launched.
See [the diagnosis guide](../science/SCIENCE_DIAGNOSTICS.md) for the user-run commands and
the limits of interpreting these descriptive metrics.

## English/German dashboard: 24 October 2026

The dashboard defaults to English without a saved preference, including when local storage is blocked. A saved manual English/German choice is independent of the theme setting. The same dependency-free JavaScript harness executed successfully in V8 locally and is wired into pytest through Node when available. It exercises all six views in both languages, model identities, locale-specific numbers, CSV labels/values and formula escaping, public experiment context, language persistence, blocked/invalid storage, live theme labels, card filters, sorting, imported report/model selection, translated outcomes/wonders, missing older data and import errors. The local Python dashboard/report checks passed; the Node wrapper is skipped locally because Node is unavailable. The public demo and local dashboard were regenerated from unchanged `analysis.json`; no new games or model changes were involved. Browser visual QA was unavailable. Bilingual public context and the self-contained language asset are part of the generated HTML.

The Node language contract also passed in GitHub Actions. That run exposed an existing deterministic opponent-batching test relying on exact greedy tie-breaking: actions 21 and 22 have identical feature vectors for the failing observation. The regression now compares all batched/serial logits and values within float32 numerical tolerances and permits differing greedy choices only for numerically tied scores. Stochastic actions, seat RNG states, frozen weights and batching counts remain checked. The full local suite then passed (105 passed; the Node wrapper skipped locally). No training or inference implementation changed.

## Public dashboard snapshot: 23 October 2026

All 105 Python tests and four CTests passed before publication. The simulator audit still matches all 49 inventoried files. Public-export tests cover removal of local checkpoint paths, HTML escaping, preservation of zero values and provenance, refusal of incomplete telemetry or inconsistent scores, and explicit disclosure of failed evaluations. The public Seed-47 snapshot includes four model evaluations: 480 completed games each for the final model and both trained comparisons; 479/480 for the untrained baseline. That failure remains in the data and is highlighted in the introductory context. These results predate feature semantics v3. No training or evaluation games were run to create the snapshot; the export uses saved derived statistics. The Pages workflow uploads only the selected HTML, metadata and `.nojekyll` file. HTML embedding and links were checked; no browser runtime was available for visual QA. The dashboard also includes a system-aware light/dark switch with a locally saved manual preference.

## Local interactive dashboard: 23 October 2026

All 99 Python tests passed. The detailed report writer now generates a self-contained `dashboard.html`, including automatic generation from report-only evaluations. Dashboard regressions verify data embedding, preservation of zero versus missing values, script-breakout escaping, local comparison inputs and generation without loading weights. JavaScript executed in V8 passed numeric ascending/descending sorting with missing values last, combined card filters and formatted CSV export checks. A lightweight DOM harness exercised all six views, search/filter rendering, sort interaction, report switching, old-data missing states and rejection of invalid imports. No browser runtime was available for visual or real-browser QA; that remains a verification limit. The generated Seed-47 dashboard embeds the existing Seed-31 holdout as a second dataset. No games were rerun and the simulator inventory remains unchanged. See [dashboard usage](../../guides/EVALUATION.md#interaktives-dashboard).

## Feature semantics v3: 23 October 2026

All 97 Python tests and four CTests passed. New feature regressions cover chain prerequisites/alternatives, free action costs, held/new discounts, scientific symbols and separate jokers for actor and critic over all seat rotations, choice-production effects and rejection of old feature semantics. A temporary C++ probe compiled against unchanged engine sources confirms real zero-cost chains, both raw-discount neighbor directions, manufactured discounts, and science/guild interpretation. A fresh 2,048-transition neural-only training run (seed 67) completed with global/action dimensions 102/99; its standalone detailed evaluation completed 36 games across initial, intermediate and final models, with complete final/age/action telemetry and no score-sum mismatches or unknown built cards. These are technical checks, not evidence of stronger learned strategy. The 49-file simulator inventory is unchanged. See [the correction log](../selfplay/FEATURE_CORRECTIONS_2026-10-23.md).

## Rolling trained checkpoint references: 23 October 2026

All 91 Python tests passed. In-training self-play evaluation now selects the latest strictly older trained snapshot plus a distinct snapshot nearest half the current step, excluding step zero, the current learner and future snapshots. No initial untrained evaluation runs. New tests cover sparse/empty pools, deterministic refresh and suppression of comparisons across different plans. A further real-engine subprocess test completed successive measurements at steps 256 and 512, verifying that references changed from [128] to [384, 256], plan identities differed, no cross-plan paired gains were reported and the latest fully evaluated learner was saved as `evaluated.pt`. The separate holdout evaluator, training opponent selection and PPO objective are unchanged. The simulator audit still matches all 49 files.

## Evaluation views and telemetry: 23 October 2026

All 89 Python tests passed after adding compact category distributions and standalone detailed reports. New tests cover separate treasury scoring (including coin boundaries), mixed science distributions, missing-data handling, seed-block uncertainty and real-engine telemetry. Detailed and compact evaluation on identical seeded games produce the same totals, win shares, margins and action counts, while preserving both global Torch and NumPy RNG states. All three age snapshots are captured after the proper transitions; the final snapshot matches final scoring for every player. A separate 36-game CLI check exercised an intermediate comparison model and all report exports. An existing 960-game holdout was rendered offline without new games, honestly marking unavailable old telemetry. The simulator audit still matches all 49 inventoried files; no C++ source changed. Commands and report semantics are documented in [EVALUATION.md](../../guides/EVALUATION.md).

## Self-play runtime optimizations: 23 October 2026

Requested improvements 1–3 are implemented: automatic RAM exchange with disk fallback, a 32,768-transition default evaluation interval and grouped frozen-opponent inference with seat-local RNGs. All 82 Python tests and four CTests passed. Six short paired neural-only training runs produced identical rollout hashes, final model weights and final evaluation outcomes across serial/disk/dense-evaluation and optimized variants. Collection used 18–27% less time in this small comparison. Full measurements, the evaluation-frequency distinction and limitations are in [SELFPLAY_RUNTIME_OPTIMIZATION_2026-10-23.md](../performance/SELFPLAY_RUNTIME_OPTIMIZATION_2026-10-23.md). No C++ source changed; the 49-file simulator inventory still matches.

## Neural self-play prototype: 23 October 2026

The three prototype changes are implemented: immediate neural-only opponents, per-game opponent identities/hashes and fixed neural evaluation with a pre-training baseline and separate holdout tool. All 75 Python tests and four CTests passed. The initial pilot attempts exposed a reproducible choice-production allocation leak between player decisions; ENG-006 in [SIMULATOR_CHANGES.md](../../SIMULATOR_CHANGES.md) records its before/after regression and minimal C++ correction. The exact upstream patch reconstructs all 49 inventoried C++/build/test files byte for byte; CI checks that inventory.

Two fresh 8,192-transition runs each completed 390 training games. Separate test-seed evaluations show improvements against initial neural references and trained same-run intermediate policies, with 1,152 additional comparison games all completed. Full results, uncertainty and limitations are in [SELFPLAY_POC_RESULTS_2026-10-23.md](../selfplay/SELFPLAY_POC_RESULTS_2026-10-23.md). These provide initial learning evidence, not a general playing-strength or long-run convergence claim. The moderate next run is prepared but has not been started.

## Model architecture correction: 23 October 2026

New training uses architecture v2 with a nonlinear joint state/action policy head. The observation schema remains unchanged: wonder starting resources are already included in current production totals. Wonder-stage rules and special abilities were not updated.

All 71 Python tests passed, including short subprocess-training integration. New regressions verify that changing only global context can reverse action preferences, that policy loss alone sends gradients into the global encoder, that all seven resource types reach the flattened input for each relative seat, and that real initial engine states expose the expected wonder starting resources. Checkpoint tests cover new architecture metadata, prediction-preserving legacy inference and rejection of inconsistent metadata.

An additional 256-transition, two-environment self-play integration run completed with snapshots every 32 transitions and zero pool lag. Scheduled evaluations completed; `latest.pt`, `best.pt` and the final snapshot loaded as architecture v2, a snapshot policy selected a legal action, and an actual historical checkpoint loaded as architecture v1 with explicit legacy permission. This checks the existing mixed self-play path; it does not introduce pure self-play or establish learning quality. Artifacts are under `/tmp/sevenwonders-model-v2-g4z21am4/`.

Changed Python entry points compiled, and `git diff --check` passed. No C++ source changed for this correction. The runtime benchmark below predates architecture v2 and was not repeated.

## Runtime and training implementation: 23 October 2026

Implemented the restart audit's points 1–5. Opponent strategies, pool sampling rules and model architecture were not redesigned. The implementation was checked on Ubuntu/WSL, Python 3.12.3; full runtime/dependency information is in the [throughput record](../performance/RUNTIME_BENCHMARK_2026-10-23.json).

| Check | Result |
| --- | --- |
| Unchanged starting point | Release build, 2 CTest programs, 30 unit/component and 6 integration tests passed |
| Corrected C++ build | Release build completed; 3 CTest programs passed with assertions active |
| Python suite | All 60 tests passed, including engine and subprocess-training integration |
| Mathematical regression | Independent environments, distinct bootstrap values, staggered ends, valid truncation, one-sample normalization and environment permutation checked |
| Final scoring | Controlled C++ last-turn setup verifies seventh-card execution, final age-III battle tokens, guild copy, skipped commands and final JSON |
| Failure handling | Startup error, timeout, killed engine, malformed JSON and a final-publication/process-exit race checked |
| Failed PPO rollout | Injected transition error preserves the previous checkpoint/update and persists diagnostics before cleanup |
| Repeated CPU training | Identical two-environment runs produced bitidentical model weights; Python set-order tie resolution corrected |
| One-environment development run | 2,048 transitions, 16 updates, two complete 8-game evaluations and final checkpoint |
| Two-environment development runs | Twenty paired 2,048-transition runs completed with final checkpoints; each has 16 updates |
| Evaluator agreement | Fixed and detailed evaluators agree on final scores/outcomes/action counts and preserve the learner's Torch RNG |
| Separate detailed evaluation | 8 seeded stochastic games completed; incremental game/action records and summary written |
| Existing snapshot path | Short 512-transition self-play regression completed with frequent snapshots and zero lag; this checks integration, not self-play quality |
| Checkpoint loading | Model dimensions/hidden size, semantic compatibility and RNG preservation checked |
| Entry points and syntax | Trainer, evaluator and runtime benchmark help/syntax checked |

### Paired throughput experiment

Five paired repeats per configuration, two environments. Each transport trial plays 20 complete games; each training trial performs 2,048 transitions with horizon 64. Warm-up is excluded. Training wall time includes Python startup and checkpoint output; evaluation is disabled. The baseline is the corrected pipeline with ext4 exchange and a busy C++ wait, not an untouched historical binary.

| Storage | C++ wait | Transport median steps/s | Training median steps/s | Paired training ratio vs baseline |
| --- | ---: | ---: | ---: | ---: |
| ext4 | Busy loop | 219.4 | 82.9 | 1.00 |
| ext4 | 1 ms | 317.9 | 99.8 | 1.24 |
| tmpfs | Busy loop | 350.4 | 100.0 | 1.20 |
| tmpfs | 1 ms | 420.3 | 124.8 | 1.51 |

All paired full-status fingerprints and trained-weight fingerprints matched. Across measured trials, 40,960 training transitions, 1,920 completed training games and 400 transport games were processed without failures. The paired ratios are medians of per-repeat ratios, so they need not equal ratios of the speed medians. Transport timings varied more than training timings; see standard deviations in the machine-readable record. These measurements support yielding the busy loop and offering RAM-backed exchange on this machine; they do not establish a universal speedup or playing strength.

The detailed profiling still attributes a substantial share of time to opponent decisions and feature construction. Those strategies remain unchanged. A pipe/direct-binding rewrite is not justified solely by the number of file calls.

Full local logs/checkpoints are under `/tmp/sevenwonders-runtime-benchmark-v2/`, the final single-environment check under `/tmp/sevenwonders-final-one/`, and the separate detail evaluation under `/tmp/sevenwonders-final-detail/`. These are temporary local artifacts, not distributed model releases. The condensed throughput record is versioned alongside this document.

Remaining model, wonder-feature, rule and resume limitations are described in [KNOWN_ISSUES.md](../../reference/KNOWN_ISSUES.md). Historical performance curves remain unvalidated after the corrected training/scoring contracts. No long training campaign was performed.

## Historical validation: 19 October 2026

Checked on 19 October 2026 using Ubuntu/WSL and Python 3.12.3.

| Check | Result |
| --- | --- |
| Source comparison | Compared the engine against upstream `a0bc333a7a7f21f83156de8a2d522e7baa913f0a`; four engine/build files differ before the documentation notices |
| C++ build | Fresh CMake configure/build completed in the reviewed source checkout |
| CTest | Both existing choice-production regression programs passed |
| Python suite | All 36 existing tests passed, including engine integration |
| Documented entry points | Main trainer and detailed evaluator `--help` commands completed |
| Package metadata | Wheel built without dependency downloads; GPLv3 classifier and license file present |
| License preservation | Root license retained; Python distribution receives an identical copy; vendored notices unchanged |
| Documentation | Relative Markdown links checked; references to the private restart guide removed |

The runtime environment, PPO implementation, feature schema and earlier engine corrections were not changed by this documentation cleanup. Source notices were added to the modified engine files. No new training experiment, checkpoint evaluation or benchmark was run. Passing interface tests does not resolve the training issues described in [KNOWN_ISSUES.md](../../reference/KNOWN_ISSUES.md).

No TeX compilation, image review or screenshots were performed.

## Runtime optimization validation — 25 October 2026

Implemented the low/medium effort measures from the performance audit. The
existing Python suite passed after the core changes; the final targeted run
covering runtime optimizations, collector behavior, the green-priority CLI and
neural self-play snapshot persistence passed **17 tests**. It also verifies
parallel/serial detailed-record equality for greedy and constrained stochastic
selection, unchanged learner weights/RNG/mode, and retention of completed games
when another worker raises. All four CTests passed, and the regenerated simulator
inventory check matches all 49 files. `git diff --check` passed.

A separate benchmark ran 33 measured trials with three repetitions, training
warm-up and alternating configuration order, without concurrent builds/tests.
All six matching training pairs retained identical trajectory and final-weight
hashes. Feature, complete engine-status and detailed evaluation fingerprints also
matched. The detailed evaluation benchmark completed 864 games without failures
or truncation. See [the result report](../performance/PERFORMANCE_RESULTS_2026-10-25.md) and
[compact measurements](../performance/PERFORMANCE_RESULTS_2026-10-25.json) for timings,
provenance, raw artifact locations and limits. No new long-run playing-strength
claim is made for the optional eight-environment preset.

## Specialist reward setup — 25 October 2026

Implemented versioned terminal-only `standard`, `science` and `military` reward
profiles plus compatible checkpoint weight initialization with fresh optimizer and
run state. The mathematical contract and pilot settings are in
[SPECIALIST_REWARDS.md](../specialists/SPECIALIST_REWARDS.md).

The component/unit suite passed 117 tests with one skip before the final focused
additions. The final specialist component checks passed 12 tests; the combined
specialist, training-smoke and dashboard selection passed 19 tests with one
JavaScript-runtime-dependent skip. Tests cover signed military bonus, zero weight,
nonterminal/timeout rewards, source provenance and RNG preservation, real short
training and unshaped holdout results, and pretrained-model reporting. Shell
syntax checks and `git diff --check` passed.

Two additional setup runs initialized from the intended local v3 standard model
completed 1,024 transitions each with eight environments and three valid final
evaluation games each. Each recorded 48 completed training episodes. Observed
terminal category bonuses ranged from 0 to 26 science points and −6 to +18
military points. These are technical setup checks, not evidence of learned
specialization. Local artifacts and provenance are under
`python/runs/specialists-v1-setup-check/`, including `setup_validation.json`.
The 524,288-transition specialist runs were prepared but not launched.

### Fixed normal opponents for specialist training

The specialist command generator now emits `--frozen-opponents` and supports a
combined `--phase all` sequence. Fixed sources are copied and hashed into each
run, validated before simulation, and never expanded with learner snapshots.
Training evaluation uses the fixed neural plan; the separate holdout uses other
ordinary checkpoints. Existing growing self-play remains the default without the
new option. New v2 output directories separate this experiment from the old plan.

The targeted specialist/restart/self-play selection passed 26 tests. Two new
integration checks (science and military) each train 128 transitions, publish five
learner checkpoints, verify every logged opponent against the sole frozen source,
cover all three learner seats and check two complete, paired evaluations against
an unchanged plan. The combined shell script passes `bash -n`; its generated
contents match the generator. Long training was not launched.

## Specialist result audit — 26 October 2026

Both v2 frozen-opponent trainings completed 524,288 transitions/1,024 updates and
24,960 completed training episodes. The offline audit checked all opponent paths
and hashes, per-episode reward decomposition against actual final scores,
finite update metrics and completion counts. No saved holdout engine seed overlaps
these new training games. All 1,920 holdout records have valid final states, full
age sequences and acknowledged action traces. VP totals, margins, strict/shared
wins, military results from age shield comparisons and science results under the
implemented joker rule were recomputed without discrepancies. The duplicate
starting-policy evaluations agree after excluding runtime and checkpoint-file IDs.

A separate exhaustive joker assignment identified an inherited scoring limitation;
no simulator or training code was changed. See the [result report](../specialists/SPECIALIST_RESULTS_2026-10-26.md)
and [compact audit data](../specialists/SPECIALIST_RESULTS_2026-10-26.json). The local executable
checker is `python/runs/specialists-v2-analysis-20261026/analyze.py`. No new games,
training or pool promotion was performed for this analysis.
