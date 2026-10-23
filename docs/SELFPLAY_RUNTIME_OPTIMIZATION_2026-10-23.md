# Self-play runtime optimizations: 23 October 2026

Implemented the requested improvements 1–3 without changing C++ simulator files, the model architecture, PPO/GAE, rewards or opponent sampling:

1. Self-play training and its holdout CLI automatically prefer temporary exchange under `/dev/shm/sevenwonders`. If unavailable/unwritable, automatic selection falls back to disk. Explicit `--io-root` takes precedence and propagates failures. Unique run directories are cleaned; models, logs and diagnostics remain in the persistent output directory.
2. The default evaluation interval is 32,768 transitions. Initial and final neural evaluation remain mandatory; explicit shorter intervals are respected. The moderate-run command already specified this interval, so that command does not gain another evaluation-frequency reduction from changing the default.
3. Frozen neural opponents sharing the same cached policy are evaluated together across collector environments. Standalone evaluation groups shared opponents within one game. Each observation is built from its own pre-turn state and each player keeps its own RNG. Validated actions are passed to the environment without sampling them again. Different policies and unsupported opponent types retain separate paths. `--opponent-inference serial` provides the comparison setting.

The value-bootstrap optimization previously listed as proposal 4 was not implemented.

## Verification

All 82 Python tests and four CTests passed. New checks cover shared versus distinct policies, stochastic and greedy decisions, seat-local RNG state, unchanged frozen weights, grouping four requests into one model call, illegal-action rejection, automatic RAM selection, disk fallback and explicit overrides. The neural-only integration test still forbids bot creation and verifies actual grouping plus initial/final evaluation with a sparse interval. The simulator checksum inventory still matches all 49 files.

## Short paired measurement

Six successful 2,048-transition self-play runs: two seeds, three configurations, reversed configuration order on the second repeat. Two environments, horizon 128, snapshots every 512 transitions, one evaluation seed block (12 games) per evaluation. An unmeasured fixed-mode warm-up preceded the runs. Wall times include process startup and checkpoint/evaluation output.

| Configuration | Repeat | Total seconds | Collection seconds | Evaluation seconds | Opponent decisions seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Disk, serial, evaluation every 512 | 1 | 34.29 | 15.99 | 11.31 | 6.97 |
| RAM, serial, evaluation every 32,768 | 1 | 24.18 | 13.78 | 3.92 | 6.41 |
| RAM, batched, evaluation every 32,768 | 1 | 25.37 | 13.11 | 4.27 | 4.92 |
| Disk, serial, evaluation every 512 | 2 | 33.49 | 15.07 | 11.19 | 6.36 |
| RAM, serial, evaluation every 32,768 | 2 | 25.69 | 14.89 | 3.82 | 6.94 |
| RAM, batched, evaluation every 32,768 | 2 | 21.23 | 11.05 | 3.60 | 3.96 |

All rollout trajectory hashes, final model-weight hashes and final evaluation outcomes matched exactly within each repeat, despite the different evaluation frequencies. The initial/final reference plan and learning experience remain the same; the denser reference variant additionally evaluates intermediate steps. Selected `best.pt` can differ when fewer intermediate candidates are evaluated; the final `latest.pt` weights are the equivalence target.

The optimized collection used 18–27% less time than disk/serial collection. Opponent decisions alone used 23–43% less time with batching than serial RAM. Total time was 26–37% lower than the deliberately dense-evaluation reference, but that includes fewer evaluations and must not be transferred directly to the already-sparse moderate-run command. In the first repeat batching improved collection but total time was slightly higher than serial RAM, illustrating short-run noise and the contribution of other work.

Two short paired repeats do not establish a universal speedup. Grouping is most effective while multiple seats use the same cached snapshot; a diverse pool with unique policies per request reduces that benefit. Feature extraction and engine exchanges remain. No new long training or playing-strength experiment was run for this optimization.

Full local artifacts: `/tmp/sevenwonders-batched-runtime-rczv8loi/`. The [compact measurement record](SELFPLAY_RUNTIME_OPTIMIZATION_2026-10-23.json) stores configuration, versions, timings, behavior fingerprints and final evaluation outcomes without model weights. Existing learning evidence is documented separately in [the prototype results](SELFPLAY_POC_RESULTS_2026-10-23.md).
