# Remaining limitations

The corrected runtime and short development checks do not establish general playing strength or complete board-game rule correctness.

## Model and features

New models use architecture v2: a nonlinear joint policy head after concatenating action and global embeddings. Historical architecture-v1 models had a global additive term that cancelled in softmax. They retain their original linear head during explicit `--allow-legacy` inference; their weights are not migrated into the new architecture. New training starts with fresh weights.

The current production input already includes the wonder's starting resource, for self and both neighbours. There is no separate wonder identity or future-stage cost/effect description in the model input. Updating the old engine's wonder stages and special abilities to the newer edition is deferred. The existing production observation and action space are unchanged by the architecture correction.

Wonder-action features are still partly unimplemented. Detailed gain statistics are feature estimates and explicitly identify unavailable wonder features; they must not be interpreted as complete realized resource/VP accounting. Extra Halikarnassos discard choices remain unsupported (`extra` is empty). Other inherited rule issues may exist beyond the targeted regressions.

## Training and historical data

Historical runs used mixed-environment GAE and premature end-state scoring. Their learning curves are not evidence for the corrected trainer. Older feature semantics are rejected by default. The detailed evaluator permits explicit historical inference with `--allow-legacy`; that does not make results comparable or provide training resume.

The unified trainer is CPU-only, requires step budgets divisible by the environment count and has no exact-resume option. A technical transition failure stops training and discards the pending rollout. The last successfully saved checkpoint remains intact. Automatic recovery of partially collected games is not implemented.

## Opponents and performance

The self-play mode now uses only fresh neural snapshots, including an immediate random initial checkpoint, and evaluates against fixed initial neural references with balanced seats. The initial references establish basic learning only; use separate test seeds and trained intermediate opponents before inferring broader strength. Active opponent sampling still uses a simple recent/uniform mixture; no adaptive league or independent trained population is implemented. Reward remains VP margin while model selection uses tied-win share. Heuristic policies remain available in the separate fixed-opponent mode.

The transport remains file-based and starts one process per game. Self-play training and holdout evaluation automatically prefer RAM exchange, with a disk fallback; `--io-root` explicitly overrides this. Logs and weights remain persistent. Frozen opponent inference is grouped across environments when the same cached policy serves multiple seats; distinct snapshots, feature construction and engine processes still incur their own work. Grouping helps less when every request uses a different snapshot. `--engine-poll-us 0` retains a busy-loop comparison, while the default 1000 microseconds yields while waiting. Performance depends on storage, CPU and workload; use paired measurements instead of assuming every filesystem call reaches an SSD.

The pool bounds eligible opponents but does not delete checkpoint files. Persistent checkpoint retention remains the maintainer's responsibility. New leagues should use a separate directory with current checkpoint semantics.

Dependencies are not locked. Same-build seeded replay and repeated CPU training are tested; identical results across operating systems, compiler/STL or library releases are not promised.
