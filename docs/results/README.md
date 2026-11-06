# Results and evidence

## Current Second Edition agent

The public snapshot evaluates the saved **196,608-decision** policy on **40
engine-seed blocks / 1,080 completed games**, all nine ordered pairs of three own
trained references, and all three seats. Mean own VP is **46.61**, mean margin to
the strongest rival **−2.80**, and shared win rate **41.9%**. There were no failed
games. The [snapshot](../../demo/snapshot.json) records panel, model/schema hashes,
engine version, intervals and aggregates.

This is one same-run development panel. It is not evidence of general playing
strength, an isolated memory benefit, or improvement over an untrained model.
The public example is a saved evaluation and may lag local training.

## Historical learning evidence

The retired protocol-2 agent demonstrated learning on its own rule and feature
contract. In a fresh-seed long-run reference test, final versus start improved
VP margin by **+3.80**, with a 95% block-bootstrap interval **[+2.74, +4.78]**;
shared win rate moved from **36.10% to 52.04%**. Further continuation showed an
uncertain **+1.14 VP** margin gain **[−0.72, +2.98]** on its smaller control panel.
These results motivated stronger evaluation; they do not transfer to the new agent.

Full protocols, raw structured summaries and historical code remain available
in the immutable pre-cleanup source:

- [Fresh-seed long-run assessment](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/archive/long-training/LONG_FRESH_RESULTS_2026-10-26.md)
- [Continuation and uncertainty](https://github.com/martenkrohn/seven-wonders-rl/blob/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/archive/long-training/LONG_TRAINING_V2_RESULTS_2026-10-29.md)
- [Historical experiment archive](https://github.com/martenkrohn/seven-wonders-rl/tree/ab3860311cbd03f22a63ec242929e10f7d626e11/docs/archive)

## Runtime measurements, 6 November 2026

On an i7-7700 under WSL, isolated CPU optimizations improved serial training
throughput by about 30–34%. A four-game shared-learner prototype reached about
42 decisions/s, roughly 42% faster than its own one-game baseline. Four-process
evaluation reached about 155 games/min including startup on the short plan;
the optimized GPU setup did not improve end-to-end throughput.

These are historical measurements, not a performance guarantee for every machine
or a new benchmark of the consolidated trainer. Worker count also changes the
per-game GAE horizon. The [structured measurement record](performance-20261106.json)
retains the unique prototype evidence; rejected prototype implementations and
redundant step-by-step acceptance logs are no longer maintained. Use
`python/scripts/benchmark_decision_performance.py` for a bounded measurement of
the current implementation.
