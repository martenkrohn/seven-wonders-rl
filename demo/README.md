# Public dashboard snapshot

This directory contains the deliberately selected public example, rather than
the private training run directory. `index.html` embeds derived evaluation
statistics and the dashboard assets; `snapshot.json` records the description,
export time, source-analysis hash and available software provenance.

The current example uses the completed **Seed 31, feature semantics v3** run
(`poc-moderate-features-v3-seed31-restart1`): 262,144 training transitions,
followed by evaluation seed 59001 with 40 engine-seed blocks and 480 games per
model. The initial, 65,536-step, 131,072-step and final models share a test plan
against frozen opponents from steps 65,536 and 131,072.

See the [assessment](../docs/SELFPLAY_FEATURE_V3_RESULTS_2026-10-24.md) for
results and limitations. This example replaces the earlier Seed-47 snapshot;
different opponents, seeds and feature versions prevent a controlled direct
comparison with that example.

The GitHub Pages workflow publishes only `index.html`, `snapshot.json` and
`.nojekyll`. No model weights, raw game logs or private run folders are uploaded.

See [the update guide](../docs/DASHBOARD_DEMO.md) to replace the example while
keeping the same public URL.
