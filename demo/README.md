# Public dashboard snapshot

This directory contains the deliberately selected public example, rather than
the private training run directory. `index.html` embeds derived evaluation
statistics and the dashboard assets; `snapshot.json` records the description,
export time, source-analysis hash and available software provenance.

The current example uses the **Seed 47** experiment, before feature semantics v3.
It illustrates the evaluation tooling and observed strategic limitations, not
the effects of the subsequent feature corrections.

The GitHub Pages workflow publishes only `index.html`, `snapshot.json` and
`.nojekyll`. No model weights, raw game logs or private run folders are uploaded.

See [the update guide](../docs/DASHBOARD_DEMO.md) to replace the example while
keeping the same public URL.
