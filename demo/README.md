# Public Second Edition evaluation

`index.html` is a standalone interactive report. `snapshot.json` contains only
sanitized aggregate evaluation data, a fixed plan, model/schema hashes and
engine provenance. It evaluates the saved 196,608-decision policy over 40 seed
blocks and 1,080 games; it is not a live feed or a general-strength benchmark.

The Pages workflow publishes only `index.html`, `snapshot.json` and `.nojekyll`.
Weights, raw games and local training manifests remain local. To replace the
snapshot, follow the [evaluation guide](../docs/guides/EVALUATION.md).
