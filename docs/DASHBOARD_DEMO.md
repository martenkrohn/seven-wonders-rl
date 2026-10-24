# Public dashboard demo

The public demo is a versioned snapshot of one completed detailed evaluation.
It uses the same dashboard as local reports. Exporting a new dataset replaces
the snapshot at the same URL; it does not change training or model weights.

Expected URL after Pages deployment:
<https://martenkrohn.github.io/seven-wonders-rl/>

## Current example

The completed Seed-31 restart uses feature semantics v3 and 262,144 training
transitions. Its detailed evaluation uses seed 59001, 40 engine-seed blocks
and the same frozen opponents and seat rotations for all four tested models.
Findings are documented in the [assessment](SELFPLAY_FEATURE_V3_RESULTS_2026-10-24.md).
The original local dashboard is generated alongside its `analysis.json` in
`python/runs/poc-moderate-features-v3-seed31-restart1-detailed/`.

## Replace the example after the current training run

1. Finish training and run the separate detailed evaluation. Training checkpoint
   evaluations alone do not contain the card and age traces needed for the demo.
2. From `python/`, export the completed evaluation, replacing the placeholders:

   ```sh
   .venv/bin/python scripts/prepare_dashboard_demo.py \
     --evaluation-dir runs/YOUR-DETAILED-EVALUATION \
     --title "Self-play experiment — YOUR-SEED" \
     --description "Training seed, training budget, feature version, evaluation opponents and limitations."
   ```

3. Open `demo/index.html` from the repository root in a browser. Review the
   description, model labels, data-quality view and published statistics.
4. Commit and push `demo/index.html`, `demo/snapshot.json` and `demo/.nojekyll`
   to `main`. The **Dashboard demo** Actions workflow deploys the new snapshot.
   It can also be started manually from the Actions tab.

The exporter requires evaluations without failed or truncated games by default,
and final states, all three age snapshots and action traces for completed games.
`--allow-evaluation-failures` permits an explicitly reviewed incomplete evaluation;
it retains failure counts and automatically adds a visible warning. The initial
Seed 47 example uses this option: the untrained baseline completed 479/480 games;
the final model and both trained comparisons completed 480/480 each.
Known score/age consistency
errors block export. These checks establish report completeness, not optimal
play or statistical significance. The title and description should distinguish
observed results from hypotheses and state which feature version was trained.

Only derived statistics and selected provenance are embedded. Checkpoint paths
become filenames; local file references are removed. Model and engine hashes
remain available for identification. Raw logs and model weights stay local.
The public interface currently uses the same German labels as the local dashboard;
the introductory experiment context is in English.

## GitHub Pages setup

In the repository, select **Settings → Pages → Build and deployment → Source:
GitHub Actions** once. The repository's Pages environment and Actions settings
must allow deployment from `main`.

The workflow publishes only the three selected files, never the entire checkout.
Changes to unrelated code do not deploy a new dataset. An absent snapshot causes
the workflow to fail rather than publishing an empty demo.

Configuration follows GitHub's [static Pages workflow](https://github.com/actions/starter-workflows/blob/main/pages/static.yml)
and [publishing-source documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).
