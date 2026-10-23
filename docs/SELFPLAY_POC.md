# Self-play proof of concept

The first two short runs and separate holdouts are complete: see [results and limitations](SELFPLAY_POC_RESULTS_2026-10-23.md). They show useful learning against both initial references and trained same-run intermediate opponents. The moderate run below is the next experiment.

The subsequent runtime changes and paired behavior/speed checks are recorded in [the optimization report](SELFPLAY_RUNTIME_OPTIMIZATION_2026-10-23.md). They are execution changes, not another learning campaign.

The first goal is evidence that a fresh neural policy learns useful decisions. All three players use neural policies, starting with random network weights. No heuristic teacher, human demonstrations or old experiment models are used. The current terminal reward remains `(own VP - best opponent VP) / reward_scale`; this tests learning under the existing points objective. Wonder stages, special abilities and model size remain unchanged.

Current training uses [corrected feature semantics v3](FEATURE_CORRECTIONS_2026-10-23.md). Start a fresh run; old weights are not accepted by the corrected observation contract. Earlier learning reports describe feature version 2, not a measured result of these fixes.

## Training and references

Run `python/scripts/ppo_training/train_unified.py` from `python/`. `--mode selfplay` is neural-only. Before the first game it saves `checkpoint_0.pt`, which immediately supplies both opponents. Further checkpoints are sampled independently for the two opponent seats (50% recent, 50% entire eligible pool). Opponents stay frozen throughout each game. An empty pool or incompatible/broken checkpoint stops the run; there is no bot fallback. A custom league directory must be fresh.

Runtime defaults now use `/dev/shm/sevenwonders` for temporary exchange when writable, falling back to the usual disk exchange directory otherwise. Models, results and diagnostics stay in `--out-dir`. The resolved exchange directory is printed at startup and recorded in configuration. An explicit `--io-root` always takes precedence and is never silently replaced. The holdout tool uses the same automatic choice. Each run owns a unique subdirectory and cleans up only that subdirectory.

The default evaluation interval is 32,768 transitions; the final evaluation also runs for shorter budgets when an older trained snapshot exists. Override `--eval-every` for a deliberately denser pilot. `--opponent-inference batched` groups requests to the same cached frozen policy across environments, with separate RNGs per seat/game. Evaluation also groups the same snapshot within each game. The `serial` setting provides a reference path for comparison. Unsupported/custom opponents retain their existing serial behavior. Batching changes execution, not the PPO update, reward, snapshot selection or number of training transitions.

The console announces each evaluation and names its trained reference steps. It shows completion, win share, VP margin and compact category distributions. Changes are compared only on identical reference/seed plans. Progress lines show elapsed and estimated remaining time; the estimate adds remaining evaluations at their observed mean duration. Already running processes retain their previous evaluation behavior.

Only one player trajectory per environment supplies PPO data; its seat rotates between games. This avoids mixing frozen-opponent actions into on-policy updates. No changes to PPO/GAE or network size are introduced by this prototype.

In-training self-play evaluations now use rolling trained references: the most recent strictly older snapshot and a different snapshot nearest half the current step. Step zero and the current/future model are excluded. With two references each seed block covers both homogeneous pairs and both mixed orders over all three learner seats (12 games). With one available trained reference it covers all three seats (3 games). `--eval-games-per-matchup` counts these seed blocks. No initial untrained evaluation runs. If no older trained snapshot exists at a scheduled evaluation, the evaluation is deferred with a console explanation.

Evaluation opponents are refreshed at every measurement. Consequently, raw win shares across different plans are not a fixed benchmark. The exact references, steps, hashes and plan identity are stored in every summary; CSV rows also record plan identity and reference steps. Paired changes and console deltas are computed only when plans match. `evaluated.pt` is the last successfully evaluated learner, not a global best. `latest.pt` remains the final learner. Self-play no longer writes `best.pt`; fixed-opponent mode retains its existing best-model selection.

`references.json` retains initial reference identities for optional separate post-training baseline tests; these are not used by checkpoint evaluations. Separate holdouts can instead supply trained references with `--reference` and compare model stages on an unchanged test plan.
## Short pilot

Use fresh output directories for two independent training seeds:

```sh
.venv/bin/python scripts/ppo_training/train_unified.py --mode selfplay --seed 7 --total-timesteps 8192 --n-envs 2 --rollout-horizon 128 --snapshot-every 1024 --eval-every 4096 --eval-games-per-matchup 6 --out-dir runs/poc-seed7
.venv/bin/python scripts/ppo_training/train_unified.py --mode selfplay --seed 19 --total-timesteps 8192 --n-envs 2 --rollout-horizon 128 --snapshot-every 1024 --eval-every 4096 --eval-games-per-matchup 6 --out-dir runs/poc-seed19
```

`opponents.jsonl` logs each started game's neural opponents, seeds and learner seat. `episodes.jsonl` repeats those identities with outcomes and final state. `updates.jsonl` contains finite-loss checks and optimizer diagnostics. Each `eval_STEP_summary.json` stores the actual evaluation references and plan identity. There is no `eval_0` pre-training evaluation. Whole-seed-block bootstrap intervals are available for paired evaluations on unchanged plans.

Inspect completion counts and errors first. Changing weights or decreasing loss alone do not establish better play. Look for improved results over the unchanged initial policy, reasonably consistent across independent training seeds. Intervals from a few seed blocks are approximate; use the separate holdout before treating a selected checkpoint as evidence.

## Holdout and trained opponents

Use test seeds that were not used for model selection. Evaluate a specified final checkpoint rather than choosing among checkpoints on the holdout:

```sh
.venv/bin/python scripts/eval_selfplay.py --run-dir runs/poc-seed7 --blocks 12 --seed 9001 --out-dir runs/poc-seed7-holdout
.venv/bin/python scripts/eval_selfplay.py --run-dir runs/poc-seed7 --reference runs/poc-seed7/checkpoint_2048.pt runs/poc-seed7/checkpoint_4096.pt --blocks 12 --seed 19001 --out-dir runs/poc-seed7-trained-opponents
```

The tool plays both the final candidate and the original initial model against the same reference plan and reports paired improvement. `candidate.win_share` is also useful against trained opponents; roughly one third is the symmetry benchmark when all policies are equivalent, using shared credit for tied top VP totals. VP ties follow the existing simulator/evaluator objective, not a newly implemented edition-specific tie breaker. Comparison against weak initial opponents alone does not prove robust strategy or long-term improvement.

The standalone evaluator now produces a detailed console report and saved text/Markdown, CSV and JSON reports. Training checkpoint evaluations remain compact, with category means, P10–P90, maxima and science participation. See [evaluation views and commands](EVALUATION.md) for intermediate-model comparisons, age/card telemetry and offline reporting of existing games. Each phase is announced before its games begin; full JSON is saved rather than printed to the console.

## Moderate next run

Once the pilot is numerically stable, strictly neural and shows credible improvement on separate seeds, one moderate run is a useful next investment:

```sh
.venv/bin/python scripts/ppo_training/train_unified.py --mode selfplay --seed 31 --total-timesteps 262144 --n-envs 2 --rollout-horizon 256 --snapshot-every 4096 --eval-every 32768 --eval-games-per-matchup 12 --out-dir runs/poc-moderate-seed31
.venv/bin/python scripts/eval_selfplay.py --run-dir runs/poc-moderate-seed31 --reference runs/poc-moderate-seed31/checkpoint_65536.pt runs/poc-moderate-seed31/checkpoint_131072.pt --blocks 40 --seed 29001 --out-dir runs/poc-moderate-seed31-holdout
```

Estimate duration from the pilot's `result.json`, allowing for evaluation costs. A successful moderate run should improve against earlier trained opponents as well as initial references and complete without technical failures. It is evidence for continued investment, not proof of optimal play. No exact training resume exists yet; choose a bounded uninterrupted budget. The active pool is bounded but checkpoint files remain on disk for same-run comparisons.

If learning remains flat, check observed action diversity, entropy, reward variance, paired intervals and seat-specific outcomes before increasing runtime or changing the simulator. Wunder-rule migration, recurrent networks, adaptive leagues and reward redesign are subsequent experiments, not prerequisites added to this prototype.
