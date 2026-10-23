# Self-play proof of concept

The first two short runs and separate holdouts are complete: see [results and limitations](SELFPLAY_POC_RESULTS_2026-10-23.md). They show useful learning against both initial references and trained same-run intermediate opponents. The moderate run below is the next experiment.

The subsequent runtime changes and paired behavior/speed checks are recorded in [the optimization report](SELFPLAY_RUNTIME_OPTIMIZATION_2026-10-23.md). They are execution changes, not another learning campaign.

The first goal is evidence that a fresh neural policy learns useful decisions. All three players use neural policies, starting with random network weights. No heuristic teacher, human demonstrations or old experiment models are used. The current terminal reward remains `(own VP - best opponent VP) / reward_scale`; this tests learning under the existing points objective. Wonder stages, special abilities and model size remain unchanged.

## Training and references

Run `python/scripts/ppo_training/train_unified.py` from `python/`. `--mode selfplay` is neural-only. Before the first game it saves `checkpoint_0.pt`, which immediately supplies both opponents. Further checkpoints are sampled independently for the two opponent seats (50% recent, 50% entire eligible pool). Opponents stay frozen throughout each game. An empty pool or incompatible/broken checkpoint stops the run; there is no bot fallback. A custom league directory must be fresh.

Runtime defaults now use `/dev/shm/sevenwonders` for temporary exchange when writable, falling back to the usual disk exchange directory otherwise. Models, results and diagnostics stay in `--out-dir`. The resolved exchange directory is printed at startup and recorded in configuration. An explicit `--io-root` always takes precedence and is never silently replaced. The holdout tool uses the same automatic choice. Each run owns a unique subdirectory and cleans up only that subdirectory.

The default evaluation interval is 32,768 transitions; initial and final evaluation still run, including for shorter budgets. Override `--eval-every` for a deliberately denser pilot. `--opponent-inference batched` groups requests to the same cached frozen policy across environments, with separate RNGs per seat/game. Evaluation also groups the same snapshot within each game. The `serial` setting provides a reference path for comparison. Unsupported/custom opponents retain their existing serial behavior. Batching changes execution, not the PPO update, reward, snapshot selection or number of training transitions.

Only one player trajectory per environment supplies PPO data; its seat rotates between games. This avoids mixing frozen-opponent actions into on-policy updates. No changes to PPO/GAE or network size are introduced by this prototype.

Evaluation references are fixed for the whole run: the learner's random initial model and an independently initialized neural model. The second model is excluded from the training pool. For each evaluation seed, the plan includes both homogeneous pairs and both mixed orders, with the learner in every seat: 12 games per seed block. `--eval-games-per-matchup` is the number of these seed blocks in self-play. Stochastic, locally seeded evaluation is the default. Evaluation happens before training, at the configured interval and at completion. `best.pt` maximizes the win share on this unchanged reference plan and can remain the initial policy if no evaluated update improves it.

These initial opponents measure acquisition of basic useful play. They do not establish strength against trained opponents. Later holdout comparisons can use frozen trained checkpoints from the same run.

## Short pilot

Use fresh output directories for two independent training seeds:

```sh
.venv/bin/python scripts/ppo_training/train_unified.py --mode selfplay --seed 7 --total-timesteps 8192 --n-envs 2 --rollout-horizon 128 --snapshot-every 1024 --eval-every 4096 --eval-games-per-matchup 6 --out-dir runs/poc-seed7
.venv/bin/python scripts/ppo_training/train_unified.py --mode selfplay --seed 19 --total-timesteps 8192 --n-envs 2 --rollout-horizon 128 --snapshot-every 1024 --eval-every 4096 --eval-games-per-matchup 6 --out-dir runs/poc-seed19
```

`opponents.jsonl` logs each started game, including seed, learner seat, selection step, both checkpoint paths and SHA-256 hashes. `episodes.jsonl` repeats those identities alongside completed outcomes. `updates.jsonl` contains finite-loss checks and optimizer diagnostics. `eval_0_summary.json` is the pre-training result; later summaries add paired changes in VP margin and win share, with bootstrap intervals over complete engine-seed blocks. Seat rotations and repeated reference pairings are correlated and are not treated as independent samples. `references.json` fixes reference identities. Initial scores need not be exactly one third against independently initialized models.

Inspect completion counts and errors first. Changing weights or decreasing loss alone do not establish better play. Look for improved results over the unchanged initial policy, reasonably consistent across independent training seeds. Intervals from a few seed blocks are approximate; use the separate holdout before treating a selected checkpoint as evidence.

## Holdout and trained opponents

Use test seeds that were not used for model selection. Evaluate a specified final checkpoint rather than choosing among checkpoints on the holdout:

```sh
.venv/bin/python scripts/eval_selfplay.py --run-dir runs/poc-seed7 --blocks 12 --seed 9001 --out-dir runs/poc-seed7-holdout
.venv/bin/python scripts/eval_selfplay.py --run-dir runs/poc-seed7 --reference runs/poc-seed7/checkpoint_2048.pt runs/poc-seed7/checkpoint_4096.pt --blocks 12 --seed 19001 --out-dir runs/poc-seed7-trained-opponents
```

The tool plays both the final candidate and the original initial model against the same reference plan and reports paired improvement. `candidate.win_share` is also useful against trained opponents; roughly one third is the symmetry benchmark when all policies are equivalent, using shared credit for tied top VP totals. VP ties follow the existing simulator/evaluator objective, not a newly implemented edition-specific tie breaker. Comparison against weak initial opponents alone does not prove robust strategy or long-term improvement.

## Moderate next run

Once the pilot is numerically stable, strictly neural and shows credible improvement on separate seeds, one moderate run is a useful next investment:

```sh
.venv/bin/python scripts/ppo_training/train_unified.py --mode selfplay --seed 31 --total-timesteps 262144 --n-envs 2 --rollout-horizon 256 --snapshot-every 4096 --eval-every 32768 --eval-games-per-matchup 12 --out-dir runs/poc-moderate-seed31
.venv/bin/python scripts/eval_selfplay.py --run-dir runs/poc-moderate-seed31 --reference runs/poc-moderate-seed31/checkpoint_65536.pt runs/poc-moderate-seed31/checkpoint_131072.pt --blocks 40 --seed 29001 --out-dir runs/poc-moderate-seed31-holdout
```

Estimate duration from the pilot's `result.json`, allowing for evaluation costs. A successful moderate run should improve against earlier trained opponents as well as initial references and complete without technical failures. It is evidence for continued investment, not proof of optimal play. No exact training resume exists yet; choose a bounded uninterrupted budget. The active pool is bounded but checkpoint files remain on disk for same-run comparisons.

If learning remains flat, check observed action diversity, entropy, reward variance, paired intervals and seat-specific outcomes before increasing runtime or changing the simulator. Wunder-rule migration, recurrent networks, adaptive leagues and reward redesign are subsequent experiments, not prerequisites added to this prototype.
