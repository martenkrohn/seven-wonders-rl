# One multi-head exploration experiment

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Run from `python/`:

```bash
.venv/bin/python scripts/ppo_training/run_multihead_experiment.py
```

This starts **one** training run, followed by evaluation. It does not start
single-head control runs, prior experiments, or a sequence of training variants.
Existing output directories are preserved; use a new `--out-dir` for another run.
Use `--dry-run` to validate checkpoints and print the commands without starting.

## First experiment

- Four actor/critic pairs with shared state/action encoders.
- Start from `runs/science-diagnostic-v1-baseline-seed31/latest.pt`.
  Copy its encoders and actor/critic into all heads. Fresh optimizer and counters.
- Select a head uniformly once per game, separately for each environment. Keep
  it across collection batches and PPO updates until that game ends. A head's
  weights can still change during an ongoing game; this is not a frozen plan.
- Ordinary terminal VP-margin reward, unchanged gamma/lambda/entropy settings.
- **262,144 additional transitions total**, shared across all heads (roughly
  65,536 each, with random variation). Eight environments, horizon 64.
- Frozen random priors implemented but **disabled (`--prior-strength 0`)**.
- A fixed opponent population; learner snapshots never enter the training pool.

The population is `python/configs/multihead_v1_opponents.json`. Each opponent
seat independently selects a group by weight and a checkpoint uniformly within
the group. Its initial weights give:

| Opponents | Per-seat probability |
| --- | ---: |
| Standard baseline final checkpoint | 1/3 |
| Standard-reward lambda-1.0 final checkpoint | 1/3 |
| Science specialist final checkpoint | 1/6 |
| Military specialist final checkpoint | 1/6 |

The two standard models have measured differences in early discards and later
building; they are related training runs, not independent strategy families.
The military specialist is included to create military pressure despite its
lower overall strength. The mix is an experimental starting point.

Population JSON paths are relative to the trainer's working directory (`python/`
for this driver). Group weights do not depend on how many checkpoints a group
contains. Sources are copied into the run, hash-checked and recorded with their
group and weight. Changing the population requires a new experiment directory.

## Training correctness and compatibility

Rollouts store head IDs alongside old probabilities and values. PPO recomputes
probabilities with the same actor, and GAE bootstraps with the same critic before
any episode reset. Each private head learns from its own trajectories; encoders
learn from all heads. No cross-head off-policy PPO updates or bootstrap masks.

The multi-head architecture has version 3; existing version-2 single-head models
remain supported. Single-head weights expand only through explicit
`--num-heads 4` initialization. Multi-head initialization requires matching head
count and prior strength. This is a fresh optimization run, not exact resume.

Cached opponent models have no mutable current-head field. The environment owns
each opponent seat's head assignment, with a separate reproducible RNG stream.
Batched and serial opponents use the same assignments. Checkpoints include all
heads and frozen priors. Explicit head IDs are required for multi-head inference.

With priors enabled, action logits are `learned_logits + strength * prior` before
the legal-action mask. Each prior reads raw state and action features through its
own frozen 32-unit tanh network, with bounded output. It does not depend on the
changing shared encoders. Prior strength is fixed throughout a run and uncalibrated;
it should be chosen after inspecting results, not automatically enabled next.

## Evaluation and decisions afterward

Training checks the random-head policy every 65,536 transitions on all 16 ordered
opponent pairs and all three learner seats: 48 games per check, fixed validation
seeds. Checkpoints and timing logs are retained.

Final evaluation uses eight new engine-seed blocks (seed 89001), all 16 pairs and
all seats: **384 games per evaluated policy**, six policies / 2,304 games total:
starting model, final random-head policy, and four individual final heads.
The random head is selected once per game even with greedy action selection.
Matchups have equal coverage, rather than the weighted training distribution.
Opponents are familiar; these are unseen seeds, not unseen opponent strategies.

Outputs:

- `runs/multihead-v1-seed31/`: weights, config, episodes with head/opponent
  identities, timing, and per-update head sample counts and policy disagreement.
  `head_js_divergence` compares all heads on the same 64 rollout states (in nats),
  and `head_greedy_disagreement` compares choices against head 0. These diagnostics
  measure policy differences, not useful strategy discovery.
- `runs/multihead-v1-seed31-holdout/dashboard.html`: model/head selection,
  scores, cards, resources by age and opponent matchup comparisons.
- `exploration.json` in that holdout directory: paired changes against the
  starting model, opponent-group results, early grey-card builds, Age III legal
  card-building opportunities, and discards without card-building alternatives.
  A missing card-building alternative does not exclude a possible wonder action.

Performance is assessed using score margin and win share, with paired uncertainty
over engine-seed blocks. Behavioural differences and fewer discards alone are
not success. Without an equal-budget single-head control, improvement cannot be
attributed separately to heads, extra training or opponent diversity. One seed
and a moderate budget also cannot establish that a failed experiment disproves
the approach.

If heads remain similar, consider a later run with priors. If they differ but
perform poorly, inspect which matchups/investments fail before changing the
exploration mechanism. No further experiment starts automatically.

To change only the budget and evaluation size:

```bash
.venv/bin/python scripts/ppo_training/run_multihead_experiment.py \
  --total-timesteps 131072 --blocks 4 --out-dir runs/multihead-v1-short-seed31
```

To evaluate an already completed run separately:

```bash
.venv/bin/python scripts/ppo_training/run_multihead_experiment.py \
  --phase evaluate --out-dir runs/multihead-v1-seed31
```

## Implementation validation

Eight focused tests cover warm-start equivalence, checkpoint compatibility,
selected-head gradients, frozen priors, PPO likelihoods, episode-boundary
bootstrap, population weights, and real-engine training/evaluation. Serial and
batched opponent paths agree, including per-game head choices. Existing targeted
training, checkpoint, evaluation and report regression checks also pass. The
optional JavaScript language harness is skipped because Node is unavailable.

A complete smoke workflow used the actual trained checkpoints with priors off:
1,024 transitions, 48 in-training evaluation games, and 288 final evaluation games
(48 for each of six policies), all completed. Outputs are temporarily retained
under `/tmp/sevenwonders-multihead-v1-smoke` and its `-holdout` directory.
This verifies execution and reporting, not learning quality or comparative speed.
The full default experiment has not been started.
