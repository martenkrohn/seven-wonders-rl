# Next experiment: four heads with calibrated frozen priors

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Prepared on 26 October 2026 after inspecting the first multi-head run.
The full run is now complete. See [the audited results](MULTIHEAD_PRIORS_RESULTS_2026-10-26.md).

From `python/`, run:

```bash
bash scripts/ppo_training/train_and_evaluate_priors_v2.sh
```

This executes one training run, evaluates the starting and final policies plus
each final head, then compares the results with v1 using saved games.

## Fixed settings

- Original standard starting checkpoint: the byte-identical copy retained at
  `runs/multihead-v1-seed31/frozen_opponents/opponent_000.pt`, SHA-256
  `c81d68ebdf9ec7f4f3cd21ca35014809b612872a0a65ac202818caa81a5fd302`.
  This starts again from the standard model, not the final v1 learner.
- Four heads, same model/training seed 31. The initial learned weights and random
  prior weights match v1 exactly; **prior strength changes from 0 to 32**.
- Same optimizer defaults, reward, eight environments, horizon 64, and **262,144
  total additional transitions**, shared across the heads.
- Same fixed opponent population and weights: standard baseline and lambda-1
  models, each 1/3 per opponent seat; science and military specialists, each 1/6.
  The new population config references v1's fixed copies to avoid changes to
  mutable `latest.pt` files. Both seats sample independently.
- The priors remain frozen; their contribution is included in both collection
  probabilities and PPO likelihood recomputation. Head selection is once per game.

The launcher verifies the calibrated checkpoint hash, population hashes/weights,
and selected strength before training. Existing output directories are preserved.

## Calibration evidence

`scripts/ppo_training/calibrate_priors.py` collected **96 games / 1,728 executed
decision states**, with the prior-free starting policy against all ordered
opponent pairs and all three learner seats. Two engine-seed blocks use seed
91001, separate from training/model-selection/holdout streams. No weights were
updated. Every strength was evaluated on the same states, using the exact frozen
prior networks that will initialize v2.

| Strength | Mean total variation from baseline | Preferred action changes | Head JS divergence |
| --- | ---: | ---: | ---: |
| 0.5 | 0.14% | 0.17% | 0.00004 |
| 2 | 0.54% | 0.54% | 0.00064 |
| 8 | 2.16% | 1.84% | 0.00968 |
| 16 | 4.28% | 3.82% | 0.03226 |
| **32** | **8.19%** | **7.70%** | **0.08885** |

Total variation is half the sum of absolute changes in action probabilities,
averaged over heads and sampled states; it measures probability mass shifted.
Strength 32 is a visible but limited perturbation in this calibration. Across
individual heads, mean variation ranges from 5.91% to 13.22%; preferred-action
changes range from 4.80% to 14.12%. The 90th percentile of head/state variation
is 28.60%, so the effect is larger on some states.

The number 32 is specific to this network's output scale. Smaller coefficients
barely affect its existing preferences. It is an engineering choice based on
perturbation, **not** a claim of optimal strength. No game rewards, wins, or
science scores were used to select it.

The sampled baseline gives grey builds only 0.235% probability on early states
where they are legal. The four-head average with strength 32 is 1.032%, with
different directions per head (one suppresses grey further). This was not the
selection objective and remains low: the priors are not a grey-card bonus and do
not guarantee resource-strategy discovery. Entropy changes slightly downward
(0.355 to 0.347 nats); these are persistent preference differences, not a uniform
increase in random action selection.

States were visited by the baseline, not by the perturbed policy. Two seed blocks
are limited coverage; perturbations after the learner visits new states or trains
may differ. This calibration does not establish better play or stable long-run
learning.

Calibration outputs are in `python/runs/multihead-prior-calibration-20261026/`:
`summary.json`, `selection.json`, `games.jsonl`, and compressed raw `states.npz`.
The saved states allow further strength analysis without collecting more games.

## Evaluation and comparison

Outputs:

- `runs/multihead-v2-priors-seed31/`: training, checkpoints and in-training checks.
- `runs/multihead-v2-priors-seed31-holdout/`: the dashboard, per-head results and
  `exploration.json`.
- `comparison-to-v1.md` / `comparison-to-v1.json` in that holdout directory:
  paired comparisons with the original prior-free source policy and v1's final
  random-head policy, including standard-only tables. These reuse v1's saved
  evaluation games and do not add simulations.

The new run's `initial` checkpoint includes the priors before any learning, so
its own before/after comparison measures improvement from that perturbed starting
policy. **Use `comparison-to-v1` for comparison with the original standard policy
and the previous experiment.** The dashboard labels the prior-bearing start.

Evaluation retains seed 89001, eight blocks, all 16 opponent pairs and all seats:
384 games each for the initial, final random-head and four individual heads.
These seeds have already been inspected in v1; this is a matched exploratory
comparison, not fresh confirmatory holdout evidence. Opponent identities are
matched by checkpoint hashes, since copied reference paths differ between runs.
Eight blocks and one training seed remain substantial uncertainty limits.

Judge success by improved actual score margins and win share across tables,
especially standard-only tables, together with resource investment and fewer
late discards without card-building alternatives. Diversity alone is insufficient.

## Validation

Nine engine-free calibration/multi-head checks pass. A short real-model check at
strength 32 completed 512 training transitions and 48 evaluation games without
failures. Its initial state dict matches v1's initial weights exactly, and the
prior weights remain unchanged after optimization. A saved-data comparison check
returned zero differences when comparing identical policies under relocated paths.
The shell launcher syntax and hash preflight pass. No full v2 run was launched.
