# First multi-head experiment: results, 26 October 2026

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

The run improved score margin against its mixed opponent population. Heads 2 and
3 show more useful behaviour than heads 0 and 1, but broad strategic exploration
and reliable grey-resource investment have not emerged. Improvement is largest
against science-heavy tables; standard-only performance is essentially unchanged.

## Audit and scope

- Completed 262,144 additional transitions, 512 updates, 12,480 complete training
  games and eight unfinished games at the budget boundary. Elapsed 23m22s,
  187 transitions/s including initialization, saving and in-training evaluations.
  This is not a controlled speed comparison with earlier runs.
- All four initial actor/critic pairs and shared encoders exactly match the
  standard baseline checkpoint at source step 524,288. Fresh optimizer/counters.
- Four heads; ordinary reward; prior strength zero. Prior weights remain identical.
  Per-head training transitions: 65,612 / 67,219 / 64,457 / 64,856. No head starvation.
- All recorded opponents are copied fixed references with matching source/copy
  SHA-256 hashes; no learner snapshots enter training. Among 24,976 opponent-seat
  selections: 16,474 standard, 4,264 science, 4,238 military, close to the intended mix.
- No truncated updates, nonfinite scalar training metrics or failed evaluation
  games. The 2,304 holdout games have complete final states, age sequences and
  executed action traces; no score-sum inconsistencies or unknown built cards.
- Holdout: eight engine-seed blocks, all 16 ordered opponent pairs, all three
  learner seats, 384 games per policy. These are familiar opponents on new seeds,
  not a test against unseen strategy families. Equal matchup coverage differs
  from the weighted training distribution.

## Performance on the common holdout

Win share splits ties between tied winners. Margin is own VP minus the best
opponent's VP. All values below use the same opponents, seats and engine seeds.

| Policy | Win share | Mean VP | Mean margin | Science VP | Military VP | Blue VP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Starting standard policy | 31.6% | 42.68 | -5.27 | 2.96 | 8.89 | 15.76 |
| Final, random head per game | 37.4% | 45.33 | -2.52 | 4.60 | 3.60 | 21.34 |
| Head 0 | 36.0% | 44.55 | -2.95 | 4.48 | 3.61 | 20.62 |
| Head 1 | 32.9% | 43.92 | -3.78 | 3.93 | 3.85 | 19.87 |
| Head 2 | 40.2% | 46.37 | -1.52 | 5.16 | 3.23 | 22.03 |
| Head 3 | 40.2% | 45.80 | -1.96 | 4.61 | 3.73 | 22.61 |

The deployed random-head policy gains **2.75 margin points**. Paired bootstrap
95% interval: **[1.16, 3.97]**. Win-share gain is **5.73 percentage points**, with
interval **[-0.39, 11.33] percentage points**. The margin improvement is clearer
than the win-share improvement.

Bootstrap resamples whole engine-seed blocks, 10,000 samples and fixed diagnostic
seed 0. Intervals are exploratory: only eight independent engine-seed blocks and
one training seed. Existing dashboard intervals use 2,000 bootstrap samples and
therefore differ slightly in endpoints.

Heads 2 and 3 are the most promising candidates on this holdout. Head 2's margin
advantage over head 3 is 0.45 points, interval [-0.10, 1.04]; their win shares are
identical. Selection after inspecting all heads is exploratory and needs fresh
validation before claiming one is definitively strongest.

## Which opponent situations improved?

| Opponent groups, ordered left / right | Initial share | Final random-head share | Initial margin | Final margin |
| --- | ---: | ---: | ---: | ---: |
| Standard / standard | 26.6% | 25.0% | -5.55 | -5.98 |
| Standard / science | 11.5% | 42.0% | -11.17 | -2.25 |
| Science / standard | 19.8% | 31.2% | -8.56 | -5.79 |
| Science / science | 62.5% | 100.0% | 2.00 | 20.13 |
| Standard / military | 39.6% | 35.4% | -3.23 | -3.83 |
| Military / standard | 37.5% | 31.2% | -4.10 | -3.46 |
| Military / science | 45.8% | 60.4% | -2.71 | 2.96 |
| Science / military | 37.5% | 43.1% | -4.08 | -2.58 |
| Military / military | 37.5% | 14.6% | -3.13 | -6.17 |

Standard/standard has 96 games, mixed standard/specialist groups have 48 each,
and each specialist/specialist ordered group has 24. These are repeated matchups
within eight seed blocks, not that many independent random board layouts.

The double-science matchup contributes 1.13 points (41%) of the overall +2.75
margin gain, despite being only 1/16 of the evenly covered matchup plan. However,
excluding it still yields **+1.73 margin**, interval **[0.17, 2.91]**, across the
remaining 360 games. The result is not entirely explained by this one matchup.

Reweighting all matchups to the training population gives a smaller **+1.73 margin**
interval [0.42, 2.75] and **+3.78 percentage points win share**, interval [-1.74, 8.85].
This is an offline reweighting of existing evaluations, not additional games.
Against standard-only tables the random policy gains -0.43 margin, interval
[-1.14, 0.25]; this provides no evidence of improvement there.

The large category shift is away from military (-5.29 VP) toward blue cards
(+5.58 VP), with a modest science increase (+1.64 VP). That is consistent with
learning to exploit science-heavy opponents and conceding military more often.
It is not evidence of discovering a strong science strategy itself. The weaker
military/military result shows a cost of this shift.

## Resource investment and discards

Numbers are per game. Grey builds count acknowledged structure/free builds in
Ages I and II; wonder builds and discarded grey cards do not count.

| Policy | Grey builds I+II | Discards I | Discards II | Discards III | III discards without legal card build |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 0.042 | 0.216 | 0.417 | 0.865 | 0.701 |
| Random head | 0.096 | 0.245 | 0.500 | 1.016 | 0.617 |
| Head 0 | 0.013 | 0.299 | 0.737 | 1.323 | 0.714 |
| Head 1 | 0.008 | 0.266 | 0.797 | 1.430 | 0.682 |
| Head 2 | 0.161 | 0.172 | 0.237 | 0.727 | 0.622 |
| Head 3 | 0.128 | 0.219 | 0.302 | 0.766 | 0.622 |

There is a real behavioural split. Heads 2/3 make more investments and discard
less; heads 0/1 almost never build grey and frequently discard despite card-building
alternatives. At game end heads 0/1 hold 14.73/15.73 coins on average, compared
with 11.82/11.65 for heads 2/3 and 12.77 initially.

Nevertheless, head 2 builds only **62 grey cards in 384 games**, across both early
ages combined. Heads 0 and 1 build zero grey cards in Age I. This is far from
reliable early manufactured-resource investment. Some end-of-age production comes
from starting wonders and other effects; grey-card counts are not total resource
access, and buying from neighbours can be rational.

The random-head policy's Age III decisions with a legal card build rise only from
86.85% to 88.06%. Its discards without such alternatives fall from 0.701 to 0.617
per game, but total Age III discards **increase** from 0.865 to 1.016. Of those
late discards, 153/390 have a legal card-build alternative, versus 63/332 initially.
A legal alternative does not prove that discarding was wrong; no card-build
alternative does not exclude a possible wonder action. Resource insufficiency is
still a hypothesis, not a fully diagnosed cause for each discard.

## Did the heads stay different?

They are not identical, but there are no clear science/military/resource roles.
Last-quarter mean JS divergence on shared rollout states is 0.0189 nats, versus
0.0037 in the first quarter. The corresponding disagreement diagnostic rises
from 1.13% to 3.59% (it compares each head against head 0 and includes head 0's
zero disagreement). Individual final batches fluctuate, so their last value
should not be treated as the overall level of diversity.

Random evaluation selects a head using engine seed and seat; each selection is
repeated across that seed/seat's 16 opponent pairings. There are only 24 independent
head-selection contexts. Observed counts are 48/128/112/96 games for heads 0/1/2/3.
Consequently, this random-policy estimate is not exactly an equal arithmetic
average of the four separately evaluated heads. All heads have equal full-plan
coverage in their individual evaluations.

## Interpretation and next experiment

Useful partial progress: the population produces different pressures, the heads
have begun to diverge, and two heads show better investment/discard behaviour.
The targeted discovery problem remains: early grey production and substantial
science investment are still rare, and standard-only performance has not improved.
No equal-budget single-head control was run, so the gains cannot be attributed
separately to heads, more training, or the new opponent population.

Keep the current checkpoints and population as references. A reasonable next
single experiment is the same standard starting checkpoint, four heads, same
opponents and budget, with frozen priors enabled after a brief check that the
chosen strength actually changes legal action preferences without overwhelming
learned preferences. Starting again from the same standard model would make the
comparison easier to interpret than changing the initialization too. This remains
an experimental PPO exploration mechanism, with no guaranteed benefit.
No new training or evaluation games were launched for this analysis.

## Reproduction and artifacts

- Offline audit: `python/runs/multihead-v1-analysis-20261026/analyze.py`.
- Full audited metrics: `python/runs/multihead-v1-analysis-20261026/analysis.json`.
- Original report: `python/runs/multihead-v1-seed31-holdout/dashboard.html`.
- Original raw games: `initial.jsonl`, `candidate.jsonl`, and `head_0.jsonl` through
  `head_3.jsonl` in that holdout directory.

Run the offline audit from `python/` with:

```bash
.venv/bin/python runs/multihead-v1-analysis-20261026/analyze.py
```
