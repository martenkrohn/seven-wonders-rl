"""PFSP-inspired matchmaking over ordered (left, right) frozen-policy pairs.

Estimates are discounted online training outcomes, averaged over learner heads,
seats, wonders and sampled opponent heads. They are not exact current-policy
payoffs or a multiplayer equilibrium solver. No evaluation outcomes enter here.
"""
from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np

from sevenwonders_rl.selfplay.opponent_pool import OpponentPool


@dataclass
class PairPFSPPool:
    league: OpponentPool
    power: float = 2.0
    control_fraction: float = .10
    half_life: float = 131_072
    prior_games: float = 3.0
    _results: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if (not all(math.isfinite(x) for x in
                    (self.power, self.control_fraction, self.half_life, self.prior_games))
                or min(self.power, self.half_life, self.prior_games) <= 0
                or not 0 < self.control_fraction <= 1):
            raise ValueError('Invalid pair-PFSP parameters')

    @property
    def max_size(self):
        return self.league.max_size

    def register(self, path):
        self.league.register(path)

    def eligible(self, current_step):
        # Also exclude snapshots left beyond the durable checkpoint by a crash.
        items = [(s, p) for s, p in self.league._list_checkpoints()
                 if s <= current_step and (s == 0 or s <= current_step - self.league.min_step_lag)]
        return items[-self.max_size:] if self.max_size > 0 else items

    def pool_size(self, current_step):
        return len(self.eligible(current_step))

    def total_size(self):
        return self.league.total_size()

    def distribution(self, current_step):
        items = self.eligible(current_step)
        if not items:
            raise ValueError('Pair-PFSP requires its published checkpoint_0.pt')
        pairs = [(a, b) for a in items for b in items]
        rows = [self._results.get((a[0], b[0])) for a, b in pairs]
        effective = np.array([r['count'] if r else 0. for r in rows])
        scores = np.array([r['score'] if r else 0. for r in rows])
        last_steps = np.array([r['step'] if r else current_step for r in rows])
        if np.any(last_steps > current_step):
            raise ValueError('Pair-PFSP statistics are ahead of the learner')
        decay = np.exp2(-(current_step - last_steps) / self.half_life)
        effective *= decay
        scores *= decay
        # Exchangeable three-player neutral start: expected win share = 1/3.
        rates = (scores + self.prior_games / 3.) / (effective + self.prior_games)
        weights = np.power(1. - rates, self.power)
        hard = weights / weights.sum() if weights.sum() > 0 else np.full(len(pairs), 1. / len(pairs))
        probabilities = (1. - self.control_fraction) * hard + self.control_fraction / len(pairs)
        return pairs, rates, effective, hard, probabilities

    def sample_pair(self, rng, current_step):
        pairs, rates, effective, hard, probabilities = self.distribution(current_step)
        control = bool(rng.random() < self.control_fraction)
        index = int(rng.integers(len(pairs)) if control else rng.choice(len(pairs), p=hard))
        left, right = pairs[index]
        row = self._results.get((left[0], right[0]))
        metadata = dict(group='pair_pfsp', pair_left_step=left[0], pair_right_step=right[0],
            pair_probability=float(probabilities[index]),
            pair_selection='uniform_control' if control else 'pfsp',
            pair_win_share_estimate=float(rates[index]), pair_effective_games=float(effective[index]),
            pair_games=int(row['games']) if row else 0,
            pair_last_result_step=int(row['step']) if row else None)
        return str(left[1]), str(right[1]), metadata

    def record_result(self, left_step, right_step, totals, agent_seat, current_step):
        if len(totals) != 3 or agent_seat not in (0, 1, 2) or not all(math.isfinite(x) for x in totals):
            raise ValueError('Pair-PFSP requires three valid terminal totals')
        if min(left_step, right_step) < 0 or max(left_step, right_step) > current_step:
            raise ValueError('Pair-PFSP outcome references a future snapshot')
        best = max(totals)
        score = 1. / sum(t == best for t in totals) if totals[agent_seat] == best else 0.
        key = (int(left_step), int(right_step))
        old = self._results.get(key, dict(count=0., score=0., games=0, step=current_step))
        if old['step'] > current_step:
            raise ValueError('Pair-PFSP outcome moved backwards in training time')
        decay = math.exp2(-(current_step - old['step']) / self.half_life)
        self._results[key] = dict(count=old['count'] * decay + 1.,
            score=old['score'] * decay + score, games=old['games'] + 1, step=int(current_step))

    def state_dict(self):
        return dict(version=1, parameters=dict(power=self.power, control_fraction=self.control_fraction,
            half_life=self.half_life, prior_games=self.prior_games),
            results=[dict(left=k[0], right=k[1], **r) for k, r in sorted(self._results.items())])

    def load_state_dict(self, state):
        if state.get('version') != 1 or state.get('parameters') != self.state_dict()['parameters']:
            raise ValueError('Incompatible pair-PFSP checkpoint state')
        restored = {}
        for row in state['results']:
            key = (row['left'], row['right'])
            if (key in restored or min(key) < 0 or row['step'] < max(key)
                    or row['games'] < 1 or not 0 <= row['score'] <= row['count'] <= row['games']
                    or not all(math.isfinite(row[k]) for k in ('score', 'count', 'step', 'games'))):
                raise ValueError('Invalid pair-PFSP result state')
            restored[key] = {k: row[k] for k in ('score', 'count', 'step', 'games')}
        self._results = restored

    def report(self, current_step):
        pairs, rates, effective, _, probabilities = self.distribution(current_step)
        return dict(timestep=current_step, method='ordered_pair_pfsp',
            metric='terminal_win_share', estimates='discounted online training results, not fixed-policy evaluations',
            **self.state_dict(),
            active_checkpoints=[dict(step=s, checkpoint=str(Path(p).resolve())) for s, p in self.eligible(current_step)],
            pairs=[dict(left=a[0], right=b[0], win_share=float(rate), effective_games=float(n),
                        probability=float(prob))
                   for (a, b), rate, n, prob in zip(pairs, rates, effective, probabilities)])
