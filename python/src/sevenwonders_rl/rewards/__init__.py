"""Versioned, terminal-only training objectives; evaluation defaults to standard."""
from dataclasses import dataclass
import math

REWARD_PROFILES = ('standard', 'science', 'military')
REWARD_VERSION = 1


@dataclass(frozen=True)
class RewardSpec:
    profile: str = 'standard'
    specialist_weight: float = 1.0
    scale: float = 1.0

    def __post_init__(self):
        if self.profile not in REWARD_PROFILES:
            raise ValueError(f'Unknown reward profile: {self.profile}')
        if not math.isfinite(self.specialist_weight) or self.specialist_weight < 0:
            raise ValueError('specialist_weight must be finite and nonnegative')
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('reward scale must be finite and positive')

    def describe(self):
        category = {'science': 'scientific', 'military': 'military'}.get(self.profile)
        return dict(version=REWARD_VERSION, profile=self.profile, timing='finished_only',
                    category=category, specialist_weight=self.specialist_weight if category else 0.0,
                    scale=self.scale,
                    formula='(own_vp - best_opponent_vp + specialist_weight * category_points) / scale')

    def terminal_components(self, status, seat):
        if status.get('game', {}).get('phase') != 'finished':
            raise ValueError('Terminal reward requires a completed engine state')
        if seat not in (0, 1, 2):
            raise ValueError('Invalid reward seat')
        points = [status['players'][str(s)]['points'] for s in range(3)]
        margin = float(int(points[seat]['total']) - max(int(p['total']) for s, p in enumerate(points) if s != seat))
        description = self.describe()
        category = description['category']
        category_points = float(points[seat][category]) if category else 0.0
        raw_bonus = description['specialist_weight'] * category_points
        return dict(reward_profile=self.profile, raw_vp_diff=margin,
                    category=category, specialist_points=category_points,
                    specialist_weight=description['specialist_weight'], raw_specialist_bonus=raw_bonus,
                    base_reward=margin / self.scale, specialist_bonus=raw_bonus / self.scale,
                    training_reward=(margin + raw_bonus) / self.scale)
