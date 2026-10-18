"""Bot policies and opponent strategies."""
from .specialist_policies import (
    BaselinePolicy,
    MilitaryTempoPolicy,
    MadScientistPolicy,
)
from .random_policy import RandomPolicy
from .registry import make_policy, KNOWN_POLICIES
from .opponent_sampler import OpponentSampler

# For backward compatibility
HeuristicPolicy = BaselinePolicy
ScienceSetPolicy = MadScientistPolicy

__all__ = [
    "BaselinePolicy",
    "HeuristicPolicy",
    "MilitaryTempoPolicy",
    "MadScientistPolicy",
    "ScienceSetPolicy",
    "RandomPolicy",
    "make_policy",
    "KNOWN_POLICIES",
    "OpponentSampler",
]
