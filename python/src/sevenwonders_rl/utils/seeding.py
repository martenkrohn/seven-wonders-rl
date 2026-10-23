"""Stable, role-separated seeds. No wall clock or process-dependent hash()."""
from __future__ import annotations

import numpy as np

SEED_VERSION = 1
ROLES = {"model": 1, "learner": 2, "minibatch": 3, "engine": 4,
         "opponents": 5, "evaluation": 6, "policy": 7}


def derive_seed(master: int, role: str, *indices: int) -> int:
    return int(np.random.SeedSequence([master, SEED_VERSION, ROLES[role], *indices])
               .generate_state(1, dtype=np.uint64)[0])
