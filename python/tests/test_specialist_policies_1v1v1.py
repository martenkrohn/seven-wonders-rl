import itertools
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pytest

from sevenwonders_rl.envs import SevenWondersEnv, SeatPolicy
from sevenwonders_rl.policies import BaselinePolicy, MilitaryTempoPolicy, MadScientistPolicy
from sevenwonders_rl.policies.feature_policy_base import get_action_type
from sevenwonders_rl.policies.specialist_policies import _is_science_action, _is_military_action


POLICY_NAMES = ("baseline", "military", "science")


def _repo_root() -> Path:
    # .../python/tests/test_*.py -> parents[0]=tests, [1]=python, [2]=repo_root
    return Path(__file__).resolve().parents[2]


def _make_policies() -> Dict[str, Any]:
    # Seeds purely to make tie-breaks reproducible inside FeaturePolicy.select_action
    return {
        "baseline": BaselinePolicy(seed=42),
        "military": MilitaryTempoPolicy(seed=43),
        "science": MadScientistPolicy(seed=44),
    }


def _seatpolicy_list(perm: Tuple[str, str, str], policies: Dict[str, Any]) -> list[SeatPolicy]:
    # IMPORTANT: env ignores the agent_seat entry, but we still provide 3 entries.
    return [
        SeatPolicy(perm[0], policy=policies[perm[0]]),
        SeatPolicy(perm[1], policy=policies[perm[1]]),
        SeatPolicy(perm[2], policy=policies[perm[2]]),
    ]


def _assert_agent_hard_bans(agent_name: str, obs: Dict[str, Any], action_idx: int) -> None:
    mask = np.asarray(obs["action_mask"], dtype=np.int8)
    assert 0 <= action_idx < mask.shape[0] == 28
    assert mask[action_idx] == 1, f"Agent chose illegal action_idx={action_idx}"

    feats = np.asarray(obs["action_feats"], dtype=np.float32)
    f = feats[action_idx]
    action_type = int(get_action_type(f))

    if agent_name == "military":
        assert not _is_science_action(f, action_type), "Military policy picked a science action"
    if agent_name == "science":
        assert not _is_military_action(f, action_type), "Science policy picked a military action"


def _run_one_game(
    *,
    perm: Tuple[str, str, str],
    agent_seat: int,
    game_seed: int,
    policies: Dict[str, Any],
    io_dir: Path,
) -> Dict[str, Any]:
    """
    Runs a full 1v1v1 game where all 3 seats are specialist bots,
    but we only assert per-step invariants for the agent seat (because env does not expose opponent actions).

    Returns final info dict from terminal step (includes status/final_totals/raw_vp_diff).
    """
    assert agent_seat in (0, 1, 2)
    repo_root = _repo_root()

    opponent_policies = _seatpolicy_list(perm, policies)

    # Make the agent strict: if *agent* ever sends illegal, raise (should never happen).
    # Opponents are guarded by SeatPolicy's legality check + fallback.
    env = SevenWondersEnv(
        repo_root=str(repo_root),
        engine_path="build/7Wonders",
        io_dir=str(io_dir),
        agent_seat=agent_seat,
        opponent_policies=opponent_policies,
        invalid_action_fallback="raise",
    )

    rng = np.random.default_rng(game_seed)

    try:
        obs, info = env.reset(seed=game_seed)

        terminated = False
        truncated = False
        steps = 0
        max_steps = 40  # hard safety; game should end at turn==20

        agent_name = perm[agent_seat]
        agent_policy = policies[agent_name]

        while not (terminated or truncated):
            steps += 1
            assert steps <= max_steps, "Exceeded max_steps (engine likely stalled)"

            action = int(agent_policy.select_action(obs, rng))
            _assert_agent_hard_bans(agent_name, obs, action)

            obs, reward, terminated, truncated, info = env.step(action)

            assert not truncated, f"Game truncated: {info.get('truncated_reason', '')}"

        # Terminal invariants (per your spec)
        assert terminated
        assert info["turn"] == 20
        assert "final_totals" in info and len(info["final_totals"]) == 3

        totals = info["final_totals"]
        agent_total = int(totals[agent_seat])
        best_opp = max(int(totals[i]) for i in (0, 1, 2) if i != agent_seat)
        expected_diff = float(agent_total - best_opp)

        # Env reports this explicitly
        assert "raw_vp_diff" in info
        assert np.isclose(float(info["raw_vp_diff"]), expected_diff)

        # Reward is defined as raw_vp_diff / reward_scale (reward_scale defaults to 1.0)
        assert np.isclose(float(reward), float(info["raw_vp_diff"]))

        return info

    finally:
        env.close()


@pytest.mark.integration
def test_specialist_policies_1v1v1_all_seats(require_engine, tmp_path: Path) -> None:
    """
    Integration test: run full games with Baseline vs MilitaryTempo vs MadScientist.

    Scheduling:
      - all 6 seat permutations
      - repeated n times
      - for each permutation, run 3 games rotating agent_seat (0,1,2)

    What we assert:
      - every game terminates cleanly (no truncation, turn==20)
      - agent's chosen actions are legal
      - military agent never chooses science
      - science agent never chooses military
      - terminal reward/raw_vp_diff consistency
    """
    n_repeats = int(os.getenv("SEVENWONDERS_TEST_N_REPEATS", "1"))
    base_seed = int(os.getenv("SEVENWONDERS_TEST_BASE_SEED", "1000"))

    policies = _make_policies()
    perms = list(itertools.permutations(POLICY_NAMES, 3))

    # Run schedule
    k = 0
    for perm in perms:
        for rep in range(n_repeats):
            for agent_seat in (0, 1, 2):
                game_seed = base_seed + 1000 * rep + 100 * k + agent_seat
                io_dir = tmp_path / f"perm_{perm[0]}_{perm[1]}_{perm[2]}" / f"rep_{rep}" / f"agent_{agent_seat}" / "io"
                io_dir.mkdir(parents=True, exist_ok=True)

                _run_one_game(
                    perm=perm,
                    agent_seat=agent_seat,
                    game_seed=game_seed,
                    policies=policies,
                    io_dir=io_dir,
                )
            k += 1
