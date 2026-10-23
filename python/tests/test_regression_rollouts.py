import numpy as np
import pytest
from pathlib import Path

from sevenwonders_rl.envs.sevenwonders_env import SevenWondersEnv


pytestmark = pytest.mark.integration


def _make_env(tmp_path, seed: int) -> SevenWondersEnv:
    repo_root = Path(__file__).resolve().parents[1]  # .../python
    io_dir = tmp_path / f"seed_{seed}" / "io"
    env = SevenWondersEnv(
        repo_root=str(repo_root),
        io_dir=str(io_dir),
        seed=seed,
    )
    original_read = env._read_status_blocking
    env._read_status_blocking = lambda timeout=15.0, _orig=original_read: _orig(timeout=15.0)
    return env


def _pick_legal_action(obs: dict) -> int:
    mask = np.asarray(obs["action_mask"]).astype(int)
    legal = np.flatnonzero(mask == 1)
    assert len(legal) > 0, "No legal actions in mask"
    # deterministic: choose the lowest-index legal action
    return int(legal[0])


def _run_episode(env: SevenWondersEnv, max_steps: int = 200) -> dict:
    obs, info = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        action = _pick_legal_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        assert np.isfinite(total_reward), "total_reward became non-finite"
        assert np.isfinite(np.asarray(obs["global_table"])).all(), "global_table contains NaN/inf"
        assert np.isfinite(np.asarray(obs["game_meta"])).all(), "game_meta contains NaN/inf"

        if terminated or truncated:
            return {
                "steps": step + 1,
                "total_reward": total_reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "final_game_meta": np.asarray(obs["game_meta"]).astype(int).tolist(),
                "final_global_table": np.asarray(obs["global_table"]).astype(float),
            }

    raise AssertionError(f"Episode did not finish within max_steps={max_steps}")


def test_rollout_smoke_n_games(require_engine, tmp_path):
    # Keep it small; this is just “system still runs”.
    seeds = [0, 1, 2, 3, 4]

    summaries = []
    for s in seeds:
        env = _make_env(tmp_path, seed=s)
        try:
            summ = _run_episode(env, max_steps=250)
            summaries.append(summ)
        finally:
            env.close()

    # Very light invariants across episodes
    for summ in summaries:
        assert summ["terminated"] is True and summ["truncated"] is False
        era, turn, clockwise = summ["final_game_meta"]
        assert era == 4
        assert turn == 21
        assert clockwise in (0, 1)

        # Coins should be non-negative for all 3 players at end.
        final_gt = summ["final_global_table"]
        coins = final_gt[:, 0]
        assert (coins >= 0).all(), f"coins negative at end: {coins}"
