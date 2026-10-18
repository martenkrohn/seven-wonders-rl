import numpy as np
import pytest
from pathlib import Path

from sevenwonders_rl.envs.sevenwonders_env import SevenWondersEnv


pytestmark = pytest.mark.integration


def _make_env(tmp_path, **kwargs) -> SevenWondersEnv:
    # Passing repo_root as the python/ folder is fine; the env auto-detects the real repo root.
    repo_root = Path(__file__).resolve().parents[1]  # .../python
    io_dir = tmp_path / "io"
    return SevenWondersEnv(
        repo_root=str(repo_root),
        io_dir=str(io_dir),  # absolute, isolated per test
        **kwargs,
    )


def _pick_legal_action(obs: dict) -> int:
    mask = np.asarray(obs["action_mask"]).astype(int)
    legal = np.flatnonzero(mask == 1)
    assert len(legal) > 0, "No legal actions in mask"
    return int(legal[0])


def test_reset_returns_valid_obs(require_engine, tmp_path):
    env = _make_env(tmp_path)
    try:
        obs, info = env.reset(seed=123)

        assert isinstance(obs, dict), f"obs must be dict, got {type(obs).__name__}"
        assert isinstance(info, dict), f"info must be dict, got {type(info).__name__}"

        # Minimal contract checks
        for k in ("action_mask", "game_meta", "hand_ids", "global_table"):
            assert k in obs, f"obs missing key: {k}"

        assert env.action_space.n == 28, f"expected action_space.n == 28, got {env.action_space.n}"
        assert np.asarray(obs["action_mask"]).shape == (28,), "action_mask must be shape (28,)"
        assert np.asarray(obs["game_meta"]).shape == (3,), "game_meta must be shape (3,)"
    finally:
        env.close()


def test_step_advances_turn_and_does_not_hang(require_engine, tmp_path):
    env = _make_env(tmp_path)
    try:
        obs, _ = env.reset(seed=0)
        prev_turn = int(np.asarray(obs["game_meta"])[1])

        # Run a few steps, always choosing a legal action from the mask.
        for _ in range(10):
            action = _pick_legal_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            assert isinstance(reward, (int, float, np.floating)), f"reward must be numeric, got {type(reward).__name__}"
            assert isinstance(terminated, (bool, np.bool_)), f"terminated must be bool, got {type(terminated).__name__}"
            assert isinstance(truncated, (bool, np.bool_)), f"truncated must be bool, got {type(truncated).__name__}"
            assert isinstance(info, dict), f"info must be dict, got {type(info).__name__}"

            turn = int(np.asarray(obs["game_meta"])[1])
            assert turn > prev_turn, f"turn did not advance: prev={prev_turn} now={turn}"
            prev_turn = turn

            if terminated or truncated:
                break
    finally:
        env.close()


def test_close_terminates_engine_process(require_engine, tmp_path):
    env = _make_env(tmp_path)
    env.reset(seed=0)

    proc = getattr(env, "proc", None)
    assert proc is not None, "env.proc should exist after reset()"

    env.close()

    # After close, process should be terminated (or proc cleared).
    proc2 = getattr(env, "proc", None)
    if proc2 is not None:
        assert proc2.poll() is not None, "engine subprocess still running after env.close()"


def test_invalid_action_fallback_raise(require_engine, tmp_path):
    env = _make_env(tmp_path, invalid_action_fallback="raise")
    try:
        obs, _ = env.reset(seed=0)
        mask = np.asarray(obs["action_mask"]).astype(int)

        illegal_idxs = np.flatnonzero(mask == 0)
        if len(illegal_idxs) > 0:
            illegal = int(illegal_idxs[0])
        else:
            # Extremely unlikely, but keep the test deterministic:
            # use out-of-range action, which must error regardless.
            illegal = 28

        with pytest.raises(Exception):
            env.step(illegal)
    finally:
        env.close()
