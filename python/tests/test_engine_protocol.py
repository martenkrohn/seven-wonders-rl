import json
import sys
from pathlib import Path

import numpy as np
import pytest

from sevenwonders_rl.envs.sevenwonders_env import SevenWondersEnv, EngineError
from sevenwonders_rl.features.obs_builder import build_obs
from sevenwonders_rl.utils.ppo_utils import flatten_global

pytestmark = pytest.mark.integration
ROOT = Path(__file__).parents[2]


def make_env(path, **kwargs):
    return SevenWondersEnv(repo_root=str(ROOT), io_dir=str(path / "io"), **kwargs)


def play(path, seed, poll_us=1000):
    env = make_env(path, engine_poll_us=poll_us)
    trace, military_before_final = [], None
    try:
        obs, info = env.reset(seed=seed)
        for turn in range(21):
            assert info["status"]["game"]["phase"] == "playing"
            assert info["turn"] == turn
            if turn == 20:
                military_before_final = [info["status"]["players"][str(s)]["points"]["military"] for s in (0,1,2)]
            trace.append(info["status"])
            action = int(np.flatnonzero(obs["action_mask"])[0])
            obs, reward, terminated, truncated, info = env.step(action)
            assert not truncated
            assert terminated == (turn == 20)
        final = info["status"]
        assert final["game"]["phase"] == "finished" and info["turn"] == 21
        assert final["game"]["engine_seed"] == seed
        assert not obs["action_mask"].any()
        assert env.observation_space.contains(obs)
        players = final["players"]
        shields = [players[str(s)]["resources"]["shields"] for s in (0,1,2)]
        for s in (0,1,2):
            expected = military_before_final[s] + sum(5 if shields[s] > shields[n] else -1 if shields[s] < shields[n] else 0 for n in (0,1,2) if n != s)
            assert players[str(s)]["points"]["military"] == expected
            assert final["last_actions"][str(s)]["turn"] == 20
        assert reward == info["final_totals"][0] - max(info["final_totals"][1:])
        assert env.proc.wait(timeout=5) == 0
        text = (path / "results.txt").read_text()
        for s, total in enumerate(info["final_totals"]):
            assert f"Player {s+1} score: {total}" in text
        trace.append(final)
        return trace
    finally:
        env.close()


def test_final_status_includes_last_turn_and_age_three_battles(require_engine, tmp_path):
    play(tmp_path, 0)


def test_replay_is_independent_of_exchange_path_and_polling(require_engine, tmp_path):
    assert play(tmp_path / "a", 123, 0) == play(tmp_path / "b", 123, 1000)
    assert play(tmp_path / "c", 124) != play(tmp_path / "d", 123)


def test_exit_cannot_return_stale_status_as_valid_transition(require_engine, tmp_path):
    env = make_env(tmp_path)
    try:
        _, info = env.reset(seed=0)
        env.proc.kill()
        env.proc.wait(timeout=5)
        _, reward, terminated, truncated, info = env.wait_next(info["turn"])
        assert truncated and not terminated and reward == 0
        assert info["transition_valid"] is False
        assert info["truncated_reason"] == "engine_exit"
    finally:
        env.close()


def test_timeout_and_malformed_status_are_not_transitions(require_engine, tmp_path):
    env = make_env(tmp_path, max_wait_seconds=.05)
    try:
        _, info = env.reset(seed=0)
        _, _, _, truncated, failure = env.wait_next(info["turn"])
        assert truncated and failure["truncated_reason"] == "timeout"
        (env.io_dir / "game_status.json").write_text("not json")
        _, _, _, truncated, failure = env.wait_next(info["turn"])
        assert truncated and failure["truncated_reason"] == "protocol_error"
    finally:
        env.close()


def test_startup_failure_closes_process_and_log(tmp_path):
    env = make_env(tmp_path, engine_path=sys.executable)
    with pytest.raises(EngineError, match="exited"):
        env.reset(seed=0)
    assert env.proc is None and env._stderr_file is None


def test_wonder_starting_resources_are_visible_in_initial_model_input(require_engine, tmp_path):
    # The old engine uses IDs 0..6 for A sides and 7..13 for B sides.
    # Neither side changes the starting resource. Stage rules are deliberately untouched.
    starting_columns = (3, 2, 0, 1, 6, 5, 4)  # Gizah, Babylon, Olympia, Rhodos, Ephesos, Alexandria, Halikarnassos
    for seed in (0, 1, 2):
        env = make_env(tmp_path / str(seed))
        try:
            _, info = env.reset(seed=seed)
            status = info["status"]
            for seat in (0, 1, 2):
                expected = np.zeros((3, 7), dtype=np.int32)
                for row, player_seat in enumerate((seat, (seat - 1) % 3, (seat + 1) % 3)):
                    player = status["players"][str(player_seat)]
                    assert player["wonder_stage"] == 0 and not player["cards_played"]
                    expected[row, starting_columns[player["wonder_id"] % 7]] = 1
                obs = build_obs(status, seat)
                np.testing.assert_array_equal(obs["table_prod"], expected)
                np.testing.assert_array_equal(flatten_global(obs)[42:63], expected.reshape(-1))
        finally:
            env.close()
