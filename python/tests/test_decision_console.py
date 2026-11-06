import pytest

"""Stable progress columns across digit boundaries, resume totals and zero time."""
from sevenwonders_rl.utils.decision_console import DecisionTrainingConsole


def test_progress_columns_and_session_rate(capsys):
    console = DecisionTrainingConsole("/tmp/run")
    console.started = 0
    rows = []
    for total, games, elapsed in [
        (9, 1, 0),
        (1000, 100, 120),
        (10**18, 10**15, 3600 * 12345),
    ]:
        console._progress(
            {
                "completed_games_total": total,
                "completed_games": games,
                "training_seconds_session": elapsed,
                "training_seconds_total": elapsed + 86400,
            },
            elapsed,
        )
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1
        rows.append(lines[0])
    assert len({len(row) for row in rows}) == 1
    for word in ("Games", "Session:", "Games/min:"):
        assert len({row.index(word) for row in rows}) == 1
    assert rows[0].endswith("      0.00")
    assert rows[1].endswith("     50.00")
    assert "1,000" in rows[1] and "00:02:00" in rows[1]
    assert "24:02:00" in rows[1]
    assert "e+" in rows[2]  # Huge values retain alignment instead of widening columns.


pytestmark = pytest.mark.unit
