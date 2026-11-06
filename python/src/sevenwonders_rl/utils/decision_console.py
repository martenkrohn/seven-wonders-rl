"""Human-readable progress for the active trainer, based on committed updates."""

import math
import time


def number(value):
    return f"{value:,}"


def duration(seconds):
    seconds = max(0, math.ceil(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def numeric_column(value, width, *, decimals=None):
    text = number(value) if decimals is None else f"{value:.{decimals}f}"
    if len(text) > width:
        text = f"{value:.3e}"
    return text.rjust(width)


def duration_column(seconds):
    text = duration(seconds)
    if len(text) > 12:
        text = f"{seconds / 86400:.3e} d"
    return text.rjust(12)


def intermediate_evaluation_lines(summary):
    """Aligned means and changes for the short in-training evaluation."""

    def row(label, value, unit="", *, scale=1, signed=False):
        if value is None:
            text = "n/a".rjust(12)
        elif signed:
            text = f"{value * scale:+.2f}".rjust(12)
        else:
            text = numeric_column(value * scale, 12, decimals=2)
        return f"  {label + ':':36} {text} {unit}".rstrip()

    model = summary["models"]["current"]
    lines = []
    if not summary["valid"]:
        lines.append(
            "  Intermediate evaluation incomplete; results have limited reliability."
        )
    lines.append("  POINT BREAKDOWN · Average per game")
    categories = model["categories"]
    labels = {
        "civilian": "Blue cards",
        "military": "Military",
        "scientific": "Science",
        "commercial": "Yellow cards",
        "guild": "Guilds",
        "wonder": "Wonder",
        "coins": "Coins",
    }
    for key, label in labels.items():
        if key in categories:
            lines.append(row(label, categories[key], "points"))
    for key in sorted(categories.keys() - labels.keys() - {"total"}):
        lines.append(row(key, categories[key], "points"))
    lines.append(
        row(
            "Average total per game",
            categories.get("total", model["own_points"]),
            "points",
        )
    )
    lines.extend(
        [
            "",
            row("Average margin to best opponent", model["margin"], "points"),
            row("Win share", model["win_share"], "%", scale=100),
        ]
    )
    change = summary["paired_change"]
    if change:
        lines.extend(
            [
                "",
                "  CHANGE FROM PREVIOUS EVALUATION",
                row(
                    "Average own points per game",
                    change["own_points"]["change"],
                    "points",
                    signed=True,
                ),
                row(
                    "Average margin to best opponent",
                    change["margin"]["change"],
                    "points",
                    signed=True,
                ),
                row(
                    "Win share",
                    change["win_share"]["change"],
                    "percentage points",
                    scale=100,
                    signed=True,
                ),
            ]
        )
    return lines


class DecisionTrainingConsole:
    def __init__(self, output, *, resume=False, interval=10, clock=time.monotonic):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("The output interval must be positive and finite")
        self.output, self.resume, self.interval = output, resume, interval
        self.clock = clock
        self.started = self.last_print = None
        self.latest = None

    def __call__(self, metrics):
        self.latest = metrics
        now = self.clock()
        if self.started is None:
            self.started = self.last_print = now
            league = metrics["league"]
            evaluation = metrics.get("intermediate_evaluation", {})
            evaluation_text = (
                f"every {number(evaluation['interval_games'])} training games; "
                f"next at {number(evaluation['next_games'])}"
                if evaluation.get("interval_games")
                else "disabled"
            )
            print(
                "\n" + "=" * 72 + "\n"
                f"  SEVEN WONDERS · {'RESUME TRAINING' if self.resume else 'START TRAINING'}\n"
                + "="
                * 72
                + "\n"
                f"  {'Training games so far:':23}{number(metrics['completed_games_total'])}\n"
                f"  {'This session:':23}{('Continuous until stopped' if metrics.get('continuous') else number(metrics['configuration']['steps']) + ' additional decisions')}\n"
                f"  {'Opponents:':23}{'Own snapshot league · Pair-PFSP' if league else 'Configured opponents / Random opponents'}\n"
                f"  {'Concurrent games:':23}{metrics['configuration'].get('workers', 1)}\n"
                f"  {'Saving:':23}After every PPO update\n"
                f"  {'Intermediate eval:':23}{evaluation_text}\n"
                f"  {'Run directory:':23}{self.output}\n"
                "  Only completed training games are counted; evaluation games are excluded.\n"
                + "-"
                * 72,
                flush=True,
            )
            if not metrics.get("training_duration_complete", True):
                print(
                    "  Total time: recorded training time; earlier sessions without timing data are excluded.",
                    flush=True,
                )
            return
        if now - self.last_print >= self.interval:
            self._progress(metrics, now)

    def _progress(self, metrics, now, final=False):
        self.last_print = now
        elapsed = metrics.get("training_seconds_session", max(0, now - self.started))
        games = metrics["completed_games"]
        games_minute = games * 60 / elapsed if elapsed else 0
        print(
            f"Games {numeric_column(metrics['completed_games_total'], 14)}"
            f"  Time {duration_column(metrics.get('training_seconds_total', elapsed))}"
            f" | Session: Games {numeric_column(games, 12)}"
            f"  Time {duration_column(elapsed)}"
            f" | Games/min: {numeric_column(games_minute, 10, decimals=2)}",
            flush=True,
        )

    def finish(self, metrics):
        if self.started is not None:
            self._progress(metrics, self.clock(), final=True)
            status = (
                "CLEANLY STOPPED" if metrics.get("status") == "stopped" else "COMPLETED"
            )
            print(
                f"\n  {status}\n  Resumable checkpoint: {self.output}/checkpoint.pt\n"
                + "=" * 72,
                flush=True,
            )
