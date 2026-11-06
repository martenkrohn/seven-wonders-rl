"""Game-count based training controls, evaluated only after durable PPO updates."""

import json
from pathlib import Path

import torch

from sevenwonders_rl.selfplay.decision_league import atomic_json
from sevenwonders_rl.utils.decision_console import (
    intermediate_evaluation_lines,
    numeric_column,
)


class IntermediateEvaluations:
    def __init__(self, output, every_games, workers, old_metrics):
        self.output = Path(output) if output else None
        self.interval = every_games
        self.workers = workers
        games = old_metrics.get(
            "completed_games_total", old_metrics.get("completed_games", 0)
        )
        saved = old_metrics.get("intermediate_evaluation", {})
        self.next_games = (
            (
                saved["next_games"]
                if saved.get("interval_games") == every_games
                else (games // every_games + 1) * every_games
            )
            if every_games
            else None
        )
        self.last = saved.get("last")
        if every_games and self.output is None:
            raise ValueError(
                "Intermediate evaluation requires a league output directory"
            )
        # An evaluation can finish immediately before a hard kill prevents the
        # next learner save. Reconcile its small commit record on resume.
        schedule = (
            self.output / "intermediate_evaluation/schedule.json"
            if self.output
            else None
        )
        if (
            every_games
            and saved.get("interval_games") == every_games
            and schedule.is_file()
        ):
            committed = json.loads(schedule.read_text())
            last = committed.get("last") or {}
            if (
                committed.get("interval_games") == every_games
                and last.get("steps", float("inf")) <= old_metrics.get("steps", 0)
                and last.get("completed_games_total", float("inf")) <= games
                and committed.get("next_games", 0) > self.next_games
            ):
                self.next_games = committed["next_games"]
                self.last = last

    def state_dict(self):
        return {
            "interval_games": self.interval,
            "next_games": self.next_games,
            "last": self.last,
        }

    def run_due(self, metrics, engine, emit=print):
        games = metrics["completed_games_total"]
        if not self.interval or games < self.next_games:
            return False
        from sevenwonders_rl.evaluation.decision_evaluation import evaluate_run

        threshold = self.next_games
        directory = self.output / "intermediate_evaluation"
        if emit:
            emit(
                "\nINTERMEDIATE EVALUATION"
                f" | Training games: {numeric_column(games, 14)}"
                f" | Step: {numeric_column(metrics['steps'], 14)}"
            )
        try:
            panel = directory / "panel/plan.json"
            if panel.exists():
                count = len(json.loads(panel.read_text())["references"])
            else:
                state = torch.load(
                    self.output / "checkpoint.pt", map_location="cpu", weights_only=True
                )
                trained = [
                    s
                    for s in state["league"]["snapshots"]
                    if 0 < s["step"] <= state["steps"]
                ]
                count = min(3, len(trained)) or 1
            # At least 27 games, balanced across every pair and all three seats.
            blocks = (27 + count * count * 3 - 1) // (count * count * 3)
            # Model construction in the evaluator consumes Torch randomness.
            # The learner and the serial frozen opponents must see the same RNG
            # sequence as they would without intermediate evaluations.
            with torch.random.fork_rng(devices=[]):
                summary, path = evaluate_run(
                    self.output,
                    engine,
                    directory,
                    blocks=blocks,
                    seed=9159001,
                    progress=None,
                    workers=self.workers,
                    training_control=True,
                )
            model = summary["models"]["current"]
            self.last = {
                "threshold_games": threshold,
                "completed_games_total": games,
                "steps": metrics["steps"],
                "valid": summary["valid"],
                "directory": str(path.relative_to(self.output)),
                "own_points": model["own_points"],
                "margin": model["margin"],
                "win_share": model["win_share"],
                "completed": model["completed"],
                "attempted": model["attempted"],
                "failed": model["failed"],
            }
            if emit:
                emit("\n".join(intermediate_evaluation_lines(summary)) + "\n")
        except Exception as exc:
            self.last = {
                "threshold_games": threshold,
                "completed_games_total": games,
                "steps": metrics["steps"],
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if emit:
                emit(
                    f"  INTERMEDIATE EVALUATION FAILED: {self.last['error']}\n"
                    "  The training checkpoint is saved; training will continue."
                )
        # Do not run an evaluation storm when enabling the feature on an old
        # run or crossing several boundaries in one (unusually large) update.
        self.next_games = (games // self.interval + 1) * self.interval
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "schedule.json", self.state_dict())
        return True
