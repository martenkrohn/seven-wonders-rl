#!/usr/bin/env python3
"""Train a fresh BGA decision policy; legacy checkpoints are incompatible."""

import argparse, json
from pathlib import Path
import torch
from sevenwonders_rl.envs import DecisionEnv
from sevenwonders_rl.selfplay.decision_league import LeagueConfig
from sevenwonders_rl.utils.decision_ppo import train, evaluate
from sevenwonders_rl.utils.decision_console import DecisionTrainingConsole
from sevenwonders_rl.utils.decision_control import request_stop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--steps",
        type=int,
        help="Additional decisions; default 256 without --continuous",
    )
    p.add_argument(
        "--continuous",
        action="store_true",
        help="Train without a step budget until stopped",
    )
    p.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume output-dir/checkpoint.pt when present",
    )
    p.add_argument(
        "--stop",
        action="store_true",
        help="Request a clean stop after the current PPO update",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--agent-seat",
        type=int,
        choices=range(3),
        help="Initial seat (default: 0; inherited on resume)",
    )
    p.add_argument(
        "--eval-games",
        type=int,
        default=0,
        help="Optional technical evaluation against random options; no league feedback",
    )
    p.add_argument(
        "--eval-every-games",
        type=int,
        help="Intermediate evaluation every N completed training games (default: league 5000, random 0; 0 disables)",
    )
    p.add_argument("--resume", type=Path)
    p.add_argument("--allow-development", action="store_true")
    p.add_argument("--opponent-mode", choices=("league", "random"), default="league")
    p.add_argument(
        "--snapshot-every",
        type=int,
        help="Default: 65536; resume inherits the saved setting",
    )
    p.add_argument(
        "--pool-max", type=int, help="Maximum active snapshots (default: 64)"
    )
    p.add_argument("--pfsp-power", type=float)
    p.add_argument("--pfsp-control-fraction", type=float)
    p.add_argument("--pfsp-half-life", type=float)
    p.add_argument("--pfsp-prior-games", type=float)
    p.add_argument(
        "--fixed-seat",
        action="store_true",
        default=None,
        help="Disable the default seat rotation",
    )
    p.add_argument("--rollout-size", type=int, help="Default: 64; inherited on resume")
    p.add_argument("--epochs", type=int, help="Default: 4; inherited on resume")
    p.add_argument(
        "--workers",
        type=int,
        help="Simultaneous CPU games; default 4, inherited on resume. Changes per-game rollout horizon.",
    )
    p.add_argument(
        "--log-seconds",
        type=float,
        default=10,
        help="Console progress interval, after completed updates (default: 10 seconds)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print only the final machine-readable JSON summary",
    )
    a = p.parse_args()
    if a.eval_every_games is None:
        a.eval_every_games = 5000 if a.opponent_mode == "league" else 0
    if a.stop:
        active = request_stop(a.output_dir)
        print(
            "Stop requested: the current PPO update will finish and be saved."
            if active
            else "No training is active in this run directory.",
            flush=True,
        )
        return
    if a.continuous and a.steps is not None:
        p.error("--continuous has no --steps budget")
    if a.continuous and a.eval_games:
        p.error(
            "--eval-games is only available for bounded runs; use --eval-every-games or eval_decision.py for continuous training."
        )
    if a.auto_resume and a.resume:
        p.error("Choose --auto-resume or --resume")
    if a.auto_resume and (a.output_dir / "checkpoint.pt").exists():
        a.resume = a.output_dir / "checkpoint.pt"
    if not a.continuous and a.steps is None:
        a.steps = 256
    if a.eval_games < 0:
        p.error("--eval-games must be nonnegative")
    if a.eval_every_games < 0:
        p.error("--eval-every-games must be nonnegative")
    if a.opponent_mode == "random" and a.eval_every_games:
        p.error(
            "Intermediate evaluation requires --opponent-mode league; use --eval-every-games 0 for random opponents."
        )
    try:
        console = DecisionTrainingConsole(
            a.output_dir, resume=bool(a.resume), interval=a.log_seconds
        )
    except ValueError as exc:
        p.error(str(exc))
    checkpoint = (
        torch.load(a.resume, map_location="cpu", weights_only=True)
        if a.resume
        else None
    )
    saved_training = (
        checkpoint.get("metrics", {}).get("configuration", {}) if checkpoint else {}
    )
    for name, default in (
        ("rollout_size", 64),
        ("epochs", 4),
        ("agent_seat", 0),
        ("workers", 4),
    ):
        if getattr(a, name) is None:
            setattr(a, name, saved_training.get(name, default))
    if a.workers < 1:
        p.error("--workers must be positive")
    settings = {
        k: v
        for k, v in {
            "snapshot_every": a.snapshot_every,
            "pool_max": a.pool_max,
            "power": a.pfsp_power,
            "control_fraction": a.pfsp_control_fraction,
            "half_life": a.pfsp_half_life,
            "prior_games": a.pfsp_prior_games,
            "rotate_seats": False if a.fixed_seat else None,
        }.items()
        if v is not None
    }
    if a.opponent_mode == "random" and settings:
        p.error("PFSP settings require --opponent-mode league")
    league_config = False
    if a.opponent_mode == "league":
        if a.resume:
            saved = checkpoint.get("league")
            if saved is None:
                p.error(
                    "This checkpoint has no league; use --opponent-mode random for non-league resume"
                )
            settings = {**saved["config"], **settings}
        try:
            league_config = LeagueConfig(**settings)
        except ValueError as exc:
            p.error(str(exc))
    env = DecisionEnv(a.engine, a.agent_seat, allow_development=a.allow_development)
    try:
        model, metrics = train(
            env,
            a.steps,
            a.seed,
            a.output_dir,
            a.resume,
            rollout_size=a.rollout_size,
            epochs=a.epochs,
            league_config=league_config,
            progress=None if a.json else console,
            workers=a.workers,
            evaluation_every_games=a.eval_every_games,
        )
        if not a.json:
            console.finish(metrics)
        # Independent environment: evaluation cannot mutate the training league,
        # consume its sampler RNG or replace its durable opponent assignment.
        eval_games = 0 if metrics["status"] == "stopped" else a.eval_games
        metrics["evaluation"] = []
        if eval_games:
            evaluation_env = DecisionEnv(
                a.engine, a.agent_seat, allow_development=a.allow_development
            )
            if not a.json:
                print(
                    f"\n  Separate technical evaluation: {eval_games} games (excluded from training counts)",
                    flush=True,
                )
            try:
                metrics["evaluation"] = evaluate(
                    evaluation_env, model, eval_games, a.seed + 1000
                )
            finally:
                evaluation_env.close()
            # Keep post-training diagnostics separate from the durable training
            # metrics, which a subsequent resume may already have advanced.
            (a.output_dir / f"evaluation_step_{metrics['steps']}.json").write_text(
                json.dumps(
                    {"steps": metrics["steps"], "evaluation": metrics["evaluation"]},
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
        if a.json:
            print(
                json.dumps(
                    {
                        "status": metrics["status"],
                        "steps": metrics["steps"],
                        "completed_games": metrics["completed_games"],
                        "completed_games_total": metrics["completed_games_total"],
                        "actually_applied": metrics["actually_applied"],
                        "league": metrics["league"],
                        "evaluation": metrics["evaluation"],
                    },
                    indent=2,
                )
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
