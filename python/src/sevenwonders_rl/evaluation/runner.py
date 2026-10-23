from __future__ import annotations

import json
import time
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from sevenwonders_rl.envs import SevenWondersEnv, SeatPolicy
from sevenwonders_rl.envs.sevenwonders_env import EngineError
from sevenwonders_rl.features.action_feature_schema import IDX
from sevenwonders_rl.policies.registry import make_policy
from sevenwonders_rl.policies.ppo_checkpoint_policy import PPOCheckpointPolicy
from sevenwonders_rl.utils.checkpoints import file_hash
from sevenwonders_rl.utils.ppo_utils import select_action_from_obs
from sevenwonders_rl.utils.runtime import seat_order
from sevenwonders_rl.utils.seeding import derive_seed
from sevenwonders_rl.evaluation.telemetry import status_snapshot, action_event, category_statistics


def run_game(env, select_action, record, on_step=None, *, detailed=False):
    """Execute one attempt; only a final protocol state supplies result fields.

    Expected transport failures become records; programming errors propagate.
    on_step observes requested actions and the engine's acknowledgement.
    """
    result = {**record, "schema_version": 3, "status": "failed", "reason": None,
              "steps": 0, "totals": None, "agent_vp": None, "vp_diff": None,
              "win": None, "win_share": None, "vp_tie": None, "action_counts": {}}
    start = time.perf_counter()
    info = {}
    stage = "startup"
    try:
        obs, info = env.reset(seed=record["engine_seed"])
        if detailed:
            result.update(age_states=[], actions=[], initial_state=status_snapshot(info.get('status', {})))
        stage = "step"
        for _ in range(22):
            before = obs
            before_status = info.get('status', {})
            action = int(select_action(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            result["steps"] += 1
            after_status = info.get('status', {})
            if detailed and info.get('transition_valid', True) and before_status and after_status:
                event = action_event(before_status, after_status, before, action, env.agent_seat)
                if event is not None:
                    result['actions'].append(event)
                old_age = before_status.get('game', {}).get('era')
                new_age = after_status.get('game', {}).get('era')
                if old_age in (1, 2, 3) and (new_age != old_age or terminated):
                    result['age_states'].append(status_snapshot(after_status, old_age))
            if on_step is not None and info.get("transition_valid", True):
                on_step(before, action, info)
            if terminated:
                status = info["status"]
                if status["game"].get("phase") != "finished" or not info.get("transition_valid", True):
                    raise EngineError("protocol_error", "Missing valid final status")
                players = status["players"]
                totals = [int(players[str(s)]["points"]["total"]) for s in (0, 1, 2)]
                agent = env.agent_seat
                diff = totals[agent] - max(t for s, t in enumerate(totals) if s != agent)
                result.update(status="completed", totals=totals, agent_vp=totals[agent],
                              vp_diff=diff, win=int(diff > 0), vp_tie=int(diff == 0),
                              win_share=(1. / sum(t == max(totals) for t in totals) if diff >= 0 else 0.),
                              points_breakdown=players[str(agent)]["points"],
                              final_state=status_snapshot(status, 3),
                              winner_seats=[s for s, t in enumerate(totals) if t == max(totals)])
                break
            if truncated:
                reason = info.get("truncated_reason", "external_limit")
                result["status"] = "truncated" if info.get("transition_valid", True) or reason == "timeout" else "failed"
                result["reason"] = reason
                break
        else:
            raise EngineError("protocol_error", "Exceeded maximum game length")
    except (EngineError, OSError) as exc:
        result["reason"] = getattr(exc, "reason", "startup_error" if stage == "startup" else "io_error")
        result["status"] = "truncated" if result["reason"] == "timeout" else "failed"
        result["error"] = str(exc)
        result["error_stage"] = stage
    finally:
        env.close()
        result["duration_seconds"] = time.perf_counter() - start
        if result["status"] != "completed" and hasattr(env, "io_dir"):
            error_log = env.io_dir.parent / "engine_stderr.log"
            if error_log.exists():
                result["engine_stderr_tail"] = error_log.read_text(errors="replace")[-65536:]
    return result, info


def aggregate(records):
    completed = [r for r in records if r.get("status") == "completed"]
    counts = Counter(r["status"] for r in records)
    def mean(key):
        return float(np.mean([r[key] for r in completed])) if completed else None
    return {
        "attempted": len(records), "completed": len(completed),
        "truncated": counts["truncated"], "failed": counts["failed"],
        "completion_rate": len(completed) / len(records) if records else None,
        "win_rate": mean("win"), "win_share": mean("win_share"), "vp_tie_rate": mean("vp_tie"),
        "vp_diff_mean": mean("vp_diff"), "agent_vp_mean": mean("agent_vp"),
        "vp_diff_std": float(np.std([r["vp_diff"] for r in completed])) if completed else None,
        "total_games": len(completed),
        "valid_for_selection": bool(records) and len(completed) == len(records),
        "failure_reasons": dict(Counter(r["reason"] for r in records if r["status"] != "completed")),
        "categories": category_statistics(records),
    }


def evaluate_model(model, device, specs, repo_root, io_root, jsonl_path,
                   action_mode="greedy", engine_poll_us=1000, poll_seconds=.001,
                   checkpoint_id=None, *, batch_opponents=True, detailed=False, progress_callback=None):
    """Evaluate an explicit, fixed game plan without consuming training RNGs."""
    records = []
    snapshot_cache = {}
    snapshot_hashes = {}
    previous_training = model.training
    model.eval()
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with Path(jsonl_path).open("x", encoding="utf-8") as output:
            for game_id, spec in enumerate(specs):
                seed = int(spec["engine_seed"])
                agent = spec.get("agent_seat", 0)
                _, left, right = seat_order(agent)
                opponents = [SeatPolicy("random") for _ in range(3)]
                policy_seeds = {str(s): derive_seed(seed, "policy", s) for s in (0, 1, 2)}
                for seat, name in ((left, spec["left"]), (right, spec["right"])):
                    if isinstance(name, dict):
                        path = str(Path(name["checkpoint"]).resolve())
                        if path not in snapshot_cache:
                            snapshot_cache[path] = PPOCheckpointPolicy(path, device=str(device))
                            snapshot_hashes[path] = file_hash(path)
                        if snapshot_hashes[path] != name["sha256"]:
                            raise ValueError(f"Evaluation reference changed: {path}")
                        opponents[seat] = SeatPolicy("ppo_snapshot", policy=snapshot_cache[path])
                    else:
                        opponents[seat] = SeatPolicy(name, policy=make_policy(name, policy_seeds[str(seat)]))
                env = SevenWondersEnv(repo_root=str(repo_root), agent_seat=agent,
                    io_dir=str(Path(io_root) / f"game_{game_id:04d}" / "io"),
                    opponent_policies=opponents, invalid_action_fallback="raise",
                    engine_poll_us=engine_poll_us, poll_seconds=poll_seconds,
                    batch_opponents=batch_opponents)
                rng = np.random.default_rng(policy_seeds[str(agent)])
                counts = Counter()
                def on_step(obs, action, info):
                    ack = info.get("status", {}).get("last_actions", {}).get(str(agent), {})
                    if ack.get("status") != "applied":
                        return
                    kind = action // 7
                    feats = obs["action_feats"][action]
                    prod = sum(feats[IDX[f"delta_prod_{r}"]] for r in ("wood", "ore", "clay", "stone", "loom", "glass", "papyrus"))
                    prod += feats[IDX["delta_prod_choice_raw"]] + feats[IDX["delta_prod_choice_manuf"]]
                    counts["discard" if kind == 3 else "build_wonder" if kind == 1 else "prod_build" if prod > 0 else "other"] += 1
                record, _ = run_game(env, lambda obs: select_action_from_obs(
                    obs, model, device, deterministic=action_mode == "greedy", rng=rng),
                    {"game_id": game_id, "engine_seed": seed, "seed": seed,
                     "agent_seat": agent, "policy_seeds": policy_seeds,
                     "checkpoint_id": checkpoint_id, "action_selection_mode": action_mode,
                     "opponents": {"left": spec["left"], "right": spec["right"]},
                     "opponents_by_seat": {str(s): opponents[s].name for s in (left, right)}}, on_step,
                    detailed=detailed)
                record["action_counts"] = dict(counts)
                output.write(json.dumps(record, allow_nan=False) + "\n")
                output.flush()
                records.append(record)
                shutil.rmtree(env.io_dir.parent)
                if progress_callback is not None:
                    progress_callback(game_id + 1, len(specs))
    finally:
        model.train(previous_training)
    return {**aggregate(records), "jsonl_path": str(jsonl_path)}
