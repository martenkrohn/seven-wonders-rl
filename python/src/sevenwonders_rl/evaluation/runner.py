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
from sevenwonders_rl.policies.action_constraints import ACTION_CONSTRAINTS, constrained_mask, green_build_mask
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


def _evaluate_serial(model, device, specs, repo_root, io_root, jsonl_path,
                   action_mode="greedy", engine_poll_us=1000, poll_seconds=.001,
                   checkpoint_id=None, *, batch_opponents=True, detailed=False, progress_callback=None,
                   action_constraint='none', head_id=None, _game_ids=None):
    """Evaluate an explicit, fixed game plan without consuming training RNGs."""
    records = []
    count_heads = getattr(model, 'num_heads', 1)
    if head_id is not None and not 0 <= head_id < count_heads:
        raise ValueError('Evaluation head ID out of range')
    if action_constraint not in ACTION_CONSTRAINTS:
        raise ValueError(f'Unknown action constraint: {action_constraint}')
    snapshot_cache = {}
    snapshot_hashes = {}
    previous_training = model.training
    model.eval()
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with Path(jsonl_path).open("x", encoding="utf-8") as output:
            game_ids = range(len(specs)) if _game_ids is None else _game_ids
            for game_id, spec in zip(game_ids, specs):
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
                    batch_opponents=batch_opponents,
                    observation_memory=getattr(model, 'memory_enabled', False), public_information=(
                        getattr(model, 'public_information', False) or any(
                            getattr(getattr(p.policy, 'model', None), 'public_information', False)
                            for p in opponents)))
                rng = np.random.default_rng(policy_seeds[str(agent)])
                head_rng = np.random.default_rng(derive_seed(seed, 'evaluation_head', agent))
                selected_head = (head_id if head_id is not None else int(head_rng.integers(count_heads)))
                counts = Counter()
                constraint_counts = Counter(applied_decisions=0, green_opportunities=0,
                    restricted_decisions=0, green_builds=0, violations=0)
                def on_step(obs, action, info):
                    ack = info.get("status", {}).get("last_actions", {}).get(str(agent), {})
                    if ack.get("status") != "applied":
                        return
                    green = green_build_mask(obs)
                    available = bool(green.any())
                    constraint_counts['applied_decisions'] += 1
                    constraint_counts['green_opportunities'] += int(available)
                    constraint_counts['green_builds'] += int(bool(green[action]))
                    constraint_counts['restricted_decisions'] += int(action_constraint == 'green_priority'
                        and available and np.any((obs['action_mask'] > 0) & (green == 0)))
                    constraint_counts['violations'] += int(action_constraint == 'green_priority'
                        and available and not green[action])
                    kind = action // 7
                    feats = obs["action_feats"][action]
                    prod = sum(feats[IDX[f"delta_prod_{r}"]] for r in ("wood", "ore", "clay", "stone", "loom", "glass", "papyrus"))
                    prod += feats[IDX["delta_prod_choice_raw"]] + feats[IDX["delta_prod_choice_manuf"]]
                    counts["discard" if kind == 3 else "build_wonder" if kind == 1 else "prod_build" if prod > 0 else "other"] += 1
                record, _ = run_game(env, lambda obs: select_action_from_obs(
                    {**obs, 'action_mask': constrained_mask(obs, action_constraint)},
                    model, device, deterministic=action_mode == "greedy", rng=rng, head_id=selected_head),
                    {"game_id": game_id, "engine_seed": seed, "seed": seed,
                     "agent_seat": agent, "policy_seeds": policy_seeds,
                     "checkpoint_id": checkpoint_id, "action_selection_mode": action_mode,
                     "action_constraint": action_constraint,
                     "head_id": selected_head, "head_selection": 'fixed' if head_id is not None else 'per_game',
                     "opponents": {"left": spec["left"], "right": spec["right"]},
                     "opponents_by_seat": {str(s): opponents[s].name for s in (left, right)}}, on_step,
                    detailed=detailed)
                record['opponent_head_ids'] = {str(k): v for k, v in getattr(env, '_opponent_head_ids', {}).items()}
                record["action_counts"] = dict(counts)
                record['constraint_statistics'] = dict(constraint_counts)
                output.write(json.dumps(record, allow_nan=False) + "\n")
                output.flush()
                records.append(record)
                shutil.rmtree(env.io_dir.parent)
                if progress_callback is not None:
                    progress_callback(game_id + 1, len(specs))
    finally:
        model.train(previous_training)
    statistics = Counter()
    for record in records:
        statistics.update(record['constraint_statistics'])
    return {**aggregate(records), "jsonl_path": str(jsonl_path),
            'action_constraint': action_constraint, 'constraint_statistics': dict(statistics)}


_WORKER_MODEL = None


def _init_evaluation_worker(model):
    import torch
    global _WORKER_MODEL
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _WORKER_MODEL = model.eval()


def _evaluation_chunk(task):
    specs, game_ids, kwargs = task
    _evaluate_serial(_WORKER_MODEL, 'cpu', specs, _game_ids=game_ids, **kwargs)
    return str(kwargs['jsonl_path'])


def evaluate_model(model, device, specs, repo_root, io_root, jsonl_path,
                   action_mode="greedy", engine_poll_us=1000, poll_seconds=.001,
                   checkpoint_id=None, *, batch_opponents=True, detailed=False,
                   progress_callback=None, action_constraint='none', workers=1, head_id=None):
    """Evaluate a fixed plan, optionally using independent CPU worker processes.

    Game IDs, seed streams and output order are independent of worker scheduling.
    Models are loaded once per chunk and reference caches survive all its games.
    The calling process's training mode, weights and RNG are left untouched.
    """
    if workers < 1:
        raise ValueError('workers must be positive')
    if action_constraint not in ACTION_CONSTRAINTS:
        raise ValueError(f'Unknown action constraint: {action_constraint}')
    kwargs = dict(repo_root=repo_root, io_root=io_root, jsonl_path=jsonl_path,
                  action_mode=action_mode, engine_poll_us=engine_poll_us,
                  poll_seconds=poll_seconds, checkpoint_id=checkpoint_id,
                  batch_opponents=batch_opponents, detailed=detailed,
                  action_constraint=action_constraint, head_id=head_id)
    if workers == 1 or len(specs) < 2:
        return _evaluate_serial(model, device, specs, progress_callback=progress_callback, **kwargs)
    if str(device) != 'cpu':
        raise ValueError('Parallel evaluation currently supports CPU models only')
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    from tempfile import TemporaryDirectory

    output_path = Path(jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    # Reserve the final output before launching workers; never overwrite a run.
    with output_path.open('x', encoding='utf-8') as output:
        with TemporaryDirectory(prefix='evaluation-parts-', dir=output_path.parent) as parts:
            count = min(workers, len(specs))
            tasks = []
            for worker in range(count):
                ids = list(range(worker, len(specs), count))
                tasks.append(([specs[i] for i in ids], ids,
                              {**kwargs, 'jsonl_path': Path(parts) / f'{worker}.jsonl'}))
            try:
                with ProcessPoolExecutor(max_workers=count, mp_context=get_context('spawn'),
                                         initializer=_init_evaluation_worker, initargs=(model,)) as executor:
                    futures = {executor.submit(_evaluation_chunk, task): len(task[0]) for task in tasks}
                    completed = 0
                    for future in as_completed(futures):
                        future.result()
                        completed += futures[future]
                        if progress_callback is not None:
                            progress_callback(completed, len(specs))
            finally:
                # Preserve finished attempts even if another chunk raises. The
                # executor has joined the writers before temporary parts vanish.
                for part in sorted(Path(parts).glob('*.jsonl')):
                    records.extend(json.loads(line) for line in part.read_text().splitlines())
                records.sort(key=lambda record: record['game_id'])
                for record in records:
                    output.write(json.dumps(record, allow_nan=False) + '\n')
    statistics = Counter()
    for record in records:
        statistics.update(record['constraint_statistics'])
    return {**aggregate(records), 'jsonl_path': str(jsonl_path),
            'action_constraint': action_constraint, 'constraint_statistics': dict(statistics)}
