"""Read-only protocol-3 snapshot evaluation against a versioned fixed panel.

No training pool/optimizer is created. The evaluator owns its processes, model
cache, RNGs, output lock and journals, and never locks the training directory.
"""

from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import importlib.metadata
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import shutil
import signal
import time
import uuid

import numpy as np
import torch

from sevenwonders_rl.envs import DecisionEnv
from sevenwonders_rl.features.decision_features import feature_contract
from sevenwonders_rl.selfplay.decision_league import atomic_json, file_hash
from sevenwonders_rl.utils.decision_ppo import load_checkpoint


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@contextmanager
def evaluation_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("An evaluation of this panel is already running.") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def durable_run(run, allow_initial=False):
    # One open/read sees either old or new atomic checkpoint, never a mixture.
    raw = (run / "checkpoint.pt").read_bytes()
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if state.get("format") != "decision-ppo-v2" or not state.get("league"):
        raise ValueError("Evaluation requires an active decision-ppo-v2 league run.")
    if state["contract"] != feature_contract(state["contract"].get("ruleset")):
        raise ValueError(
            "The training checkpoint observation contract is incompatible."
        )
    snapshots = state["league"]["snapshots"]
    trained = sorted(
        (s for s in snapshots if s["step"] > 0 and s["step"] <= state["steps"]),
        key=lambda s: s["step"],
    )
    if len(trained) < 3 and not allow_initial:
        raise ValueError(
            "Three trained league snapshots are not yet available. "
            "The default run requires at least 196,608 decisions; let training continue."
        )
    if allow_initial and not trained:
        trained = sorted(
            (s for s in snapshots if s["step"] == 0), key=lambda s: s["step"]
        )
    if not trained:
        raise ValueError("No committed league snapshot is available.")
    return state, trained, hashlib.sha256(raw).hexdigest()


def copy_snapshot(run, row, target, contract):
    source = run / "league" / row["name"]
    if Path(row["name"]).name != row["name"]:
        raise ValueError("Invalid snapshot name")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError(f"Snapshot changed: {source}")
    target.write_bytes(raw)
    model, data = load_checkpoint(target, contract)
    if data.get("role") != "frozen-league-snapshot" or data["steps"] != row["step"]:
        raise ValueError("Invalid league snapshot")
    return model.eval().requires_grad_(False)


def prepare_panel(directory, run, state, trained, blocks, seed, reference_steps=None):
    contract = state["contract"]
    engine_commit = state["metrics"]["engine"]["source_commit"]
    settings = {
        "blocks": blocks,
        "seed": seed,
        "action_mode": "stochastic",
        "tie_rule": "highest-total-shared-win",
        "bootstrap_resamples": 2000,
        "primary_metric": "margin_to_best_opponent",
        "seat_order": [0, 1, 2],
        "pair_order": "all-ordered-including-identical",
        "schema_hash": contract["schema_hash"],
        "engine_commit": engine_commit,
        "ruleset": contract["ruleset"],
    }
    panel_path = directory / "panel"
    if panel_path.exists():
        plan = json.loads((panel_path / "plan.json").read_text())
        if plan["settings"] != settings:
            raise ValueError(
                "Fixed evaluation plan differs. Use a new --evaluation-dir for a new plan."
            )
        if reference_steps is not None and list(reference_steps) != [
            r["step"] for r in plan["references"]
        ]:
            raise ValueError(
                "The opponent panel is frozen; do not replace references within the same panel."
            )
        if plan["plan_id"] != digest({k: v for k, v in plan.items() if k != "plan_id"}):
            raise ValueError("Evaluation plan checksum mismatch.")
    else:
        if reference_steps is None:
            references = trained[:3]
        else:
            if len(set(reference_steps)) != 3:
                raise ValueError(
                    "Three distinct trained reference checkpoints are required."
                )
            by_step = {r["step"]: r for r in trained}
            try:
                references = [by_step[s] for s in reference_steps]
            except KeyError as exc:
                raise ValueError(
                    "Reference is missing from the committed training inventory."
                ) from exc
        seeds = [
            int(
                np.random.SeedSequence([seed, 6173, b]).generate_state(
                    1, dtype=np.uint64
                )[0]
            )
            for b in range(blocks)
        ]
        if set(seeds) & set(state["metrics"]["episode_seeds"]):
            raise ValueError("Evaluation seeds overlap with logged training seeds.")
        plan = {
            "version": 1,
            "settings": settings,
            "engine_seeds": seeds,
            "references": [
                {
                    "label": chr(65 + i),
                    "step": r["step"],
                    "sha256": r["sha256"],
                    "file": f"reference_{i}.pt",
                }
                for i, r in enumerate(references)
            ],
            "reference_origin": "own-training-snapshots; development-panel, not unseen opponents",
        }
        plan["plan_id"] = digest(plan)
        temporary = directory / f".panel-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            for row, desc in zip(references, plan["references"]):
                copy_snapshot(run, row, temporary / desc["file"], contract)
            atomic_json(temporary / "plan.json", plan)
            temporary.rename(panel_path)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    models = []
    for desc in plan["references"]:
        path = panel_path / desc["file"]
        if file_hash(path) != desc["sha256"]:
            raise ValueError("Frozen reference opponent has changed.")
        model, _ = load_checkpoint(path, contract)
        models.append(model.eval().requires_grad_(False))
    return plan, models


def play_game(model, opponents, engine, plan, block, left, right, seat):
    seed = plan["engine_seeds"][block]
    row = {
        "block": block,
        "left": left,
        "right": right,
        "seat": seat,
        "engine_seed": seed,
        "status": "failed",
        "reason": None,
        "decisions": 0,
    }
    env = DecisionEnv(
        engine_path=engine,
        agent_seat=seat,
        opponent_policies={
            (seat + 2) % 3: opponents[left].select_action,
            (seat + 1) % 3: opponents[right].select_action,
        },
    )
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [plan["settings"]["seed"], 9167, block, left, right, seat]
        )
    )
    try:
        obs, info = env.reset(seed=seed)
        if info["engine"]["source_commit"] != plan["settings"]["engine_commit"]:
            raise ValueError(
                "Evaluation engine differs from the evaluation plan or training commit."
            )
        for _ in range(32):
            action = model.select_action(obs, rng)
            obs, _, done, truncated, info = env.step(action)
            row["decisions"] += 1
            if truncated:
                raise RuntimeError("Truncated engine transition")
            if done:
                totals = info["final_totals"]
                player = env.state["players"][str(seat)]
                points = dict(player["points"])
                if sum(v for k, v in points.items() if k != "total") != totals[seat]:
                    raise ValueError("Point categories do not match the final total.")
                coin_points = player["resources"]["coins"] // 3
                points["coins"] = coin_points
                points["commercial"] -= coin_points
                win = (
                    1 / sum(t == max(totals) for t in totals)
                    if totals[seat] == max(totals)
                    else 0.0
                )
                row.update(
                    status="completed",
                    totals=totals,
                    own_points=totals[seat],
                    margin=totals[seat]
                    - max(t for s, t in enumerate(totals) if s != seat),
                    win_share=win,
                    wonder_id=player["wonder_id"],
                    categories=points,
                )
                return row
        raise RuntimeError("Game exceeds the contracted decision limit.")
    except Exception as exc:
        row["reason"] = f"{type(exc).__name__}: {exc}"
        return row
    finally:
        env.close()


_EVALUATION_WORKER = None


def _init_evaluation_worker(
    session, panel, contract, model_hashes, plan, engine, groups
):
    global _EVALUATION_WORKER
    grouped = hasattr(os, "setsid")
    if grouped:
        os.setsid()
    groups.put((os.getpid(), grouped))
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    models = {}
    for name, sha in model_hashes.items():
        path = Path(session) / (name + ".pt")
        if file_hash(path) != sha:
            raise ValueError("Evaluation snapshot changed")
        model, _ = load_checkpoint(path, contract)
        models[name] = model.eval().requires_grad_(False)
    opponents = []
    for row in plan["references"]:
        path = Path(panel) / row["file"]
        if file_hash(path) != row["sha256"]:
            raise ValueError("Reference opponent changed")
        model, _ = load_checkpoint(path, contract)
        opponents.append(model.eval().requires_grad_(False))
    _EVALUATION_WORKER = (models, opponents, engine, plan)


def _parallel_game(task):
    name, coordinates = task
    models, opponents, engine, plan = _EVALUATION_WORKER
    return play_game(models[name], opponents, engine, plan, *coordinates)


def _parallel_rows(executor, name, coordinates, plan):
    """Yield every result, including failures, so completed work is journaled."""
    futures = {}
    unsubmitted = []
    for i, coordinate in enumerate(coordinates):
        try:
            futures[executor.submit(_parallel_game, (name, coordinate))] = coordinate
        except Exception as exc:
            unsubmitted = [(c, exc) for c in coordinates[i:]]
            break

    def failed(coordinate, exc):
        block, left, right, seat = coordinate
        return dict(
            block=block,
            left=left,
            right=right,
            seat=seat,
            engine_seed=plan["engine_seeds"][block],
            status="failed",
            decisions=0,
            reason=f"{type(exc).__name__}: {exc}",
        )

    try:
        for future in as_completed(futures):
            coordinate = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = failed(coordinate, exc)
            yield row
        for coordinate, exc in unsubmitted:
            yield failed(coordinate, exc)
    finally:
        for future in futures:
            future.cancel()


def _row_key(row):
    return row["block"], row["left"], row["right"], row["seat"]


def basic(rows):
    valid = [r for r in rows if r["status"] == "completed"]
    return {
        "attempted": len(rows),
        "completed": len(valid),
        "failed": len(rows) - len(valid),
        **{
            k: float(np.mean([r[k] for r in valid])) if valid else None
            for k in ("margin", "win_share", "own_points")
        },
    }


def interval(values, seed):
    if len(values) < 2:
        return None
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 8821]))
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def summarize(rows, plan):
    result = basic(rows)
    valid = [r for r in rows if r["status"] == "completed"]
    references = len(plan.get("references", [None] * 3))
    expected = plan["settings"]["blocks"] * references * references * 3
    keys = {(r["block"], r["left"], r["right"], r["seat"]) for r in rows}
    expected_keys = set(
        itertools.product(
            range(plan["settings"]["blocks"]),
            range(references),
            range(references),
            range(3),
        )
    )
    result["valid"] = (
        result["completed"] == expected
        and keys == expected_keys
        and result["failed"] == 0
    )
    result["ci95"] = {}
    if result["valid"]:
        for metric in ("margin", "win_share", "own_points"):
            values = [
                np.mean([r[metric] for r in rows if r["block"] == b])
                for b in range(plan["settings"]["blocks"])
            ]
            result["ci95"][metric] = interval(values, plan["settings"]["seed"])
    result["by_pair"] = {
        f"{l},{r}": basic([x for x in rows if (x["left"], x["right"]) == (l, r)])
        for l, r in itertools.product(range(references), repeat=2)
    }
    result["by_seat"] = {
        str(s): basic([x for x in rows if x["seat"] == s]) for s in range(3)
    }
    result["by_wonder"] = {
        str(w): basic([x for x in valid if x["wonder_id"] == w])
        for w in sorted({x["wonder_id"] for x in valid})
    }
    result["categories"] = (
        {
            k: float(np.mean([r["categories"][k] for r in valid]))
            for k in valid[0]["categories"]
        }
        if valid
        else {}
    )
    result["failure_reasons"] = sorted(
        {r["reason"] for r in rows if r["status"] != "completed"}
    )
    return result


def paired(current, previous, plan):
    if not summarize(current, plan)["valid"] or not summarize(previous, plan)["valid"]:
        return None

    def key(r):
        return r["block"], r["left"], r["right"], r["seat"]

    old = {key(r): r for r in previous}
    if {key(r) for r in current} != set(old):
        raise ValueError("Comparison plans differ.")
    result = {}
    for metric in ("margin", "win_share", "own_points"):
        blocks = [
            np.mean(
                [r[metric] - old[key(r)][metric] for r in current if r["block"] == b]
            )
            for b in range(plan["settings"]["blocks"])
        ]
        result[metric] = {
            "change": float(np.mean(blocks)),
            "ci95": interval(blocks, plan["settings"]["seed"]),
        }
    return result


def report_text(summary):
    m = summary["models"]["current"]

    def fmt(v):
        return "not available" if v is None else f"{v:+.2f}"

    def ci(value):
        return (
            "not estimable (too few seed blocks)"
            if value is None
            else f"[{value[0]:+.2f}; {value[1]:+.2f}]"
        )

    lines = [
        "=" * 76,
        " SECOND EDITION · SEPARATE SNAPSHOT EVALUATION",
        "=" * 76,
        f" Snapshot: step {summary['candidate']['step']:,} | SHA {summary['candidate']['sha256'][:12]}",
        f" Evaluation plan: {summary['plan_id'][:12]} | {summary['blocks']} seed blocks",
        f" Data quality: {m['completed']}/{m['attempted']} games completed | {m['failed']} errors",
        f" Status: {'COMPLETE' if summary['valid'] else 'INCOMPLETE – no reliable progress comparison'}",
        "",
        f" Average margin to best opponent: {fmt(m['margin'])}",
        f" 95% interval: {ci(m['ci95'].get('margin'))}",
        f" Average own points per game: {fmt(m['own_points'])}",
        f" Win share (ties split equally): {100 * m['win_share']:.1f} %"
        if m["win_share"] is not None
        else " Win share: not available",
    ]
    for ref in summary.get("references", []):
        lines.append(
            f" Opponent {ref['label']}: step {ref['step']:,} | SHA {ref['sha256'][:12]}"
        )
    win_ci = m["ci95"].get("win_share")
    if win_ci is not None:
        lines.append(
            f" 95% interval of win share: [{100 * win_ci[0]:.1f}; {100 * win_ci[1]:.1f}] %"
        )
    if summary["paired_change"]:
        change = summary["paired_change"]
        lines.extend(
            [
                "",
                " CHANGE FROM PREVIOUS EVALUATION",
                f" Comparison model: step {summary['previous']['step']:,}",
                f" Average margin: {fmt(change['margin']['change'])} | 95% {ci(change['margin']['ci95'])}",
                f" Win share: {100 * change['win_share']['change']:+.1f} percentage points",
            ]
        )
        change_ci = change["win_share"]["ci95"]
        if change_ci is not None:
            lines.append(
                f" 95% interval of the change: [{100 * change_ci[0]:+.1f}; {100 * change_ci[1]:+.1f}] percentage points"
            )
    elif not summary["previous"]:
        lines.append(
            "\n First evaluation of this panel; no previous comparison checkpoint yet."
        )
    if "previous" in summary["models"]:
        old = summary["models"]["previous"]
        lines.append(
            f"\n Comparison model: {old['completed']}/{old['attempted']} games | {old['failed']} errors"
        )
    for name, title in [
        ("by_pair", "OPPONENT PAIRS (left,right)"),
        ("by_seat", "SEATS"),
        ("by_wonder", "WONDER SIDES"),
    ]:
        lines.extend(
            [
                "",
                f" {title}",
                f" {'Group':15} {'Games':>8} {'Margin':>10} {'Win share':>12}",
            ]
        )
        for key, row in m[name].items():
            share = (
                f"{100 * row['win_share']:.1f} %"
                if row["win_share"] is not None
                else "n/a"
            )
            label = (
                ",".join(chr(65 + int(x)) for x in key.split(","))
                if name == "by_pair"
                else key
            )
            lines.append(
                f" {label:15} {row['completed']:8} {fmt(row['margin']):>10} {share:>12}"
            )
    lines.extend(["", " POINT BREAKDOWN (means)"])
    lines.extend(f" {k:20} {v:.2f}" for k, v in m["categories"].items())
    lines.extend(
        [
            "",
            " Intervals use whole seed blocks, not independent individual games.",
            " Subgroups are descriptive; the panel tracks progress against known opponents.",
            " The training league and training counters remain unchanged.",
        ]
    )
    lines.extend(" ERROR: " + reason for reason in m["failure_reasons"])
    if "previous" in summary["models"]:
        lines.extend(
            " ERROR comparison model: " + reason
            for reason in summary["models"]["previous"]["failure_reasons"]
        )
    return "\n".join(lines) + "\n"


def evaluate_run(
    run_dir,
    engine=None,
    evaluation_dir=None,
    blocks=40,
    seed=159001,
    reference_steps=None,
    progress=print,
    workers=1,
    training_control=False,
):
    if blocks < 1 or seed < 0:
        raise ValueError("A positive block count and a nonnegative seed are required.")
    if type(workers) is not int or workers < 1:
        raise ValueError("A positive worker count is required.")
    torch.set_num_threads(1)
    run = Path(run_dir).resolve()
    directory = (
        Path(evaluation_dir).resolve()
        if evaluation_dir
        else run / ("intermediate_evaluation" if training_control else "evaluation")
    )
    with evaluation_lock(directory):
        state, trained, durable_hash = durable_run(run, allow_initial=training_control)
        plan, opponents = prepare_panel(
            directory, run, state, trained, blocks, seed, reference_steps
        )
        probe = DecisionEnv(engine_path=engine)
        try:
            _, info = probe.reset(seed=plan["engine_seeds"][0])
            if info["engine"]["source_commit"] != plan["settings"]["engine_commit"]:
                raise ValueError("Engine commit does not match the evaluation plan.")
            engine_path = probe.engine_path
            engine_metadata = info["engine"]
        finally:
            probe.close()
        session = directory / (
            "eval-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        session.mkdir()
        if training_control:
            # This is the just-committed learner, which may be much newer than
            # the last league snapshot. Never publish it into the training pool.
            torch.save(state, session / "candidate.pt")
            candidate = {
                "step": state["steps"],
                "sha256": file_hash(session / "candidate.pt"),
                "role": "training-control-learner",
            }
            model, _ = load_checkpoint(session / "candidate.pt", state["contract"])
            model.eval().requires_grad_(False)
        else:
            candidate = trained[-1]
            model = copy_snapshot(
                run, candidate, session / "candidate.pt", state["contract"]
            )
        latest = directory / "latest.json"
        previous = None
        models = {"current": model}
        if latest.exists():
            prior = json.loads(latest.read_text())
            if prior["plan_id"] != plan["plan_id"]:
                raise ValueError(
                    "The latest evaluation belongs to a different evaluation plan."
                )
            previous = prior["candidate"]
            oldpath = directory / prior["directory"] / "candidate.pt"
            if file_hash(oldpath) != previous["sha256"]:
                raise ValueError("Previous evaluation model changed.")
            shutil.copy2(oldpath, session / "previous.pt")
            old, _ = load_checkpoint(session / "previous.pt", state["contract"])
            models["previous"] = old.eval().requires_grad_(False)
        manifest = {
            "plan_id": plan["plan_id"],
            "candidate": candidate,
            "previous": previous,
            "durable_checkpoint_sha256": durable_hash,
            "durable_step": state["steps"],
            "engine": engine_metadata,
            "engine_binary": str(engine_path),
            "engine_binary_sha256": file_hash(engine_path),
            "contract": state["contract"],
            "training_provenance": state["metrics"]["provenance"],
            "evaluator_source_sha256": file_hash(Path(__file__)),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch_threads": torch.get_num_threads(),
                "workers": workers,
                "versions": {
                    p: importlib.metadata.version(p)
                    for p in ("torch", "numpy", "gymnasium")
                },
            },
            "plan": plan,
            "status": "running",
        }
        atomic_json(session / "manifest.json", manifest)
        all_rows = {}
        reference_count = len(plan["references"])
        total = blocks * reference_count * reference_count * 3
        started = time.monotonic()
        executor = None
        groups = None
        try:
            if workers > 1:
                context = mp.get_context("spawn")
                groups = context.SimpleQueue()
                model_hashes = {
                    name: file_hash(
                        session / (("candidate" if name == "current" else name) + ".pt")
                    )
                    for name in models
                }
                # Worker filenames use the same names as the immutable artifacts.
                model_hashes = {
                    ("candidate" if name == "current" else name): sha
                    for name, sha in model_hashes.items()
                }
                executor = ProcessPoolExecutor(
                    max_workers=min(workers, total),
                    mp_context=context,
                    initializer=_init_evaluation_worker,
                    initargs=(
                        str(session),
                        str(directory / "panel"),
                        state["contract"],
                        model_hashes,
                        plan,
                        str(engine_path),
                        groups,
                    ),
                )
            for name, policy in models.items():
                rows = []
                coordinates = list(
                    itertools.product(
                        range(blocks),
                        range(reference_count),
                        range(reference_count),
                        range(3),
                    )
                )
                if progress:
                    progress(
                        f"\n{name}: {total} games | {reference_count} fixed opponents, {reference_count**2} pairs, three seats | {workers} processes"
                    )
                source = (
                    _parallel_rows(
                        executor,
                        "candidate" if name == "current" else name,
                        coordinates,
                        plan,
                    )
                    if executor
                    else (
                        play_game(policy, opponents, engine_path, plan, *c)
                        for c in coordinates
                    )
                )
                path = session / f"{name}.jsonl"
                try:
                    with path.open("w") as stream:
                        for row in source:
                            rows.append(row)
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            stream.flush()
                            if progress and (len(rows) % 60 == 0 or len(rows) == total):
                                progress(
                                    f"  {name}: {len(rows)}/{total} games | {time.monotonic() - started:.1f} s"
                                )
                finally:
                    source.close()
                rows.sort(key=_row_key)
                temporary = path.with_suffix(".jsonl.tmp")
                temporary.write_text(
                    "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
                )
                temporary.replace(path)
                all_rows[name] = rows
        finally:
            if executor:
                executor.shutdown(wait=True, cancel_futures=True)
            if groups:
                # Also collect C++ children of a worker killed by a broken pool.
                while not groups.empty():
                    pid, grouped = groups.get()
                    if grouped:
                        try:
                            os.killpg(pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                groups.close()
        summaries = {name: summarize(rows, plan) for name, rows in all_rows.items()}
        summary = {
            "plan_id": plan["plan_id"],
            "candidate": candidate,
            "previous": previous,
            "blocks": blocks,
            "references": plan["references"],
            "models": summaries,
            "paired_change": paired(all_rows["current"], all_rows["previous"], plan)
            if previous
            else None,
            "valid": all(m["valid"] for m in summaries.values()),
            "directory": session.name,
        }
        atomic_json(session / "summary.json", summary)
        text = report_text(summary)
        (session / "report.txt").write_text(text)
        atomic_json(
            session / "manifest.json",
            {**manifest, "status": "completed" if summary["valid"] else "failed"},
        )
        from sevenwonders_rl.evaluation.dashboard import write_dashboard

        write_dashboard(session)
        if summary["valid"]:
            atomic_json(
                latest,
                {
                    "plan_id": plan["plan_id"],
                    "candidate": candidate,
                    "directory": session.name,
                },
            )
        if progress:
            progress(text)
            progress(f"Results: {session}")
        return summary, session
