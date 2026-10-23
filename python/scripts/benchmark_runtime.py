"""Paired transport/training benchmarks; fingerprints guard behavior equivalence."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from sevenwonders_rl.envs import SevenWondersEnv
from sevenwonders_rl.envs.sevenwonders_env import EngineError
from sevenwonders_rl.utils.checkpoints import file_hash
from sevenwonders_rl.utils.runtime import atomic_json, run_io_root
from sevenwonders_rl.utils.seeding import derive_seed
from sevenwonders_rl.utils.timing import TimingBuckets

ROOT = Path(__file__).resolve().parents[2]


def filesystem(path):
    try:
        return subprocess.check_output(["findmnt", "-T", str(path), "-n", "-o", "FSTYPE,SOURCE"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def usage():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"parent_cpu_seconds": own.ru_utime + own.ru_stime,
            "child_cpu_seconds": children.ru_utime + children.ru_stime,
            "parent_peak_rss_kib_session": own.ru_maxrss,
            "child_peak_rss_kib_session": children.ru_maxrss}


def transport_trial(io_root, games, n_envs, seed, poll_us):
    timers = TimingBuckets()
    digest = hashlib.sha256()
    completed, transitions = 0, 0
    before = usage()
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="transport-", dir=io_root) as directory:
        envs = [SevenWondersEnv(repo_root=str(ROOT), io_dir=str(Path(directory) / f"env_{i}" / "io"),
                timers=timers, engine_poll_us=poll_us) for i in range(min(n_envs, games))]
        try:
            for first in range(0, games, len(envs)):
                batch = envs[:min(len(envs), games-first)]
                observations = []
                for i, env in enumerate(batch):
                    with timers.time("reset"):
                        obs, info = env.reset(seed=derive_seed(seed, "engine", first+i))
                    observations.append(obs)
                    digest.update(json.dumps(info["status"], sort_keys=True).encode())
                for _ in range(21):
                    previous_turns = [env.send_action(int(np.flatnonzero(obs["action_mask"])[0]))
                                      for env, obs in zip(batch, observations)]
                    for i, (env, previous) in enumerate(zip(batch, previous_turns)):
                        obs, _, terminated, truncated, info = env.wait_next(previous)
                        if truncated or not info.get("transition_valid", True):
                            raise EngineError(info.get("truncated_reason", "invalid"), "Benchmark transition failed")
                        observations[i] = obs
                        transitions += 1
                        completed += int(terminated)
                        digest.update(json.dumps(info["status"], sort_keys=True).encode())
        finally:
            for env in envs:
                env.close()
    elapsed = time.perf_counter() - start
    after = usage()
    return {"wall_seconds": elapsed, "steps_per_second": transitions / elapsed,
            "games_per_second": completed / elapsed, "completed": completed, "failed": 0,
            "transitions": transitions, "fingerprint": digest.hexdigest(),
            "cpu": {key: after[key]-before[key] if "cpu_seconds" in key else after[key] for key in after},
            "timing": timers.as_dict(elapsed)}


def training_trial(io_root, output, steps, n_envs, horizon, seed, poll_us):
    output.mkdir(parents=True)
    before = usage()
    start = time.perf_counter()
    with (output / "process.log").open("w") as log:
        # Trainer requires an empty output directory; process log is outside it.
        target = output / "training"
        subprocess.run([sys.executable, str(ROOT / "python/scripts/ppo_training/train_unified.py"),
            "--mode", "fixed", "--seed", str(seed), "--total-timesteps", str(steps),
            "--n-envs", str(n_envs), "--rollout-horizon", str(horizon),
            "--eval-every", str(steps+1), "--save-every", str(steps+1),
            "--engine-poll-us", str(poll_us), "--io-root", str(io_root), "--out-dir", str(target)],
            cwd=ROOT / "python", check=True, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    after = usage()
    result = json.loads((target / "result.json").read_text())
    import torch
    checkpoint = torch.load(target / "latest.pt", weights_only=True, map_location="cpu")
    digest = hashlib.sha256()
    for key, value in sorted(checkpoint["state_dict"].items()):
        digest.update(key.encode())
        digest.update(value.numpy().tobytes())
    episode_log = target / "episodes.jsonl"
    episodes = [json.loads(line) for line in episode_log.read_text().splitlines()] if episode_log.exists() else []
    completed = sum(e["status"] == "completed" for e in episodes)
    return {"wall_seconds": elapsed, "training_seconds": result["elapsed_seconds"],
            "steps_per_second": steps / elapsed, "learner_steps_per_second": result["steps_per_second"],
            "games_per_second": completed / elapsed, "completed": completed, "failed": 0,
            "transitions": steps, "fingerprint": digest.hexdigest(),
            "cpu": {key: after[key]-before[key] if "cpu_seconds" in key else after[key] for key in after},
            "timing": json.loads((target / "timing.json").read_text()), "artifacts": str(target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["transport", "training", "both"], default="both")
    parser.add_argument("--io-roots", nargs="+", default=[str(ROOT / "io_benchmark"), "/dev/shm/sevenwonders"])
    parser.add_argument("--engine-poll-us", nargs="+", type=int, default=[0, 1000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--total-timesteps", type=int, default=2048)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--rollout-horizon", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    if min(args.repeats, args.games, args.total_timesteps, args.n_envs, args.rollout_horizon) <= 0 or args.seed < 0:
        parser.error("Counts must be positive and seed nonnegative")
    if args.total_timesteps % args.n_envs or any(not 0 <= p <= 1_000_000 for p in args.engine_poll_us):
        parser.error("Invalid step budget or polling interval")
    output = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_io_root("runs/benchmarks", "paired")
    if args.out_dir and output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    roots = [Path(p).expanduser().resolve() for p in args.io_roots]
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
    modes = ["transport", "training"] if args.mode == "both" else [args.mode]
    configurations = list(itertools.product(roots, args.engine_poll_us))
    atomic_json(output / "manifest.json", {"config": vars(args),
        "engine_sha256": file_hash(ROOT / "build/7Wonders"),
        "filesystems": {str(root): filesystem(root) for root in roots},
        "free_bytes_before": {str(root): shutil.disk_usage(root).free for root in roots},
        "notes": "Warm-up excluded. Alternating configuration order. RSS is session high-water mark. Training wall includes startup/checkpoints; evaluation disabled."})
    def trial(mode, root, poll_us, seed, tag, warmup=False):
        if mode == "transport":
            return transport_trial(root, min(args.games, 2) if warmup else args.games, args.n_envs, seed, poll_us)
        return training_trial(root, output / tag,
            args.n_envs * min(args.rollout_horizon, 8) if warmup else args.total_timesteps,
            args.n_envs, args.rollout_horizon, seed, poll_us)
    for mode in modes:
        for i, (root, poll) in enumerate(configurations):
            trial(mode, root, poll, derive_seed(args.seed, "evaluation", i), f"warmup-{mode}-{i}", True)
    records = []
    for repeat in range(args.repeats):
        order = list(enumerate(configurations))
        if repeat % 2:
            order.reverse()
        seed = derive_seed(args.seed, "evaluation", repeat)
        for mode in modes:
            for config_id, (root, poll) in order:
                tag = f"{mode}-r{repeat}-c{config_id}"
                result = trial(mode, root, poll, seed, tag)
                record = {"repeat": repeat, "mode": mode, "config_id": config_id,
                          "io_root": str(root), "engine_poll_us": poll, "seed": seed, **result}
                records.append(record)
                with (output / "trials.jsonl").open("a") as f:
                    f.write(json.dumps(record, allow_nan=False) + "\n")
                print(f"{tag}: {result['steps_per_second']:.1f} steps/s, completed={result['completed']}", flush=True)
    summary = []
    for mode in modes:
        for config_id, (root, poll) in enumerate(configurations):
            group = [r for r in records if r["mode"] == mode and r["config_id"] == config_id]
            ratios = []
            equivalent = True
            for r in group:
                baseline = next(b for b in records if b["mode"] == mode and b["repeat"] == r["repeat"] and b["config_id"] == 0)
                ratios.append(r["steps_per_second"] / baseline["steps_per_second"])
                equivalent &= r["fingerprint"] == baseline["fingerprint"]
            speeds = [r["steps_per_second"] for r in group]
            summary.append({"mode": mode, "io_root": str(root), "engine_poll_us": poll,
                "repeats": len(group), "steps_per_second_median": statistics.median(speeds),
                "steps_per_second_std": statistics.pstdev(speeds),
                "paired_speed_ratio_median": statistics.median(ratios),
                "behavior_identical_to_baseline": equivalent,
                "child_cpu_seconds_median": statistics.median(r["cpu"]["child_cpu_seconds"] for r in group)})
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not all(row["behavior_identical_to_baseline"] for row in summary):
        raise SystemExit("Behavior fingerprints differ; do not accept a speed comparison as equivalent")


if __name__ == "__main__":
    main()
