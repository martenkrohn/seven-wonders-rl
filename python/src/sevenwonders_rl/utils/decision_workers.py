"""Owned engine processes with a small control pipe and shared observations."""

from collections import OrderedDict
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import signal
import tempfile
import time
import traceback

import numpy as np

from sevenwonders_rl.features.decision_memory import field_specs


def layout():
    fields = {}
    offset = 0
    for key, (shape, dtype) in field_specs().items():
        offset = (offset + 7) // 8 * 8
        fields[key] = (shape, np.dtype(dtype), offset)
        offset += int(np.prod(shape)) * np.dtype(dtype).itemsize
    return fields, offset


def observation_views(memory):
    return {
        key: np.ndarray(shape, dtype, buffer=memory.buf, offset=offset)
        for key, (shape, dtype, offset) in layout()[0].items()
    }


def _terminate(signum, frame):
    raise SystemExit("Worker terminated")


def _worker(connection, memory_name, settings, snapshot_dir, cache_size, workdir):
    import torch
    from sevenwonders_rl.envs import DecisionEnv
    from sevenwonders_rl.selfplay.decision_league import file_hash
    from sevenwonders_rl.utils.decision_ppo import load_checkpoint

    # A worker owns an entire process group, including its C++ children. Parent
    # cleanup can therefore collect an engine even if the worker itself crashes.
    grouped = hasattr(os, "setsid")
    if grouped:
        os.setsid()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _terminate)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tempfile.tempdir = workdir
    memory = SharedMemory(name=memory_name)
    views = observation_views(memory)
    env = DecisionEnv(**settings)
    cache = OrderedDict()

    def policy(row, contract):
        if Path(row["name"]).name != row["name"]:
            raise ValueError("Invalid snapshot name")
        path = Path(snapshot_dir) / row["name"]
        if file_hash(path) != row["sha256"]:
            raise ValueError("Frozen league checkpoint changed during training")
        key = row["sha256"]
        if key not in cache:
            model, data = load_checkpoint(path, contract)
            if (
                data.get("role") != "frozen-league-snapshot"
                or data["steps"] != row["step"]
            ):
                raise ValueError("Invalid frozen league checkpoint")
            cache[key] = model.eval().requires_grad_(False)
        cache.move_to_end(key)
        model = cache[key]
        while len(cache) > cache_size:
            cache.popitem(last=False)
        return model.select_action

    try:
        connection.send(("ready", grouped))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                env.agent_seat = payload["agent_seat"]
                env.opponents = (
                    {
                        (env.agent_seat + 2) % 3: policy(
                            payload["left"], payload["contract"]
                        ),
                        (env.agent_seat + 1) % 3: policy(
                            payload["right"], payload["contract"]
                        ),
                    }
                    if snapshot_dir
                    else {}
                )
                obs, info = env.reset(seed=payload["seed"])
                result = {
                    "done": False,
                    "reward": 0.0,
                    "engine": info["engine"],
                    "schema_hash": info["feature_contract"]["schema_hash"],
                }
            elif command == "step":
                obs, reward, done, truncated, info = env.step(payload)
                if truncated or info["transition"]["receipt"]["status"] != "applied":
                    raise RuntimeError("Unapplied worker transition")
                result = {
                    "done": done,
                    "reward": reward,
                    "totals": info.get("final_totals"),
                }
            else:
                raise ValueError("Unknown worker command")
            for key, value in obs.items():
                np.copyto(views[key], value)
            connection.send(("ok", result))
    except EOFError:
        pass  # A killed parent closes the only other end of the control pipe.
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        env.close()
        views.clear()
        memory.close()
        connection.close()
        # EOF also runs after a hard-killed parent. The last surviving worker can
        # remove the now-empty pool directory without a parent finalizer.
        try:
            Path(workdir).rmdir()
            Path(workdir).parent.rmdir()
        except OSError:
            pass


class DecisionWorkers:
    def __init__(self, count, env, snapshot_dir=None, cache_size=64):
        self.timeout = max(30.0, 4 * env.timeout)
        self.memories = []
        self.views = []
        self.connections = []
        self.processes = []
        self.groups = set()
        self.pending = set()
        self.workdir = tempfile.TemporaryDirectory(prefix="sevenwonders-workers-")
        settings = {
            "engine_path": str(env.engine_path),
            "agent_seat": env.agent_seat,
            "reward_scale": env.reward_scale,
            "allow_development": env.allow_development,
            "max_wait_seconds": env.timeout,
        }
        context = mp.get_context("spawn")
        try:
            for slot in range(count):
                workdir = Path(self.workdir.name) / str(slot)
                workdir.mkdir()
                memory = SharedMemory(create=True, size=layout()[1])
                self.memories.append(memory)
                self.views.append(observation_views(memory))
                parent, child = context.Pipe()
                self.connections.append(parent)
                process = context.Process(
                    target=_worker,
                    args=(
                        child,
                        memory.name,
                        settings,
                        str(snapshot_dir) if snapshot_dir else None,
                        cache_size,
                        str(workdir),
                    ),
                )
                process.start()
                child.close()
                self.processes.append(process)
            for slot in range(count):
                kind, grouped = self._read(slot)
                if kind != "ready":
                    raise RuntimeError(f"Game worker {slot} could not start: {grouped}")
                if grouped:
                    self.groups.add(slot)
        except BaseException:
            self.close()
            raise

    def _read(self, slot):
        connection = self.connections[slot]
        deadline = time.monotonic() + self.timeout
        while not connection.poll(0.1):
            if not self.processes[slot].is_alive():
                raise RuntimeError(f"Game worker {slot} has exited")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Game worker {slot} is not responding")
        try:
            return connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"Game worker {slot} has failed") from exc

    def send(self, slot, command, payload):
        if slot in self.pending:
            raise RuntimeError("Worker already has an outstanding command")
        try:
            self.connections[slot].send((command, payload))
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"Game worker {slot} has failed") from exc
        self.pending.add(slot)

    def receive(self, slot):
        if slot not in self.pending:
            raise RuntimeError("Worker has no outstanding command")
        kind, value = self._read(slot)
        self.pending.remove(slot)
        if kind != "ok":
            raise RuntimeError(f"Game worker {slot}: {value}")
        return value

    def observation(self, slot):
        if slot in self.pending:
            raise RuntimeError("Observation is still being written")
        # Rollouts must own their arrays before the next worker write.
        return {key: value.copy() for key, value in self.views[slot].items()}

    def close(self):
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for slot, process in enumerate(self.processes):
            process.join(timeout=2)
            if process.is_alive() or process.exitcode:
                if slot in self.groups:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                elif process.is_alive():
                    process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    if slot in self.groups:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        process.kill()
                    process.join(timeout=2)
        for connection in self.connections:
            connection.close()
        self.views.clear()
        for memory in self.memories:
            memory.close()
            memory.unlink()
        self.memories.clear()
        self.processes.clear()
        self.connections.clear()
        self.workdir.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class LocalDecisionWorker:
    """In-process collector implementing the same interface as process workers.

    A single game supports explicit Python callbacks without another PPO path.
    The league owns immutable opponent loading and keeps learner RNG isolated.
    """

    def __init__(self, env, league=None, callbacks=None):
        self.env = env
        self.league = league
        self.callbacks = callbacks or {}
        self.pending = None
        self.obs = None

    def send(self, slot, command, payload):
        if slot != 0 or self.pending is not None:
            raise RuntimeError("Invalid or busy local game slot")
        if command == "reset":
            self.env.agent_seat = payload["agent_seat"]
            seat = self.env.agent_seat
            self.env.opponents = (
                {
                    (seat + 2) % 3: self.league._policy(payload["pair_left_step"]),
                    (seat + 1) % 3: self.league._policy(payload["pair_right_step"]),
                }
                if self.league
                else self.callbacks
            )
            self.obs, info = self.env.reset(seed=payload["seed"])
            self.pending = {
                "done": False,
                "reward": 0.0,
                "engine": info["engine"],
                "schema_hash": info["feature_contract"]["schema_hash"],
            }
        elif command == "step":
            self.obs, reward, done, truncated, info = self.env.step(payload)
            if truncated or info["transition"]["receipt"]["status"] != "applied":
                raise RuntimeError("Unapplied worker transition")
            self.pending = {
                "done": done,
                "reward": reward,
                "totals": info.get("final_totals"),
            }
        else:
            raise ValueError("Unknown worker command")

    def receive(self, slot):
        if slot != 0 or self.pending is None:
            raise RuntimeError("No outstanding local command")
        result, self.pending = self.pending, None
        return result

    def observation(self, slot):
        if slot != 0 or self.pending is not None:
            raise RuntimeError("Observation is not ready")
        return {key: value.copy() for key, value in self.obs.items()}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.env.close()
