"""Protocol-3 adapter for the existing ordered-pair PFSP rule.

The durable learner checkpoint owns the snapshot inventory, sampler state and
journal lengths. Directory discovery and derived reports are never authoritative.
Frozen networks contain no per-seat/per-game memory; DecisionEnv owns that state.
"""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
from pathlib import Path
import os

import numpy as np
import torch

from .opponent_pool import OpponentPool
from .pair_pfsp import PairPFSPPool


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def training_directory(output, resume):
    """Reject overlapping writers and accidental fresh starts over existing runs."""
    if output is None:
        yield
        return
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another trainer holds this output directory") from exc
        try:
            if not resume and any(directory.glob("*.pt")):
                raise ValueError(
                    "Fresh training requires a new output directory; use resume"
                )
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@dataclass(frozen=True)
class LeagueConfig:
    snapshot_every: int = 65_536
    pool_max: int = 64
    power: float = 2.0
    control_fraction: float = 0.10
    half_life: float = 131_072.0
    prior_games: float = 3.0
    rotate_seats: bool = True

    def __post_init__(self):
        if (
            type(self.snapshot_every) is not int
            or self.snapshot_every < 1
            or type(self.pool_max) is not int
            or self.pool_max < 1
            or type(self.rotate_seats) is not bool
            or not all(
                math.isfinite(x) and x > 0
                for x in (self.power, self.half_life, self.prior_games)
            )
            or not math.isfinite(self.control_fraction)
            or not 0 < self.control_fraction <= 1
        ):
            raise ValueError("Invalid decision-league configuration")


class DecisionLeague:
    def __init__(
        self,
        directory,
        contract,
        config,
        seed,
        base_seat=0,
        state=None,
        durable_step=0,
        training_contract=None,
    ):
        self.directory = Path(directory).resolve()
        self.snapshot_dir = self.directory / "league"
        self.contract, self.config = contract, config
        self.base_seat = base_seat
        self.training_contract = training_contract
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 98171]))
        underlying = OpponentPool(
            str(self.snapshot_dir), max_size=config.pool_max, auto_refresh=False
        )
        underlying._checkpoints = []  # Explicit durable inventory, never scan files.
        self.pool = PairPFSPPool(
            underlying,
            power=config.power,
            control_fraction=config.control_fraction,
            half_life=config.half_life,
            prior_games=config.prior_games,
        )
        self.inventory = {}
        self.cache = OrderedDict()
        self.selections = self.completed = 0
        self.reserved_seed_max = -1
        self.active_by_slot = {}
        if state is None:
            if self.snapshot_dir.exists() and any(self.snapshot_dir.iterdir()):
                raise ValueError("Fresh league requires an empty snapshot directory")
            if any(
                (self.directory / name).exists()
                for name in ("opponents.jsonl", "episodes.jsonl")
            ):
                raise ValueError("Fresh league cannot reuse existing journals")
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._restore(state, durable_step)

    def _restore(self, state, durable_step):
        # Collector width is an explicit runtime choice, inherited by the CLI.
        # Switching it abandons open games just like an ordinary resume; all
        # other training settings must still match the durable contract.
        def core(value):
            return (
                {k: v for k, v in value.items() if k != "collector_workers"}
                if isinstance(value, dict)
                else value
            )

        if (
            state.get("version") != 1
            or state.get("config") != asdict(self.config)
            or state.get("base_seat") != self.base_seat
            or core(state.get("training_contract")) != core(self.training_contract)
        ):
            raise ValueError("Decision-league resume configuration mismatch")
        self.pool.load_state_dict(state["pfsp"])
        inventory = state["snapshots"]
        if not inventory or inventory[0]["step"] != 0:
            raise ValueError("League checkpoint is missing its initial snapshot")
        for row in inventory:
            step = row["step"]
            if (
                type(step) is not int
                or step < 0
                or step > durable_step
                or (step and step % self.config.snapshot_every)
                or step in self.inventory
                or row["name"] != f"checkpoint_{step}.pt"
            ):
                raise ValueError("Invalid durable league inventory")
            path = self.snapshot_dir / row["name"]
            if not path.is_file() or file_hash(path) != row["sha256"]:
                raise ValueError(f"League snapshot missing or changed: {path}")
            self.inventory[step] = dict(row)
            self.pool.register(path)
        expected_steps = list(range(0, durable_step + 1, self.config.snapshot_every))
        if sorted(self.inventory) != expected_steps:
            raise ValueError("Durable league inventory has missing snapshot boundaries")
        for row in state["pfsp"]["results"]:
            if (
                row["left"] not in self.inventory
                or row["right"] not in self.inventory
                or row["step"] > durable_step
            ):
                raise ValueError("PFSP result is outside the durable league inventory")
        self.selections, self.completed = state["selections"], state["completed"]
        if (
            type(self.selections) is not int
            or type(self.completed) is not int
            or not 0 <= self.completed <= self.selections
            or sum(r["games"] for r in state["pfsp"]["results"]) != self.completed
        ):
            raise ValueError("Invalid league episode counters")
        self.rng.bit_generator.state = state["rng"]
        self.reserved_seed_max = state.get("reserved_seed_max", -1)
        self._repair_journal("opponents.jsonl", self.selections)
        self._repair_journal("episodes.jsonl", self.completed)
        # The previous engine game is abandoned, never scored as a loss. Its
        # selection remains documented and the saved sampler continues after it.
        self.active_by_slot.clear()

    @property
    def active(self):
        """The serial slot, retained for existing callers and checkpoint readers."""
        return self.active_by_slot.get(0)

    def _repair_journal(self, name, count):
        path = self.directory / name
        lines = path.read_bytes().splitlines(keepends=True) if path.exists() else []
        if len(lines) < count:
            raise ValueError(f"Committed league journal is incomplete: {name}")
        for index, line in enumerate(lines[:count]):
            if json.loads(line)["index"] != index:
                raise ValueError(f"Invalid committed journal order: {name}")
        if name == "opponents.jsonl":
            # Reserve seeds of games started after the last durable PPO update
            # too. A second interruption during repair must not lose them.
            rolled = path.with_suffix(".rolled-back")
            previous = rolled.read_bytes().splitlines() if rolled.exists() else []
            for line in [*lines, *previous]:
                try:
                    seed = json.loads(line)["seed"]
                except (ValueError, KeyError):
                    continue
                self.reserved_seed_max = max(self.reserved_seed_max, seed)
        if len(lines) > count:
            with path.with_suffix(".rolled-back").open("ab") as stream:
                stream.write(b"".join(lines[count:]))
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(b"".join(lines[:count]))
            temporary.replace(path)

    def _append(self, name, row):
        with (self.directory / name).open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def publish(self, model, step):
        if step in self.inventory:
            raise ValueError("Published league snapshots are immutable")
        path = self.snapshot_dir / f"checkpoint_{step}.pt"
        if path.exists():
            # A crash may leave a snapshot outside the authoritative checkpoint.
            orphan_dir = self.snapshot_dir / "orphans"
            orphan_dir.mkdir(exist_ok=True)
            orphan = orphan_dir / f"checkpoint_{step}-{file_hash(path)}.pt"
            path.replace(orphan)
        payload = {
            "format": "decision-ppo-v2",
            "contract": self.contract,
            "architecture": model.architecture,
            "model": model.state_dict(),
            "steps": step,
            "role": "frozen-league-snapshot",
        }
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        self.inventory[step] = {
            "step": step,
            "name": path.name,
            "sha256": file_hash(path),
        }
        self.pool.register(path)

    def _policy(self, step):
        from sevenwonders_rl.utils.decision_ppo import load_checkpoint

        row = self.inventory[step]
        path = self.snapshot_dir / row["name"]
        if file_hash(path) != row["sha256"]:
            raise ValueError("Frozen league checkpoint changed during training")
        if step not in self.cache:
            # Cache misses must not consume the learner's Torch sampling stream.
            with torch.random.fork_rng(devices=[]):
                model, data = load_checkpoint(path, self.contract)
            if data.get("role") != "frozen-league-snapshot" or data["steps"] != step:
                raise ValueError("Invalid frozen league checkpoint")
            model.eval().requires_grad_(False)
            self.cache[step] = model
        self.cache.move_to_end(step)
        model = self.cache[step]
        while len(self.cache) > self.config.pool_max:
            self.cache.popitem(last=False)
        return model.select_action

    def reserve_episode(self, seed, step, slot=0):
        """Durably reserve a seed/pair before dispatching an engine reset."""
        if type(slot) is not int or slot < 0:
            raise ValueError("Invalid league slot")
        if slot in self.active_by_slot:
            raise ValueError("Cannot replace opponents during an active episode")
        if seed <= self.reserved_seed_max:
            raise ValueError("League episode seed was already reserved")
        seat = (
            (self.base_seat + self.selections) % 3
            if self.config.rotate_seats
            else self.base_seat
        )
        _, _, metadata = self.pool.sample_pair(self.rng, step)
        left, right = metadata["pair_left_step"], metadata["pair_right_step"]
        row = {
            "index": self.selections,
            "selection_step": step,
            "seed": seed,
            "agent_seat": seat,
            "slot": slot,
            **metadata,
            "left": dict(self.inventory[left]),
            "right": dict(self.inventory[right]),
        }
        self._append("opponents.jsonl", row)
        self.selections += 1
        self.reserved_seed_max = seed
        self.active_by_slot[slot] = row
        return row

    def start_episode(self, env, seed, step):
        row = self.reserve_episode(seed, step)
        seat = row["agent_seat"]
        policies = {
            (seat + 2) % 3: self._policy(row["pair_left_step"]),
            (seat + 1) % 3: self._policy(row["pair_right_step"]),
        }
        # Set assignment before reset: reset may itself advance opponents.
        env.agent_seat, env.opponents = seat, policies
        obs, info = env.reset(seed=seed)
        return obs, info

    def finish_episode(self, totals, step, slot=0):
        if slot not in self.active_by_slot:
            raise ValueError("No active league episode to score")
        active = self.active_by_slot[slot]
        self.pool.record_result(
            active["pair_left_step"],
            active["pair_right_step"],
            totals,
            active["agent_seat"],
            step,
        )
        self._append(
            "episodes.jsonl",
            {
                "index": self.completed,
                "selection_index": active["index"],
                "seed": active["seed"],
                "agent_seat": active["agent_seat"],
                "slot": slot,
                "result_step": step,
                "totals": list(totals),
                "left_step": active["pair_left_step"],
                "right_step": active["pair_right_step"],
            },
        )
        self.completed += 1
        del self.active_by_slot[slot]

    def state_dict(self):
        return {
            "version": 1,
            "config": asdict(self.config),
            "base_seat": self.base_seat,
            "training_contract": self.training_contract,
            "pfsp": self.pool.state_dict(),
            "rng": self.rng.bit_generator.state,
            "snapshots": [dict(self.inventory[s]) for s in sorted(self.inventory)],
            "selections": self.selections,
            "completed": self.completed,
            "reserved_seed_max": self.reserved_seed_max,
            "active_episode": dict(self.active) if self.active else None,
            "active_episodes": {
                str(slot): dict(row) for slot, row in self.active_by_slot.items()
            },
        }

    def report(self, step):
        return {
            **self.pool.report(step),
            "format": "decision-league-v1",
            "config": asdict(self.config),
            "selections": self.selections,
            "completed_games": self.completed,
            "abandoned_games": self.selections
            - self.completed
            - len(self.active_by_slot),
            "active_episode": self.active,
            "active_episodes": self.active_by_slot,
            "active_games": len(self.active_by_slot),
            "schema_hash": self.contract["schema_hash"],
        }
