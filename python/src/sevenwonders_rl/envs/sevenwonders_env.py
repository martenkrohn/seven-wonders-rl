# bots/sevenwonders_env.py
from __future__ import annotations

import json
import os
import subprocess
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError("Install gymnasium: pip install gymnasium") from e

from sevenwonders_rl.features.obs_builder import build_obs, compute_action_mask
from sevenwonders_rl.features.action_feature_schema import K as K_ACTION_FEATS
from sevenwonders_rl.policies.opponent_sampler import OpponentSampler
from sevenwonders_rl.policies.registry import make_policy
from sevenwonders_rl.utils.seeding import derive_seed
from sevenwonders_rl.utils.runtime import run_io_root


# ----------------------------
# Fixed action encoding (28)
# ----------------------------
# action_index = action_type * 7 + slot
# action_type:
#   0 = build_structure
#   1 = build_wonder
#   2 = build_hand_free
#   3 = discard
ACTION_TYPES = ("build_structure", "build_wonder", "build_hand_free", "discard")
N_SLOTS = 7
N_ACTIONS = len(ACTION_TYPES) * N_SLOTS


def decode_action(action_index: int) -> Tuple[int, int]:
    action_type = int(action_index) // N_SLOTS
    slot = int(action_index) % N_SLOTS
    return action_type, slot


def encode_action(action_type: int, slot: int) -> int:
    return int(action_type) * N_SLOTS + int(slot)


class EngineError(RuntimeError):
    """A failed exchange, never a valid MDP transition."""
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class SeatPolicy:
    """Callable policy: (status, seat_id, mask, rng) -> action_index"""
    name: str
    policy: Optional[Any] = None
    rng: Optional[np.random.Generator] = None

    def __call__(
        self,
        status: Dict[str, Any],
        seat_id: int,
        mask: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        if rng is None:
            rng = self.rng
        if rng is None:
            rng = np.random.default_rng()

        legal = np.flatnonzero(mask == 1)
        if legal.size == 0:
            return 0

        # If a concrete policy object was supplied, ALWAYS use it.
        if self.policy is not None:
            obs = build_obs(status, seat_id)
            a = int(self.policy.select_action(obs, rng))

            # Hard safety: policy must return legal. If not, fallback.
            if 0 <= a < len(mask) and mask[a] == 1:
                return a
            return int(rng.choice(legal))

        # If no policy but name is not "random", raise to prevent silent fallback
        if self.name != "random":
            raise ValueError(
                f"SeatPolicy '{self.name}' has no policy instance attached. "
                f"Use SeatPolicy('{self.name}', policy=my_policy_instance) or "
                f"SeatPolicy('random') for random legal actions."
            )

        # Otherwise: random legal (name is just metadata)
        return int(rng.choice(legal))


class SevenWondersEnv(gym.Env):
    """
    Gymnasium environment wrapper for your C++ 7 Wonders engine.

    - Controls all 3 seats each turn.
    - Agent controls `agent_seat`.
    - Other seats controlled by `opponent_policies` (default: random).
    - Uses fixed action space: Discrete(28) = 4 action_types * 7 hand slots.
    - Provides action mask each step.

    Observation (minimal, Step 1 + hand):
      obs = {
        "global_table": int32[3, 7]  # per seat: coins, shields, gear, compass, tablet, raw_total, manu_total
        "wonder_stage": int32[3]
        "hand_ids": int32[7]         # agent's hand ids padded with -1
        "action_mask": int8[28]      # legal actions for agent
      }
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        repo_root: str = ".",
        engine_path: str = "build/7Wonders",
        io_dir: Optional[str] = None,
        agent_seat: int = 0,
        opponent_policies: Optional[List[SeatPolicy]] = None,
        max_wait_seconds: float = 5.0,
        invalid_action_fallback: str = "random_legal",  # or "raise"
        reward_scale: float = 1.0,
        seed: Optional[int] = None,
        env_id: int = 0,
        opponent_weights: Optional[Dict[str, float]] = None,
        timers: Optional[Any] = None,
        engine_poll_us: int = 1000,
        poll_seconds: float = 0.001,
        batch_opponents: bool = True,
    ):
        super().__init__()
        assert agent_seat in (0, 1, 2)

        # Handle case where repo_root points to .../python folder
        # instead of the true repo root
        rr = Path(repo_root).resolve()
        if (rr / "build" / "7Wonders").exists():
            base = rr
        elif (rr.parent / "build" / "7Wonders").exists():
            base = rr.parent
        else:
            base = rr  # will fail later with clear error

        self.repo_root = Path(base)
        self.engine_path = (self.repo_root / engine_path).resolve()
        self.io_dir = ((self.repo_root / io_dir).resolve() if io_dir is not None
                       else run_io_root(self.repo_root / "io", "env") / "io")
        if self.io_dir.name != "io":
            raise ValueError("io_dir must end in /io: the engine uses ./io relative to its working directory")

        self.agent_seat = agent_seat
        self.batch_opponents = batch_opponents
        self.max_wait_seconds = float(max_wait_seconds)
        self.invalid_action_fallback = invalid_action_fallback
        self.reward_scale = float(reward_scale)
        self.timers = timers
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")
        if not math.isfinite(self.max_wait_seconds) or self.max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be finite and positive")
        if not 0 <= engine_poll_us <= 1_000_000 or not math.isfinite(poll_seconds) or poll_seconds < 0:
            raise ValueError("Invalid polling interval")
        self.engine_poll_us = engine_poll_us
        self.poll_seconds = poll_seconds
        self._stderr_file = None
        self._engine_seed_rng = np.random.default_rng(seed)
        self.engine_seed = None
        self._seat_rngs = [np.random.default_rng() for _ in range(3)]

        # Seat policies for seats 0..2. Agent seat is None here; applied in step().
        if opponent_policies is None:
            opponent_policies = [SeatPolicy("random"), SeatPolicy("random"), SeatPolicy("random")]
        assert len(opponent_policies) == 3
        self.opponent_policies = opponent_policies

        self.proc: Optional[subprocess.Popen] = None
        self._last_status: Optional[Dict[str, Any]] = None
        self._last_reset_seed: Optional[int] = None
        self._rng = np.random.default_rng(seed)
        
        # Episode-level opponent sampling
        self._env_id = env_id
        self._base_seed = seed if seed is not None else 0
        self._episode_idx = 0
        self._opp_left: Optional[Any] = None
        self._opp_right: Optional[Any] = None
        self._opp_names: Tuple[str, str] = ("", "")
        
        if opponent_weights is not None:
            self._opponent_sampler: Optional[OpponentSampler] = OpponentSampler(
                weights=opponent_weights,
                seed=self._base_seed,
                env_id=env_id
            )
        else:
            self._opponent_sampler = None

        # Gym spaces (fixed shapes)
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Observation space with backward compatible keys + new richer features
        self.observation_space = spaces.Dict(
            {
                # Backward compatible keys
                "global_table": spaces.Box(low=-1000, high=10000, shape=(3, 7), dtype=np.int32),
                "science_jokers": spaces.Box(low=0, high=10, shape=(3,), dtype=np.int32),
                "table_discounts": spaces.Box(low=0, high=1, shape=(3, 3), dtype=np.int32),
                "wonder_stage": spaces.Box(low=0, high=10, shape=(3,), dtype=np.int32),
                "hand_ids": spaces.Box(low=-1, high=10000, shape=(7,), dtype=np.int32),
                "action_mask": spaces.Box(low=0, high=1, shape=(N_ACTIONS,), dtype=np.int8),
                
                # Per-action features (Step 4)
                "action_feats": spaces.Box(
                    low=-1e9, high=1e9, shape=(N_ACTIONS, K_ACTION_FEATS), dtype=np.float32
                ),
                
                # New richer features (Step 1)
                # Points breakdown: [civilian, commercial, guild, military, scientific, wonder, total]
                # Military can be negative, others non-negative
                "table_points": spaces.Box(low=-50, high=300, shape=(3, 7), dtype=np.int32),
                
                # Resource production: [wood, ore, clay, stone, loom, glass, papyrus]
                "table_prod": spaces.Box(low=0, high=10, shape=(3, 7), dtype=np.int32),
                
                # Card counts: [civilian, commercial, guild, military, scientific, raw_material, manufactured_goods]
                "table_amount": spaces.Box(low=0, high=20, shape=(3, 7), dtype=np.int32),
                
                # Game metadata: [era, turn, clockwise]
                # era: 1-3, turn: 0-20, clockwise: 0-1
                "game_meta": spaces.Box(low=np.array([1, 0, 0]), high=np.array([4, 21, 1]), shape=(3,), dtype=np.int32),
            }
        )

        self.io_dir.mkdir(parents=True, exist_ok=True)

    # ------------
    # IO utilities
    # ------------
    def _player_file(self, seat_id: int) -> Path:
        # Engine reads player_1.json -> seat 0, player_2.json -> seat 1, ...
        return self.io_dir / f"player_{seat_id + 1}.json"

    def _ready_file(self) -> Path:
        return self.io_dir / "ready.txt"

    def _status_file(self) -> Path:
        return self.io_dir / "game_status.json"

    def _atomic_write_json(self, path: Path, obj: Dict[str, Any]) -> None:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj), encoding="utf-8")
        os.replace(tmp, path)

    def _atomic_write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)

    def _write_ready(self) -> None:
        # Must be exactly 3 lines of "ready\n"
        path = self._ready_file()
        content = "ready\nready\nready\n"
        self._atomic_write_text(path, content)

    def _clear_io(self) -> None:
        # Clear ready + player files + status file (safe cleanup)
        for p in [self._ready_file(), self._status_file(),
                  self._player_file(0), self._player_file(1), self._player_file(2)]:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    def _read_status(self) -> Dict[str, Any]:
        # Single attempt: read status file without retrying internally.
        # Retries are handled by the caller (_wait_for_turn_change).
        status_path = self._status_file()
        if not status_path.exists():
            raise FileNotFoundError(f"Status file not found: {status_path}")
        try:
            txt = status_path.read_text(encoding="utf-8")
            return json.loads(txt)
        except Exception as e:
            raise RuntimeError(f"Failed to parse game_status.json: {e}")
    
    def _validate_status(self, status: Dict[str, Any]) -> None:
        try:
            if status["protocol_version"] != 2:
                raise ValueError("Engine protocol v2 required; rebuild the engine")
            if status["game"]["engine_seed"] != self.engine_seed:
                raise ValueError("Status belongs to another engine seed")
            turn = status["game"]["turn"]
            phase = status["game"]["phase"]
            if not (0 <= turn <= 21) or phase != ("finished" if turn == 21 else "playing"):
                raise ValueError("Inconsistent turn/phase")
            for sid in (0, 1, 2):
                int(status["players"][str(sid)]["points"]["total"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EngineError("protocol_error", str(exc)) from exc

    def _read_status_blocking(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        return self._wait_status(None, self.max_wait_seconds if timeout is None else timeout)

    def _wait_status(self, prev_turn: Optional[int], timeout: float) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Read before polling the process: a clean exit may have published final state.
            try:
                if self.timers:
                    with self.timers.time("transport.read_parse"):
                        status = self._read_status()
                else:
                    status = self._read_status()
            except (FileNotFoundError, RuntimeError) as exc:
                # Atomic publishing means an existing malformed file is a protocol failure.
                if isinstance(exc, RuntimeError):
                    raise EngineError("protocol_error", str(exc)) from exc
            else:
                self._validate_status(status)
                turn = int(status["game"]["turn"])
                if prev_turn is None:
                    if turn != 0:
                        raise EngineError("protocol_error", "Initial status is not turn zero")
                    return status
                if turn == prev_turn + 1:
                    return status
                if turn != prev_turn:
                    raise EngineError("protocol_error", "Unexpected turn sequence")
            if self.proc is not None and self.proc.poll() is not None:
                # The process can publish its final state between our read and poll.
                # Re-read after observing exit; never accept the stale pre-exit state.
                try:
                    final = self._read_status()
                    self._validate_status(final)
                    if (prev_turn is not None and final["game"]["turn"] == prev_turn + 1
                            and final["game"]["phase"] == "finished"):
                        return final
                except (FileNotFoundError, RuntimeError):
                    pass
                raise EngineError("engine_exit", f"Engine exited with code {self.proc.returncode}")
            if self.timers:
                with self.timers.time("transport.poll_sleep"):
                    time.sleep(self.poll_seconds)
            else:
                time.sleep(self.poll_seconds)
        raise EngineError("timeout", "Timed out waiting for engine status")

    def _wait_for_turn_change(self, prev_turn: int) -> Dict[str, Any]:
        return self._wait_status(prev_turn, self.max_wait_seconds)

    # ------------------------
    # Action mask + decoding
    # ------------------------
    def _compute_mask(self, status: Dict[str, Any], seat_id: int) -> np.ndarray:
        """Wrapper for obs_builder.compute_action_mask."""
        return compute_action_mask(status, seat_id)

    def _action_to_command(self, status: Dict[str, Any], seat_id: int, action_index: int) -> Dict[str, Any]:
        p = status["players"][str(seat_id)]
        hand: List[str] = p["cards_hand"]

        action_type, slot = decode_action(action_index)
        if slot >= len(hand):
            raise ValueError(f"Slot {slot} out of range for hand size {len(hand)}")

        card_name = hand[slot]
        subcommand = ACTION_TYPES[action_type]

        return {"turn": int(status["game"]["turn"]),
                "command": {"subcommand": subcommand, "argument": card_name, "extra": ""}}

    # ------------------------
    # Observation construction
    # ------------------------
    def _build_obs(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for obs_builder.build_obs."""
        if self.timers:
            with self.timers.time("features.agent_observation"):
                return build_obs(status, self.agent_seat)
        return build_obs(status, self.agent_seat)

    # --------------
    # Gym API
    # --------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # DEBUG: track reset frequency
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0
            self._step_since_reset = 0
        self._reset_count += 1
        self._step_since_reset = 0

        self.close()
        self._clear_io()

        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine executable not found: {self.engine_path}")

        # Log opponent configuration for debugging (optional, one per reset)
        opponent_info = [self.opponent_policies[i].name for i in range(3)]
        if seed == self._last_reset_seed:
            # Same seed, don't spam
            pass
        else:
            self._last_reset_seed = seed
            # Silently track; uncomment next line to debug opponent selection
            # print(f"[{self.io_dir.parent.name}] Seed {seed}: opponents={opponent_info}")

        # Launch engine with working directory set to io_dir parent
        # This ensures each env instance writes to its own io/ subdirectory
        # Engine uses relative path ./io/ so we ensure io_dir ends with /io
        engine_cwd = self.io_dir.parent if self.io_dir.name == "io" else self.repo_root
        
        self.engine_seed = int(seed) if seed is not None else int(self._engine_seed_rng.integers(0, 2**63))
        self._seat_rngs = [np.random.default_rng(derive_seed(self.engine_seed, "policy", sid))
                           for sid in (0, 1, 2)]
        self._stderr_file = (self.io_dir.parent / "engine_stderr.log").open("w")
        try:
            self.proc = subprocess.Popen(
                [str(self.engine_path), "--seed", str(self.engine_seed),
                 "--poll-us", str(self.engine_poll_us), "--quiet"],
                cwd=str(engine_cwd), stdout=subprocess.DEVNULL, stderr=self._stderr_file)
            status = self._read_status_blocking()
        except Exception:
            self.close()
            raise
        self._last_status = status
        
        # Sample opponents per episode (if enabled)
        if self._opponent_sampler is not None:
            left_name, right_name = self._opponent_sampler.sample_two(self._episode_idx)
            self._opp_names = (left_name, right_name)
            
            # Seed opponents deterministically (different seeds for left/right)
            left_seed = self._base_seed + 17 + 997 * self._episode_idx + 10007 * self._env_id
            right_seed = self._base_seed + 29 + 997 * self._episode_idx + 10007 * self._env_id
            
            self._opp_left = make_policy(left_name, seed=left_seed)
            self._opp_right = make_policy(right_name, seed=right_seed)
            
            # Reset opponents if they support it
            if hasattr(self._opp_left, "reset"):
                obs_temp = self._build_obs(status)
                self._opp_left.reset(observation=obs_temp)
            if hasattr(self._opp_right, "reset"):
                obs_temp = self._build_obs(status)
                self._opp_right.reset(observation=obs_temp)
        
        self._episode_idx += 1

        obs = self._build_obs(status)
        info = self._build_info(status, obs)
        
        # Add opponent info to info dict
        if self._opponent_sampler is not None:
            info["opponents"] = {"left": self._opp_names[0], "right": self._opp_names[1]}
            info["episode_idx"] = self._episode_idx - 1
            info["env_id"] = self._env_id
        
        # Always track opponent policy names
        info["opponent_policies"] = [p.name for p in self.opponent_policies]
        
        return obs, info

    def _build_info(self, status: Dict[str, Any], obs: Dict[str, Any]) -> Dict[str, Any]:
        # Helpful debugging payload
        agent = status["players"][str(self.agent_seat)]
        hand = agent["cards_hand"]
        mask = obs["action_mask"]

        decoded_legal = []
        for a in np.flatnonzero(mask == 1):
            at, slot = decode_action(int(a))
            decoded_legal.append(
                {
                    "action_index": int(a),
                    "type": ACTION_TYPES[at],
                    "slot": int(slot),
                    "card": hand[slot] if slot < len(hand) else None,
                }
            )

        return {
            "engine_seed": self.engine_seed,
            "transition_valid": True,
            "status": status,
            "turn": int(status["game"]["turn"]),
            "era": int(status["game"]["era"]),
            "clockwise": bool(status["game"]["clockwise"]),
            "agent_seat": self.agent_seat,
            "hand": hand,
            "cards_playable": agent["cards_playable"],
            "decoded_legal_actions": decoded_legal,
            "opponent_policies": [p.name for p in self.opponent_policies],
        }

    # ----------------------
    # Two-phase stepping (for parallel collection)
    # ----------------------
    def send_action(self, action: int, status: Optional[Dict[str, Any]] = None,
                    *, opponent_actions: Optional[Dict[int, int]] = None) -> int:
        """Phase 1: Send action to engine without waiting.
        
        Returns the prev_turn number (needed for wait_next).
        """
        from time import perf_counter_ns
        timers = self.timers
        if opponent_actions is None and self.batch_opponents:
            from sevenwonders_rl.policies.opponent_batch import opponent_actions_batch
            opponent_actions = opponent_actions_batch([self], [status], timers)[0]
        
        if timers:
            with timers.time("env.send_action_total"):
                return self._send_action_impl(action, status, timers, opponent_actions)
        else:
            return self._send_action_impl(action, status, None, opponent_actions)
    
    def _send_action_impl(self, action: int, status: Optional[Dict[str, Any]], timers: Optional[Any],
                          opponent_actions: Optional[Dict[int, int]] = None) -> int:
        from time import perf_counter_ns
        
        if self.proc is None:
            raise RuntimeError("Call reset() before send_action().")
        
        # Clear ready
        try:
            self._ready_file().unlink()
        except FileNotFoundError:
            pass
        
        if status is not None:
            current_status = status
        else:
            current_status = self._last_status
            if current_status is None:
                # Fallback: try to read (may fail if file not ready)
                try:
                    current_status = self._read_status()
                except Exception:
                    # Use previous cached status as fallback
                    current_status = self._last_status or {}
        
        prev_turn = int(current_status.get("game", {}).get("turn", 0))
        
        # Build masks
        if timers: t0 = perf_counter_ns()
        masks = [self._compute_mask(current_status, sid) for sid in (0, 1, 2)]
        if timers: timers.add_ns("features.masks", perf_counter_ns() - t0)
        for sid, selected in (opponent_actions or {}).items():
            if sid not in (0, 1, 2) or sid == self.agent_seat:
                raise ValueError(f"Invalid opponent override seat {sid}")
            if not 0 <= int(selected) < N_ACTIONS or masks[sid][int(selected)] == 0:
                raise ValueError(f"Illegal opponent action {selected} for seat {sid}")
        
        # Validate action
        if int(action) < 0 or int(action) >= N_ACTIONS or masks[self.agent_seat][int(action)] == 0:
            if self.invalid_action_fallback == "raise":
                raise ValueError(f"Illegal action {action}")
            legal = np.flatnonzero(masks[self.agent_seat] == 1)
            action = int(self._rng.choice(legal))
        
        # Choose actions for all seats
        if timers: t0 = perf_counter_ns()
        actions = [0, 0, 0]
        for sid in (0, 1, 2):
            if sid == self.agent_seat:
                actions[sid] = int(action)
            elif opponent_actions is not None and sid in opponent_actions:
                actions[sid] = int(opponent_actions[sid])
            else:
                # Use sampled opponents if available, otherwise fall back to opponent_policies
                if self._opponent_sampler is not None:
                    # Determine which sampled opponent to use based on seat positions
                    other_seats = [s for s in (0, 1, 2) if s != self.agent_seat]
                    if sid == other_seats[0]:
                        opp_policy = self._opp_left
                    else:
                        opp_policy = self._opp_right
                    
                    # Use policy's select_action method
                    obs_for_opp = build_obs(current_status, sid)
                    action_opp = opp_policy.select_action(obs_for_opp, self._seat_rngs[sid])
                    actions[sid] = int(action_opp)
                else:
                    actions[sid] = int(self.opponent_policies[sid](current_status, sid, masks[sid], self._seat_rngs[sid]))
        if timers: timers.add_ns("policies.opponents", perf_counter_ns() - t0)
        
        # Write player JSONs and ready (atomic, no waiting)
        if timers: t0 = perf_counter_ns()
        for sid in (0, 1, 2):
            cmd = self._action_to_command(current_status, sid, actions[sid])
            self._atomic_write_json(self._player_file(sid), cmd)
        if timers: timers.add_ns("env.write_players_json", perf_counter_ns() - t0)
        
        if timers: t0 = perf_counter_ns()
        self._write_ready()
        if timers: timers.add_ns("env.write_ready", perf_counter_ns() - t0)
        
        return prev_turn

    def wait_next(self, prev_turn: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Phase 2: Wait for engine to complete this turn.
        
        Returns (obs, reward, terminated, truncated, info).
        """
        from time import perf_counter_ns
        timers = self.timers
        
        if timers:
            with timers.time("env.wait_next_total"):
                return self._wait_next_impl(prev_turn, timers)
        else:
            return self._wait_next_impl(prev_turn, None)
    
    def _wait_next_impl(self, prev_turn: int, timers: Optional[Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        from time import perf_counter_ns
        
        try:
            status2 = self._wait_for_turn_change(prev_turn=prev_turn)
        except EngineError as exc:
            status2 = self._last_status or {}
            if not status2:
                raise EngineError("protocol_error", "No cached observation for failed transition") from exc
            if timers:
                with timers.time("env.build_obs"):
                    obs = self._build_obs(status2)
                    info = self._build_info(status2, obs)
            else:
                obs = self._build_obs(status2)
                info = self._build_info(status2, obs)
            info["truncated_reason"] = exc.reason
            info["transition_valid"] = False
            info["error"] = str(exc)
            info["engine_stderr"] = str(self.io_dir.parent / "engine_stderr.log")
            return obs, 0.0, False, True, info
        
        self._last_status = status2
        
        # Track steps for debug
        if hasattr(self, '_step_since_reset'):
            self._step_since_reset += 1
        
        terminated = status2["game"]["phase"] == "finished"
        result = status2.get("last_actions", {}).get(str(self.agent_seat), {})
        if result.get("status") == "fallback_discard":
            raise EngineError("action_rejected", "Engine replaced learner action with discard")
        
        if timers: t0 = perf_counter_ns()
        obs = self._build_obs(status2)
        info = self._build_info(status2, obs)
        if timers: timers.add_ns("env.build_obs", perf_counter_ns() - t0)
        
        # Reward: only on terminal
        reward = 0.0
        if terminated:
            totals = []
            for sid in (0, 1, 2):
                totals.append(int(status2["players"][str(sid)]["points"]["total"]))
            agent_total = totals[self.agent_seat]
            opp_best = max(t for i, t in enumerate(totals) if i != self.agent_seat)
            raw_reward = float(agent_total - opp_best)
            reward = raw_reward / self.reward_scale
            info["final_totals"] = totals
            info["raw_vp_diff"] = raw_reward
        
        return obs, reward, terminated, False, info

    def step(self, action: int):
        """Single-phase step: delegates to send_action + wait_next."""
        prev_turn = self.send_action(action)
        return self.wait_next(prev_turn)

    def close(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=1.0)
                except Exception:
                    pass
            self.proc = None
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None
