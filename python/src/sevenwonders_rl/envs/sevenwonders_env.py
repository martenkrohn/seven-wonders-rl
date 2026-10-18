# bots/sevenwonders_env.py
from __future__ import annotations

import json
import os
import random
import subprocess
import time
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
        io_dir: str = "io",
        agent_seat: int = 0,
        opponent_policies: Optional[List[SeatPolicy]] = None,
        max_wait_seconds: float = 5.0,
        invalid_action_fallback: str = "random_legal",  # or "raise"
        reward_scale: float = 1.0,
        seed: Optional[int] = None,
        env_id: int = 0,
        opponent_weights: Optional[Dict[str, float]] = None,
        timers: Optional[Any] = None,
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
        self.io_dir = (self.repo_root / io_dir).resolve()

        self.agent_seat = agent_seat
        self.max_wait_seconds = float(max_wait_seconds)
        self.invalid_action_fallback = invalid_action_fallback
        self.reward_scale = float(reward_scale)
        self.timers = timers

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
                "game_meta": spaces.Box(low=np.array([1, 0, 0]), high=np.array([3, 20, 1]), shape=(3,), dtype=np.int32),
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
    
    def _read_status_blocking(self, timeout: float = 5.0) -> Dict[str, Any]:
        # Read with retries (used at startup).
        status_path = self._status_file()
        deadline = time.time() + timeout
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            if not status_path.exists():
                time.sleep(0.01)
                continue
            try:
                txt = status_path.read_text(encoding="utf-8")
                return json.loads(txt)
            except Exception as e:
                last_err = e
                time.sleep(0.01)
        raise RuntimeError(f"Failed to read/parse game_status.json within timeout. Last error: {last_err}")

    def _wait_for_turn_change(self, prev_turn: int) -> Dict[str, Any]:
        from time import perf_counter_ns
        deadline = time.time() + self.max_wait_seconds
        debug_count = 0
        status_path = self._status_file()
        last_mtime = None
        timers = self.timers
        
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                # Engine exited; try one last read (might fail if file missing)
                try:
                    return self._read_status()
                except Exception:
                    return self._last_status or {}
            
            # Always attempt to read/parse (no mtime gate)
            try:
                if timers: t0 = perf_counter_ns()
                txt = status_path.read_text(encoding="utf-8")
                if timers: timers.add_ns("env.wait_read_status", perf_counter_ns() - t0)
                
                if timers: t0 = perf_counter_ns()
                status = json.loads(txt)
                if timers: timers.add_ns("env.wait_parse_status", perf_counter_ns() - t0)
                
                turn = int(status["game"]["turn"])
                if turn != prev_turn:
                    return status
            except (FileNotFoundError, json.JSONDecodeError, KeyError, RuntimeError) as e:
                debug_count += 1
                if debug_count == 5000:  # After ~5 seconds with 1ms polling
                    print(f"\n[HANG DEBUG] Env {self.io_dir}: Unable to parse turn change")
                    print(f"  Expected turn != {prev_turn}")
                    print(f"  Status file exists: {status_path.exists()}")
                    print(f"  Error: {type(e).__name__}")
            
            if timers:
                t0 = perf_counter_ns()
                time.sleep(0.001)
                timers.add_ns("env.wait_poll_sleep", perf_counter_ns() - t0)
            else:
                time.sleep(0.001)  # 1ms polling interval
        raise RuntimeError("Timeout waiting for engine to advance to next turn.")

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

        return {"command": {"subcommand": subcommand, "argument": card_name, "extra": ""}}

    # ------------------------
    # Observation construction
    # ------------------------
    def _build_obs(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for obs_builder.build_obs."""
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
        
        self.proc = subprocess.Popen(
            [str(self.engine_path)],
            cwd=str(engine_cwd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        status = self._read_status_blocking()
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
    def send_action(self, action: int, status: Optional[Dict[str, Any]] = None) -> int:
        """Phase 1: Send action to engine without waiting.
        
        Returns the prev_turn number (needed for wait_next).
        """
        from time import perf_counter_ns
        timers = self.timers
        
        if timers:
            with timers.time("env.send_action_total"):
                return self._send_action_impl(action, status, timers)
        else:
            return self._send_action_impl(action, status, None)
    
    def _send_action_impl(self, action: int, status: Optional[Dict[str, Any]], timers: Optional[Any]) -> int:
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
        masks = [self._compute_mask(current_status, sid) for sid in (0, 1, 2)]
        
        # Validate action
        if int(action) < 0 or int(action) >= N_ACTIONS or masks[self.agent_seat][int(action)] == 0:
            if self.invalid_action_fallback == "raise":
                raise ValueError(f"Illegal action {action}")
            legal = np.flatnonzero(masks[self.agent_seat] == 1)
            action = int(self._rng.choice(legal))
        
        # Choose actions for all seats
        actions = [0, 0, 0]
        for sid in (0, 1, 2):
            if sid == self.agent_seat:
                actions[sid] = int(action)
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
                    action_opp = opp_policy.select_action(obs_for_opp, self._rng)
                    actions[sid] = int(action_opp)
                else:
                    actions[sid] = int(self.opponent_policies[sid](current_status, sid, masks[sid], self._rng))
        
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
        except RuntimeError:
            status2 = self._last_status or {}
            if timers:
                with timers.time("env.build_obs"):
                    obs = self._build_obs(status2) if status2 else self.observation_space.sample()
                    info = self._build_info(status2, obs) if status2 else {}
            else:
                obs = self._build_obs(status2) if status2 else self.observation_space.sample()
                info = self._build_info(status2, obs) if status2 else {}
            info["truncated_reason"] = "timeout_or_engine_exit"
            return obs, 0.0, False, True, info
        
        self._last_status = status2
        
        # Track steps for debug
        if hasattr(self, '_step_since_reset'):
            self._step_since_reset += 1
        
        # Termination - only when game truly ends at turn 20
        turn2 = int(status2.get("game", {}).get("turn", prev_turn))
        terminated = (turn2 == 20)
        
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
                except Exception:
                    pass
            self.proc = None
