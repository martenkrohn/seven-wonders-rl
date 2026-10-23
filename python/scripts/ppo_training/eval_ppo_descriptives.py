"""Evaluate a trained PPO model with detailed action-level statistics.

This script loads a trained PPO checkpoint and runs evaluation games with full
per-action logging, producing the same detailed statistics as simulate_specialists_descriptives.py.

Supports:
- PPO vs bots: --opponents baseline:1.0 or --opponents baseline:0.5,military:0.5
- PPO vs PPO: --opponents runs/checkpoint_50k.pt:1.0
- Mixed evaluation: --opponents runs/ckpt_50k.pt:0.5,baseline:0.5

Usage:
    # Evaluate latest checkpoint against bots
    .venv/bin/python scripts/ppo_training/eval_ppo_descriptives.py runs/200k_vs_bots/ --games 50
    
    # Evaluate specific checkpoint against mixed opponents
    .venv/bin/python scripts/ppo_training/eval_ppo_descriptives.py runs/200k_vs_bots/checkpoint_100352.pt --games 50 --opponents baseline:0.5,military:0.5
    
    # PPO vs PPO (compare two checkpoints)
    .venv/bin/python scripts/ppo_training/eval_ppo_descriptives.py runs/200k_vs_bots/latest.pt --games 50 --opponents runs/200k_vs_bots/checkpoint_50176.pt:1.0
    
    # PPO vs mixed (PPO checkpoint + bots)
    .venv/bin/python scripts/ppo_training/eval_ppo_descriptives.py runs/200k_vs_bots/latest.pt --games 50 --opponents runs/200k_vs_bots/checkpoint_50176.pt:0.5,baseline:0.5
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sevenwonders_rl.envs import SevenWondersEnv, SeatPolicy
from sevenwonders_rl.policies import BaselinePolicy, MilitaryTempoPolicy, MadScientistPolicy
from sevenwonders_rl.policies.feature_policy_base import get_action_type
from sevenwonders_rl.policies.specialist_policies import _is_science_action, _is_military_action
from sevenwonders_rl.features.action_feature_schema import IDX
from sevenwonders_rl.utils.ppo_model import ActorCritic
from sevenwonders_rl.utils.ppo_utils import flatten_global
from sevenwonders_rl.utils.checkpoints import load_model as load_checkpoint_model, manifest, file_hash
from sevenwonders_rl.utils.runtime import atomic_json, run_io_root
from sevenwonders_rl.utils.seeding import derive_seed
from sevenwonders_rl.evaluation.runner import run_game, aggregate


# Card color mapping
CARD_COLORS = {
    "brown": "card_color_brown",
    "grey": "card_color_grey",
    "blue": "card_color_blue",
    "yellow": "card_color_yellow",
    "red": "card_color_red",
    "green": "card_color_green",
    "purple": "card_color_purple",
}


def get_card_color_from_feats(f: np.ndarray) -> str:
    """Extract card color from feature vector using one-hot encoding."""
    best = None
    best_val = 0.0
    for color, key in CARD_COLORS.items():
        idx = IDX.get(key, None)
        if idx is None:
            continue
        val = float(f[idx])
        if val > best_val:
            best_val = val
            best = color

    if best is None or best_val < 0.5:
        return "unknown"
    return best


# Exclusive bucket categories for analysis
BUCKETS = ("wonder", "discard", "brown", "grey", "blue", "yellow", "red", "green", "purple", "unknown")


@dataclass
class ActionEvent:
    """Per-action event log."""
    game_idx: int
    seat: int
    policy: str
    era: int
    turn: int
    action_idx: int
    action_type: int
    is_discard: bool
    is_wonder: bool
    is_science: bool
    is_military: bool
    prod_gain: float
    shields_gain: float
    science_gain: float
    invalid_returned: bool
    card_color: str
    execution: str = "requested"


@dataclass
class PPOPolicyLogger:
    """Wraps a PPO model and logs what it selects."""
    name: str
    model: ActorCritic
    device: torch.device
    events: List[ActionEvent] = field(default_factory=list)
    deterministic: bool = True

    def select_action(self, obs: Dict[str, Any], rng: Optional[np.random.Generator] = None) -> int:
        # Convert obs to tensors
        global_feat = torch.from_numpy(flatten_global(obs)).unsqueeze(0).to(self.device)
        action_feats = torch.from_numpy(np.asarray(obs["action_feats"], dtype=np.float32)).unsqueeze(0).to(self.device)
        action_mask = torch.from_numpy(np.asarray(obs["action_mask"], dtype=np.int8)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits, _ = self.model.forward_batch(global_feat, action_feats)
            
            # Apply mask (set invalid actions to -inf)
            action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))
            
            action_probs = torch.softmax(action_logits, dim=-1)
            if self.deterministic:
                a = int(action_logits.argmax(-1).item())
            else:
                if rng is None:
                    raise ValueError("Stochastic evaluation requires a local RNG")
                probs = action_probs.squeeze(0).cpu().numpy().astype(np.float64)
                a = int(rng.choice(len(probs), p=probs / probs.sum()))

        mask = np.asarray(obs["action_mask"], dtype=np.int8)
        invalid_returned = not (0 <= a < len(mask) and mask[a] == 1)

        feats = np.asarray(obs["action_feats"], dtype=np.float32)
        if 0 <= a < feats.shape[0]:
            f = feats[a]
        else:
            f = feats[0]

        at = int(get_action_type(f))

        # meta
        era = int(obs["game_meta"][0])
        turn = int(obs["game_meta"][1])

        # classify
        is_discard = (at == 3)
        is_wonder = (at == 1)
        is_science = _is_science_action(f, at)
        is_military = _is_military_action(f, at)

        # gains
        prod_keys = [
            "delta_prod_wood", "delta_prod_ore", "delta_prod_clay", "delta_prod_stone",
            "delta_prod_loom", "delta_prod_glass", "delta_prod_papyrus",
            "delta_prod_choice_raw", "delta_prod_choice_manuf",
        ]
        prod_gain = float(sum(float((f[IDX[k]] if k in IDX else 0.0)) for k in prod_keys))

        shields_gain = float((f[IDX["delta_shields"]] if "delta_shields" in IDX else 0.0)) + float((f[IDX["wonder_delta_shields"]] if "wonder_delta_shields" in IDX else 0.0))
        science_gain = float(
            (f[IDX["delta_science_gear"]] if "delta_science_gear" in IDX else 0.0) +
            (f[IDX["delta_science_compass"]] if "delta_science_compass" in IDX else 0.0) +
            (f[IDX["delta_science_tablet"]] if "delta_science_tablet" in IDX else 0.0) +
            (f[IDX["wonder_delta_science_gear"]] if "wonder_delta_science_gear" in IDX else 0.0) +
            (f[IDX["wonder_delta_science_compass"]] if "wonder_delta_science_compass" in IDX else 0.0) +
            (f[IDX["wonder_delta_science_tablet"]] if "wonder_delta_science_tablet" in IDX else 0.0) +
            (f[IDX["wonder_effect_add_science"]] if "wonder_effect_add_science" in IDX else 0.0)
        )

        card_color = get_card_color_from_feats(f)

        game_idx = getattr(self, "_game_idx", -1)
        seat = getattr(self, "_seat", -1)

        self.events.append(
            ActionEvent(
                game_idx=game_idx,
                seat=seat,
                policy=self.name,
                era=era,
                turn=turn,
                action_idx=int(a),
                action_type=at,
                is_discard=is_discard,
                is_wonder=is_wonder,
                is_science=is_science,
                is_military=is_military,
                prod_gain=prod_gain,
                shields_gain=shields_gain,
                science_gain=science_gain,
                invalid_returned=invalid_returned,
                card_color=card_color,
            )
        )

        return int(a)


@dataclass
class BotPolicyLogger:
    """Wraps a bot policy and logs what it selects."""
    name: str
    inner: Any
    events: List[ActionEvent] = field(default_factory=list)

    def select_action(self, obs: Dict[str, Any], rng: Optional[np.random.Generator] = None) -> int:
        if rng is None:
            rng = np.random.default_rng()

        a = int(self.inner.select_action(obs, rng))

        mask = np.asarray(obs["action_mask"], dtype=np.int8)
        invalid_returned = not (0 <= a < len(mask) and mask[a] == 1)

        feats = np.asarray(obs["action_feats"], dtype=np.float32)
        if 0 <= a < feats.shape[0]:
            f = feats[a]
        else:
            f = feats[0]

        at = int(get_action_type(f))

        era = int(obs["game_meta"][0])
        turn = int(obs["game_meta"][1])

        is_discard = (at == 3)
        is_wonder = (at == 1)
        is_science = _is_science_action(f, at)
        is_military = _is_military_action(f, at)

        prod_keys = [
            "delta_prod_wood", "delta_prod_ore", "delta_prod_clay", "delta_prod_stone",
            "delta_prod_loom", "delta_prod_glass", "delta_prod_papyrus",
            "delta_prod_choice_raw", "delta_prod_choice_manuf",
        ]
        prod_gain = float(sum(float((f[IDX[k]] if k in IDX else 0.0)) for k in prod_keys))

        shields_gain = float((f[IDX["delta_shields"]] if "delta_shields" in IDX else 0.0)) + float((f[IDX["wonder_delta_shields"]] if "wonder_delta_shields" in IDX else 0.0))
        science_gain = float(
            (f[IDX["delta_science_gear"]] if "delta_science_gear" in IDX else 0.0) +
            (f[IDX["delta_science_compass"]] if "delta_science_compass" in IDX else 0.0) +
            (f[IDX["delta_science_tablet"]] if "delta_science_tablet" in IDX else 0.0) +
            (f[IDX["wonder_delta_science_gear"]] if "wonder_delta_science_gear" in IDX else 0.0) +
            (f[IDX["wonder_delta_science_compass"]] if "wonder_delta_science_compass" in IDX else 0.0) +
            (f[IDX["wonder_delta_science_tablet"]] if "wonder_delta_science_tablet" in IDX else 0.0) +
            (f[IDX["wonder_effect_add_science"]] if "wonder_effect_add_science" in IDX else 0.0)
        )

        card_color = get_card_color_from_feats(f)

        game_idx = getattr(self, "_game_idx", -1)
        seat = getattr(self, "_seat", -1)

        self.events.append(
            ActionEvent(
                game_idx=game_idx,
                seat=seat,
                policy=self.name,
                era=era,
                turn=turn,
                action_idx=int(a),
                action_type=at,
                is_discard=is_discard,
                is_wonder=is_wonder,
                is_science=is_science,
                is_military=is_military,
                prod_gain=prod_gain,
                shields_gain=shields_gain,
                science_gain=science_gain,
                invalid_returned=invalid_returned,
                card_color=card_color,
            )
        )

        return int(a)


def load_model(checkpoint_path, device, allow_legacy=False):
    return load_checkpoint_model(checkpoint_path, device, allow_legacy=allow_legacy)


def repo_root() -> Path:
    """Get repository root (3 levels up from scripts/ppo_training/)."""
    return Path(__file__).resolve().parents[3]


def parse_opponent_mix(spec: str) -> Dict[str, float]:
    """Parse opponent specification.
    
    Supports:
    - Bot names: 'baseline:0.5,military:0.3,science:0.2'
    - Checkpoint paths: 'runs/checkpoint_50k.pt:0.5,baseline:0.5'
    - Mixed: 'runs/ckpt1.pt:0.3,runs/ckpt2.pt:0.3,baseline:0.4'
    """
    mix = {}
    for part in spec.split(","):
        name, weight = part.strip().split(":")
        mix[name.strip()] = float(weight.strip())
    
    # Normalize
    if not mix or any(not np.isfinite(v) or v < 0 for v in mix.values()) or sum(mix.values()) <= 0:
        raise ValueError("Opponent weights must be finite, nonnegative and have positive sum")
    total = sum(mix.values())
    if total > 0:
        mix = {k: v / total for k, v in mix.items()}
    
    return mix


def sample_opponents(opp_mix: Dict[str, float], rng: np.random.Generator) -> Tuple[str, str]:
    """Sample two opponents based on the mix."""
    names = list(opp_mix.keys())
    weights = list(opp_mix.values())
    
    chosen = rng.choice(names, size=2, replace=True, p=weights)
    return str(chosen[0]), str(chosen[1])


def is_checkpoint_path(name: str) -> bool:
    """Check if name is a checkpoint path (contains / or ends with .pt)."""
    return "/" in name or name.endswith(".pt")


def create_policy(name: str, seed: int, device: torch.device, allow_legacy=False) -> tuple[Any, str]:
    """Create a policy instance (bot or PPO checkpoint).
    
    Returns:
        (policy_instance, policy_label)
    """
    if is_checkpoint_path(name):
        # Load PPO checkpoint
        ckpt_path = Path(name)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        # Use checkpoint filename as label (without .pt extension)
        label = f"{ckpt_path.stem}_{file_hash(ckpt_path)[:12]}"
        
        model = load_model(ckpt_path, device, allow_legacy=allow_legacy)
        return model, f"ppo_{label}"
    else:
        # Create bot policy
        if name == "baseline":
            return BaselinePolicy(seed=seed), "baseline"
        elif name == "military":
            return MilitaryTempoPolicy(seed=seed), "military"
        elif name in ("science", "scientist", "mad_scientist"):
            return MadScientistPolicy(seed=seed), "scientist"
        else:
            raise ValueError(f"Unknown bot: {name}")



def run_evaluation(model, device, n_games, opp_mix, base_seed, out_dir=None,
                   io_root=None, action_mode="greedy", engine_poll_us=1000,
                   poll_seconds=.001, allow_legacy=False, checkpoint_id=None):
    rr = repo_root()
    output = Path(out_dir) if out_dir else run_io_root("runs/evaluations", "detail")
    output.mkdir(parents=True, exist_ok=True)
    exchange = run_io_root(io_root or rr / "io", "evaluation")
    for name in ("games.jsonl", "actions.jsonl"):
        (output / name).open("x").close()
    all_events, attempts = [], []
    ppo_logger = PPOPolicyLogger("ppo", model, device, deterministic=action_mode == "greedy")
    for game_idx in range(n_games):
        game_seed = derive_seed(base_seed, "evaluation", game_idx)
        rng = np.random.default_rng(derive_seed(base_seed, "opponents", game_idx))
        left_name, right_name = sample_opponents(opp_mix, rng)
        # Relative to seat 0, left is seat 2 and right is seat 1.
        policies, labels = {}, {}
        for seat, name in ((2, left_name), (1, right_name)):
            policy, label = create_policy(name, derive_seed(game_seed, "policy", seat), device, allow_legacy)
            labels[seat] = label + ("_left" if seat == 2 else "_right")
            policies[seat] = (PPOPolicyLogger(labels[seat], policy, device,
                             deterministic=action_mode == "greedy") if isinstance(policy, ActorCritic)
                             else BotPolicyLogger(labels[seat], policy))
        loggers = [ppo_logger, policies[1], policies[2]]
        for seat, logger in enumerate(loggers):
            logger.events.clear()
            logger._game_idx, logger._seat = game_idx, seat
        env = SevenWondersEnv(repo_root=str(rr), io_dir=str(exchange / f"game_{game_idx}" / "io"),
            agent_seat=0, opponent_policies=[SeatPolicy(l.name, policy=l) for l in loggers],
            invalid_action_fallback="raise", engine_poll_us=engine_poll_us, poll_seconds=poll_seconds)
        def on_step(obs, action, info):
            acknowledgements = info.get("status", {}).get("last_actions", {})
            for seat, logger in enumerate(loggers):
                if logger.events:
                    logger.events[-1].execution = acknowledgements.get(str(seat), {}).get("status", "unknown")
        agent_rng = np.random.default_rng(derive_seed(game_seed, "policy", 0))
        record, info = run_game(env, lambda obs: ppo_logger.select_action(obs, agent_rng),
            {"game_id": game_idx, "game_idx": game_idx, "engine_seed": game_seed, "seed": game_seed,
             "agent_seat": 0, "checkpoint_id": checkpoint_id,
             "policy_seeds": {str(s): derive_seed(game_seed, "policy", s) for s in (0, 1, 2)},
             "action_selection_mode": action_mode, "legacy_override": allow_legacy,
             "opponents": {"left": labels[2], "right": labels[1]},
             "opponents_by_seat": {str(s): labels[s] for s in (1, 2)},
             "unavailable_features": ["wonder_delta_shields", "wonder_delta_science_*", "wonder_effect_add_science"]}, on_step)
        if record["status"] == "completed":
            players = info["status"]["players"]
            record.update(coins=[int(players[str(s)]["resources"]["coins"]) for s in (0,1,2)],
                          vp_breakdown={s: players[str(s)]["points"] for s in (0,1,2)},
                          wonder_stages={s: players[str(s)]["wonder_stage"] for s in (0,1,2)})
        counts = Counter()
        for event in ppo_logger.events:
            if event.execution == "applied":
                counts["discard" if event.is_discard else "build_wonder" if event.is_wonder else
                       "prod_build" if event.prod_gain > 0 else "other"] += 1
        record["action_counts"] = dict(counts)
        attempts.append(record)
        with (output / "games.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        with (output / "actions.jsonl").open("a") as f:
            for logger in loggers:
                for event in logger.events:
                    f.write(json.dumps(asdict(event), allow_nan=False) + "\n")
                    all_events.append(event)
                logger.events.clear()
        shutil.rmtree(env.io_dir.parent)
    atomic_json(output / "summary.json", aggregate(attempts))
    shutil.rmtree(exchange)
    return all_events, attempts


def categorize(e: ActionEvent) -> str:
    """Exclusive category: wonder > discard > card_color."""
    if e.is_wonder:
        return "wonder"
    if e.is_discard:
        return "discard"
    return e.card_color


def mean_std(xs: List[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = float(np.mean(xs))
    s = float(np.std(xs, ddof=0)) if len(xs) > 1 else 0.0
    return m, s


def per_game_metrics(game_events: List[ActionEvent]) -> Dict[str, float]:
    n = len(game_events)
    out = {"n": float(n)}
    
    for b in BUCKETS:
        out[f"turns_{b}"] = 0.0
        out[f"share_{b}"] = 0.0
    
    if n == 0:
        out.update({
            "prod_gain_sum": 0.0,
            "shields_gain_sum": 0.0,
            "science_gain_sum": 0.0,
            "invalid_ct": 0.0,
        })
        return out
    
    cats = defaultdict(int)
    for e in game_events:
        cats[categorize(e)] += 1
    
    for b in BUCKETS:
        ct = float(cats[b])
        out[f"turns_{b}"] = ct
        out[f"share_{b}"] = ct / n
    
    out["prod_gain_sum"] = float(np.sum([e.prod_gain for e in game_events]))
    out["shields_gain_sum"] = float(np.sum([e.shields_gain for e in game_events]))
    out["science_gain_sum"] = float(np.sum([e.science_gain for e in game_events]))
    out["invalid_ct"] = float(sum(e.invalid_returned for e in game_events))
    
    return out


def summarize(events: List[ActionEvent], finals: List[Dict[str, Any]]) -> None:
    """Print comprehensive statistics."""
    print("Completion summary:", aggregate(finals))
    completed_ids = {f["game_idx"] for f in finals if f["status"] == "completed"}
    finals = [f for f in finals if f["status"] == "completed"]
    events = [e for e in events if e.game_idx in completed_ids and e.execution == "applied"]
    if not finals:
        print("No completed games; performance metrics unavailable.")
        return
    print("Gain statistics are feature estimates; unavailable wonder gains are excluded.")
    policies = ["ppo"] + sorted(set(p for f in finals for p in f["opponents_by_seat"].values()))
    ages = (1, 2, 3)
    
    # Index events
    ev_by_gp: Dict[tuple[int, str], List[ActionEvent]] = defaultdict(list)
    ev_by_gpa: Dict[tuple[int, str, int], List[ActionEvent]] = defaultdict(list)
    
    for e in events:
        if e.game_idx < 0:
            continue
        ev_by_gp[(e.game_idx, e.policy)].append(e)
        ev_by_gpa[(e.game_idx, e.policy, e.era)].append(e)
    
    # Build final lookup
    final_by_gp: Dict[tuple[int, str], Dict[str, Any]] = {}
    for fin in finals:
        gidx = int(fin["game_idx"])
        left_label, right_label = (fin["opponents_by_seat"][str(s)] for s in (1, 2))
        vp_breakdown = fin["vp_breakdown"]
        wonder_stages = fin["wonder_stages"]
        coins = fin["coins"]
        
        policy_map = {0: "ppo", 1: left_label, 2: right_label}
        for seat in (0, 1, 2):
            pol = policy_map[seat]
            final_by_gp[(gidx, pol)] = {
                "seat": seat,
                "total_vp": int(vp_breakdown[seat]["total"]),
                "wonder_vp": int(vp_breakdown[seat]["wonder"]),
                "wonder_stage": int(wonder_stages[seat]),
                "coins": int(coins[seat]),
            }
    
    # Per-policy game metrics
    print("\n" + "=" * 80)
    print("ACTION DESCRIPTIVES (per policy; mean ± std over games)")
    print("=" * 80)
    
    per_policy_game_metrics: Dict[str, Dict[str, List[float]]] = {
        pol: defaultdict(list) for pol in policies
    }
    
    for fin in finals:
        gidx = int(fin["game_idx"])
        left_label, right_label = (fin["opponents_by_seat"][str(s)] for s in (1, 2))
        policy_map = {0: "ppo", 1: left_label, 2: right_label}
        
        for seat, pol in policy_map.items():
            evs = ev_by_gp.get((gidx, pol), [])
            m = per_game_metrics(evs)
            f = final_by_gp.get((gidx, pol), {})
            
            per_policy_game_metrics[pol]["n_actions"].append(m["n"])
            for b in BUCKETS:
                per_policy_game_metrics[pol][f"turns_{b}"].append(m[f"turns_{b}"])
                per_policy_game_metrics[pol][f"share_{b}"].append(m[f"share_{b}"])
            
            per_policy_game_metrics[pol]["prod_gain_sum"].append(m["prod_gain_sum"])
            per_policy_game_metrics[pol]["shields_gain_sum"].append(m["shields_gain_sum"])
            per_policy_game_metrics[pol]["science_gain_sum"].append(m["science_gain_sum"])
            per_policy_game_metrics[pol]["invalid_ct"].append(m["invalid_ct"])
            
            per_policy_game_metrics[pol]["total_vp"].append(float(f.get("total_vp", 0)))
            per_policy_game_metrics[pol]["wonder_vp"].append(float(f.get("wonder_vp", 0)))
            per_policy_game_metrics[pol]["wonder_stage"].append(float(f.get("wonder_stage", 0)))
    
    for pol in policies:
        d = per_policy_game_metrics[pol]
        n_games = len(d["total_vp"])
        print(f"\n[{pol.upper()}] (n={n_games} games)")
        
        m, s = mean_std(d["n_actions"])
        print(f"  actions/game:   {m:6.2f} +/- {s:6.2f}")
        
        for b in BUCKETS:
            m, s = mean_std(d[f"turns_{b}"])
            print(f"  {b:8s} turns: {m:6.2f} +/- {s:6.2f}")
        
        m, s = mean_std(d["prod_gain_sum"])
        print(f"  prod_gain sum:    {m:7.1f} +/- {s:7.1f}")
        m, s = mean_std(d["shields_gain_sum"])
        print(f"  shields_gain sum: {m:7.1f} +/- {s:7.1f}")
        m, s = mean_std(d["science_gain_sum"])
        print(f"  science_gain sum: {m:7.1f} +/- {s:7.1f}")
        m, s = mean_std(d["invalid_ct"])
        print(f"  invalid returned: {m:7.2f} +/- {s:7.2f}")
        
        m, s = mean_std(d["total_vp"])
        print(f"  total VP:       {m:7.2f} +/- {s:7.2f}")
        m, s = mean_std(d["wonder_vp"])
        print(f"  wonder VP:      {m:7.2f} +/- {s:7.2f}")
        m, s = mean_std(d["wonder_stage"])
        print(f"  wonder stages:  {m:7.2f} +/- {s:7.2f}")
    
    # Per-age build mix
    print("\n" + "=" * 80)
    print("PER-AGE BUILD MIX (per policy; mean ± std over games)")
    print("=" * 80)
    
    per_policy_age: Dict[tuple[str, int], Dict[str, List[float]]] = {
        (pol, age): defaultdict(list) for pol in policies for age in ages
    }
    
    for fin in finals:
        gidx = int(fin["game_idx"])
        left_label, right_label = (fin["opponents_by_seat"][str(s)] for s in (1, 2))
        policy_map = {0: "ppo", 1: left_label, 2: right_label}
        
        for seat, pol in policy_map.items():
            for age in ages:
                evs = ev_by_gpa.get((gidx, pol, age), [])
                n = len(evs)
                if n == 0:
                    for b in BUCKETS:
                        per_policy_age[(pol, age)][b].append(0.0)
                        per_policy_age[(pol, age)]["count_" + b].append(0.0)
                    per_policy_age[(pol, age)]["n_actions"].append(0.0)
                    continue
                
                cats = defaultdict(int)
                for e in evs:
                    cats[categorize(e)] += 1
                
                for b in BUCKETS:
                    per_policy_age[(pol, age)][b].append(cats[b] / n)
                    per_policy_age[(pol, age)]["count_" + b].append(float(cats[b]))
                per_policy_age[(pol, age)]["n_actions"].append(float(n))
    
    for pol in policies:
        print(f"\n{'='*30} POLICY: {pol.upper():<10} {'='*30}")
        
        for age in ages:
            d = per_policy_age[(pol, age)]
            mN, sN = mean_std(d["n_actions"])
            
            print(f"\n[ Age {age} ] (Avg Total Actions: {mN:.2f} ± {sN:.2f})")
            print("-" * 40)
            
            display_lines = []
            for b in BUCKETS:
                m, s = mean_std(d[b])
                c, c_std = mean_std(d["count_" + b])
                pct = m * 100
                
                display_lines.append(f"{b.capitalize():<12}: {c:>5.2f} ± {c_std:<5.2f} ({pct:>5.1f}%)")
            
            for i in range(0, len(display_lines), 2):
                left = display_lines[i]
                right = display_lines[i+1] if i+1 < len(display_lines) else ""
                print(f"  {left}    {right}")
    
    # Win statistics
    print("\n" + "=" * 80)
    print("WIN STATISTICS")
    print("=" * 80)
    
    wins = defaultdict(int)
    for f in finals:
        first_label, second_label = (f["opponents_by_seat"][str(s)] for s in (1, 2))
        policy_map = {0: "ppo", 1: first_label, 2: second_label}
        if len(f["winner_seats"]) == 1:
            wins[policy_map[f["winner_seats"][0]]] += 1
    
    total_games = len(finals)
    for pol in policies:
        appearances = sum(pol == "ppo" or pol in f["opponents_by_seat"].values() for f in finals)
        print(f"  {pol}: wins={wins[pol]}  appearances={appearances}  vp_win_rate={wins[pol]/max(1,appearances):.3f}")
    
    print("\n" + "=" * 80)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate trained PPO model with detailed action statistics."
    )
    ap.add_argument("checkpoint", type=str,
                    help="Path to checkpoint file or directory (will use latest.pt)")
    ap.add_argument("--games", type=int, default=50,
                    help="Number of evaluation games (default: 50)")
    ap.add_argument("--opponents", type=str, default="baseline:1.0",
                    help="Opponent mix specification (default: 'baseline:1.0')")
    ap.add_argument("--seed", type=int, default=2000,
                    help="Base random seed (default: 2000)")
    ap.add_argument("--device", type=str, default="cpu",
                    help="Device to run model on (default: cpu)")
    
    ap.add_argument("--out-dir")
    ap.add_argument("--io-root")
    ap.add_argument("--action-mode", choices=["greedy", "stochastic"], default="greedy")
    ap.add_argument("--engine-poll-us", type=int, default=1000)
    ap.add_argument("--poll-seconds", type=float, default=.001)
    ap.add_argument("--allow-legacy", action="store_true")
    args = ap.parse_args()
    if args.games <= 0 or args.seed < 0:
        ap.error("games must be positive and seed nonnegative")
    
    # Resolve checkpoint path
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "latest.pt"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Parse opponent mix
    opp_mix = parse_opponent_mix(args.opponents)
    
    device = torch.device(args.device)
    
    print(f"\nLoading model from: {ckpt_path}")
    model = load_model(ckpt_path, device, args.allow_legacy)
    
    print(f"Running {args.games} evaluation games...")
    print(f"Opponent mix: {opp_mix}")
    print(f"Seed: {args.seed}")
    
    output = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_io_root("runs/evaluations", "detail")
    if args.out_dir and output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "manifest.json", manifest(repo_root(), {**vars(args), "checkpoint_sha256": file_hash(ckpt_path)}))
    events, finals = run_evaluation(
        model=model,
        device=device,
        n_games=args.games,
        opp_mix=opp_mix,
        base_seed=args.seed, out_dir=output, io_root=args.io_root,
        action_mode=args.action_mode, engine_poll_us=args.engine_poll_us,
        poll_seconds=args.poll_seconds, allow_legacy=args.allow_legacy,
        checkpoint_id=file_hash(ckpt_path),
    )
    
    print(f"\nCompleted {len(finals)} games, {len(events)} total actions logged.\n")
    
    summarize(events, finals)


if __name__ == "__main__":
    main()
