"""Evaluation with per-turn trace logging for strategy analysis.

This script extends eval_policy.py to capture detailed per-turn action traces,
enabling turn-by-turn decision analysis and strategy debugging.

Trace format (one JSON line per turn):
{
  "checkpoint_step": 256000,
  "seed": 42,
  "game_turn": 5,
  "era": 1,
  "action_idx": 14,
  "action_type": "build_structure",
  "card_name": "Lumber Yard",
  "card_color": "brown",
  "card_cost_coins": 0,
  "est_trade_cost_coins": 2,
  "delta_shields": 0,
  "delta_science_gear": 0,
  "delta_science_compass": 0,
  "delta_science_tablet": 0,
  "delta_vp_fixed": 0,
  "delta_gain_coins": 0,
  "delta_prod_raw_sum": 1,
  "delta_prod_manuf_sum": 0,
  "is_chain_free": 0,
  "enables_chain_to": 0,
  ...
}
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from sevenwonders_rl.envs import SevenWondersEnv, SeatPolicy
from sevenwonders_rl.policies import HeuristicPolicy
from sevenwonders_rl.policies.registry import make_policy
from sevenwonders_rl.features.action_feature_schema import FEATURE_NAMES, K as K_ACTION_FEATS, IDX
from sevenwonders_rl.utils.ppo_model import ActorCritic
from sevenwonders_rl.utils.ppo_utils import select_action_from_obs
from sevenwonders_rl.data.cards import by_id


ACTION_TYPES = ("build_structure", "build_wonder", "build_hand_free", "discard")


def _extract_card_color(card_name: str) -> str:
    """Extract card color from card name via DB lookup."""
    try:
        card = by_id.get(card_name)
        if card:
            color = card.get("color", "unknown")
            return color.lower()
    except Exception:
        pass
    return "unknown"


def _build_trace_record(
    checkpoint_step: int,
    seed: int,
    game_turn: int,
    era: int,
    action_idx: int,
    obs: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a trace record from action features.
    
    Extracts all relevant deltas, costs, and metadata from the chosen action.
    """
    action_feats = obs.get("action_feats")
    if action_feats is None or action_idx >= len(action_feats):
        return {}
    
    row = action_feats[action_idx]
    
    # Determine action type
    action_type_idx = -1
    for i in range(4):
        if row[IDX[f"action_type_{i}"]] > 0.5:
            action_type_idx = i
            break
    
    action_type = ACTION_TYPES[action_type_idx] if action_type_idx >= 0 else "unknown"
    
    # Extract card name if applicable
    card_name = ""
    card_color = ""
    if action_type in ("build_structure", "build_hand_free", "discard"):
        hand_ids = obs.get("hand_ids", [])
        slot = action_idx % 7
        if slot < len(hand_ids) and hand_ids[slot] >= 0:
            card_name = str(hand_ids[slot])  # Engine uses names, but we have IDs
            # Try to get actual name from status if available
            status = obs.get("_status")
            if status:
                my_hand = status.get("players", {}).get("0", {}).get("cards_hand", [])
                if slot < len(my_hand):
                    card_name = my_hand[slot]
                    card_color = _extract_card_color(card_name)
    
    # Build comprehensive trace record
    trace = {
        "checkpoint_step": checkpoint_step,
        "seed": seed,
        "game_turn": game_turn,
        "era": era,
        "action_idx": action_idx,
        "action_type": action_type,
        "card_name": card_name,
        "card_color": card_color,
        
        # Costs
        "card_cost_coins": int(row[IDX["card_cost_coins"]]) if "card_cost_coins" in IDX else 0,
        "est_trade_cost_coins": int(row[IDX["est_trade_cost_coins"]]) if "est_trade_cost_coins" in IDX else 0,
        "wonder_cost_coins": int(row[IDX["wonder_cost_coins"]]) if "wonder_cost_coins" in IDX else 0,
        "wonder_est_trade_cost_coins": int(row[IDX["wonder_est_trade_cost_coins"]]) if "wonder_est_trade_cost_coins" in IDX else 0,
        
        # VP deltas
        "delta_vp_fixed": int(row[IDX["delta_vp_fixed"]]) if "delta_vp_fixed" in IDX else 0,
        "delta_gain_coins": int(row[IDX["delta_gain_coins"]]) if "delta_gain_coins" in IDX else 0,
        
        # Military
        "delta_shields": int(row[IDX["delta_shields"]]) if "delta_shields" in IDX else 0,
        
        # Science
        "delta_science_gear": int(row[IDX["delta_science_gear"]]) if "delta_science_gear" in IDX else 0,
        "delta_science_compass": int(row[IDX["delta_science_compass"]]) if "delta_science_compass" in IDX else 0,
        "delta_science_tablet": int(row[IDX["delta_science_tablet"]]) if "delta_science_tablet" in IDX else 0,
        
        # Production (raw)
        "delta_prod_wood": int(row[IDX["delta_prod_wood"]]) if "delta_prod_wood" in IDX else 0,
        "delta_prod_ore": int(row[IDX["delta_prod_ore"]]) if "delta_prod_ore" in IDX else 0,
        "delta_prod_clay": int(row[IDX["delta_prod_clay"]]) if "delta_prod_clay" in IDX else 0,
        "delta_prod_stone": int(row[IDX["delta_prod_stone"]]) if "delta_prod_stone" in IDX else 0,
        
        # Production (manufactured)
        "delta_prod_loom": int(row[IDX["delta_prod_loom"]]) if "delta_prod_loom" in IDX else 0,
        "delta_prod_glass": int(row[IDX["delta_prod_glass"]]) if "delta_prod_glass" in IDX else 0,
        "delta_prod_papyrus": int(row[IDX["delta_prod_papyrus"]]) if "delta_prod_papyrus" in IDX else 0,
        
        # Production (choice)
        "delta_prod_choice_raw": int(row[IDX["delta_prod_choice_raw"]]) if "delta_prod_choice_raw" in IDX else 0,
        "delta_prod_choice_manuf": int(row[IDX["delta_prod_choice_manuf"]]) if "delta_prod_choice_manuf" in IDX else 0,
        
        # Derived summaries
        "delta_prod_raw_sum": sum(int(row[IDX[f"delta_prod_{r}"]]) for r in ["wood", "ore", "clay", "stone"] if f"delta_prod_{r}" in IDX),
        "delta_prod_manuf_sum": sum(int(row[IDX[f"delta_prod_{m}"]]) for m in ["loom", "glass", "papyrus"] if f"delta_prod_{m}" in IDX),
        
        # Chain info
        "is_chain_free_now": int(row[IDX["is_chain_free_now"]]) if "is_chain_free_now" in IDX else 0,
        "enables_chain_to_count": int(row[IDX["enables_chain_to_count"]]) if "enables_chain_to_count" in IDX else 0,
        
        # Wonder stage info (if applicable)
        "wonder_vp_gain": int(row[IDX["wonder_vp_gain"]]) if "wonder_vp_gain" in IDX else 0,
        "wonder_effect_play_last_card": int(row[IDX["wonder_effect_play_last_card"]]) if "wonder_effect_play_last_card" in IDX else 0,
        "wonder_effect_build_discard": int(row[IDX["wonder_effect_build_discard"]]) if "wonder_effect_build_discard" in IDX else 0,
    }
    
    return trace


def evaluate_with_traces(
    checkpoint_step: int,
    seeds: list[int],
    agent_policy_name: str,
    opponents: tuple[str, str],
    out_dir: Path,
    model: ActorCritic | None = None,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Run evaluation games and write per-turn traces to JSONL.
    
    Args:
        checkpoint_step: Training step (for trace metadata)
        seeds: List of seeds to evaluate
        agent_policy_name: Policy name for agent (or "model" if using model)
        opponents: Tuple of (left_opponent_name, right_opponent_name)
        out_dir: Output directory for traces
        model: Optional PPO model (if None, uses policy_name)
        device: Torch device for model inference
    """
    from sevenwonders_rl.policies.registry import make_policy
    
    repo_root = Path(__file__).resolve().parents[2]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create policy object if not using model
    agent_policy = None
    if model is None:
        agent_policy = make_policy(agent_policy_name, seed=0)
    
    left_name, right_name = opponents
    
    for seed in seeds:
        # Build opponent policies
        left_policy = make_policy(left_name, seed=seed + 10000)
        right_policy = make_policy(right_name, seed=seed + 20000)
        
        agent_seat = 0
        other_seats = [1, 2]
        opponent_policies = [SeatPolicy("random"), SeatPolicy("random"), SeatPolicy("random")]
        opponent_policies[other_seats[0]] = SeatPolicy(left_name, policy=left_policy)
        opponent_policies[other_seats[1]] = SeatPolicy(right_name, policy=right_policy)
        
        env = SevenWondersEnv(
            repo_root=str(repo_root),
            agent_seat=agent_seat,
            opponent_policies=opponent_policies,
            reward_scale=1.0,
        )
        
        obs, info = env.reset(seed=seed)
        done = False
        
        traces = []
        turn = 0
        
        while not done:
            status = info.get("status", {})
            era = int(status.get("game", {}).get("era", 1))
            game_turn = int(status.get("game", {}).get("turn", 0))
            
            # Select action
            if model is not None:
                action = select_action_from_obs(obs, model, device, deterministic=True)
            else:
                rng = np.random.default_rng(seed + turn)
                action = agent_policy.select_action(obs, rng)
            
            # Store status in obs for trace extraction
            obs["_status"] = status
            
            # Build trace record
            trace = _build_trace_record(
                checkpoint_step=checkpoint_step,
                seed=seed,
                game_turn=game_turn,
                era=era,
                action_idx=action,
                obs=obs,
            )
            
            if trace:
                traces.append(trace)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            turn += 1
        
        env.close()
        
        # Write traces for this game
        trace_filename = f"trace_step{checkpoint_step}_seed{seed}_{left_name}_vs_{right_name}.jsonl"
        trace_path = out_dir / trace_filename
        
        with trace_path.open("w", encoding="utf-8") as f:
            for trace in traces:
                f.write(json.dumps(trace) + "\n")
        
        print(f"  Wrote {len(traces)} turns to {trace_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate with trace logging")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46", help="Comma-separated seed list")
    parser.add_argument("--opponents", type=str, default="baseline,baseline", help="left,right opponent names")
    parser.add_argument("--out-dir", type=str, default="runs/traces", help="Output directory")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    
    # Load checkpoint
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    cfg = ckpt.get("config", {})
    global_dim = int(cfg.get("global_dim", 0))
    action_feat_dim = int(cfg.get("action_feat_dim", 0))
    hidden = int(cfg.get("hidden", 128))
    checkpoint_step = int(ckpt.get("timestep", 0))
    
    if global_dim <= 0 or action_feat_dim <= 0:
        raise ValueError("Checkpoint config missing global_dim or action_feat_dim")
    
    model = ActorCritic(global_dim, action_feat_dim, hidden=hidden).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Parse opponents
    opponent_parts = args.opponents.split(",")
    if len(opponent_parts) != 2:
        raise ValueError("--opponents must be 'left,right'")
    opponents = (opponent_parts[0].strip(), opponent_parts[1].strip())
    
    out_dir = Path(args.out_dir)
    
    print(f"Running trace evaluation:")
    print(f"  Checkpoint: {args.checkpoint} (step {checkpoint_step})")
    print(f"  Seeds: {seeds}")
    print(f"  Opponents: {opponents}")
    print(f"  Output: {out_dir}")
    print()
    
    evaluate_with_traces(
        checkpoint_step=checkpoint_step,
        seeds=seeds,
        agent_policy_name="model",
        opponents=opponents,
        out_dir=out_dir,
        model=model,
        device=device,
    )
    
    print("\\nTrace evaluation complete!")


if __name__ == "__main__":
    main()
