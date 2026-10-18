#!/usr/bin/env python3
"""Comprehensive analysis of the action space visible to bots.

This script:
1. Runs a game and captures observations at multiple turns
2. Analyzes the action feature matrix structure
3. Provides examples of what different action types look like
4. Shows what information is available for decision-making
"""

from pathlib import Path
import sys
import numpy as np
import json

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))

from sevenwonders_rl.envs import SevenWondersEnv
from sevenwonders_rl.features.action_feature_schema import (
    FEATURE_NAMES, IDX, K, ACTION_TYPES, CARD_COLORS, RESOURCE_ORDER,
    SCIENCE_SYMBOLS
)


def format_action_type(action_idx: int) -> str:
    """Decode action index to human readable format."""
    action_type = action_idx // 7
    slot = action_idx % 7
    return f"{ACTION_TYPES[action_type]}[slot_{slot}]"


def analyze_observation(obs: dict, info: dict, turn_num: int) -> dict:
    """Analyze a single observation and extract key insights."""
    
    status = info["status"]
    agent_player = status["players"][str(info["agent_seat"])]
    
    mask = obs["action_mask"]
    action_feats = obs["action_feats"]
    
    legal_actions = np.flatnonzero(mask == 1)
    
    analysis = {
        "turn": turn_num,
        "era": status["game"]["era"],
        "hand_size": len([x for x in obs["hand_ids"] if x >= 0]),
        "cards_in_hand": agent_player["cards_hand"],
        "coins": agent_player["resources"]["coins"],
        "wonder_stage": agent_player["wonder_stage"],
        "wonder_name": agent_player["wonder_name"],
        "num_legal_actions": len(legal_actions),
        "legal_action_types": {},
        "example_actions": []
    }
    
    # Group legal actions by type
    for action_idx in legal_actions:
        action_type_idx = action_idx // 7
        action_type_name = ACTION_TYPES[action_type_idx]
        if action_type_name not in analysis["legal_action_types"]:
            analysis["legal_action_types"][action_type_name] = 0
        analysis["legal_action_types"][action_type_name] += 1
    
    # Collect detailed examples of each action type
    seen_types = set()
    for action_idx in legal_actions:
        action_type_idx = action_idx // 7
        action_type_name = ACTION_TYPES[action_type_idx]
        
        # Get first example of each type
        if action_type_name not in seen_types:
            seen_types.add(action_type_name)
            feats = action_feats[action_idx]
            
            slot = action_idx % 7
            card_name = agent_player["cards_hand"][slot] if slot < len(agent_player["cards_hand"]) else None
            
            example = {
                "action_index": int(action_idx),
                "action_type": action_type_name,
                "slot": slot,
                "card_name": card_name,
                "is_legal": bool(feats[IDX["is_legal"]] > 0.5),
                "features": {}
            }
            
            # Extract key features based on action type
            if action_type_name == "discard":
                example["features"]["coins_gained"] = float(feats[IDX["discard_delta_coins"]])
                example["features"]["coins_after"] = float(feats[IDX["coins_after_card_cost"]])
                
            elif action_type_name in ["build_structure", "build_hand_free"]:
                # Card identification
                card_id_norm = feats[IDX["card_id_norm"]]
                example["features"]["card_id_norm"] = float(card_id_norm)
                
                # Determine card color
                for color in CARD_COLORS:
                    if feats[IDX[f"card_color_{color}"]] > 0.5:
                        example["features"]["card_color"] = color
                        break
                
                # Costs
                example["features"]["cost_coins"] = float(feats[IDX["card_cost_coins"]])
                example["features"]["cost_raw_sum"] = float(feats[IDX["card_cost_raw_sum"]])
                example["features"]["cost_manuf_sum"] = float(feats[IDX["card_cost_manuf_sum"]])
                example["features"]["coins_after_cost"] = float(feats[IDX["coins_after_card_cost"]])
                
                # Trade estimation
                example["features"]["est_trade_cost"] = float(feats[IDX["est_trade_cost_coins"]])
                example["features"]["needs_trade"] = bool(feats[IDX["needs_trade"]] > 0.5)
                example["features"]["cannot_afford"] = bool(feats[IDX["cannot_afford_with_coins"]] > 0.5)
                
                # Immediate effects
                example["features"]["delta_shields"] = float(feats[IDX["delta_shields"]])
                example["features"]["delta_vp"] = float(feats[IDX["delta_vp_fixed"]])
                example["features"]["delta_coins"] = float(feats[IDX["delta_gain_coins"]])
                
                # Production deltas
                prod_deltas = {}
                for res in RESOURCE_ORDER:
                    delta = float(feats[IDX[f"delta_prod_{res}"]])
                    if delta > 0:
                        prod_deltas[res] = delta
                if prod_deltas:
                    example["features"]["production_gains"] = prod_deltas
                
                # Chain features
                example["features"]["is_chain_free"] = bool(feats[IDX["is_chain_free_now"]] > 0.5)
                example["features"]["enables_chains_to"] = int(feats[IDX["enables_chain_to_count"]])
                
            elif action_type_name == "build_wonder":
                example["features"]["wonder_stage_index"] = float(feats[IDX["wonder_stage_index_norm"]])
                example["features"]["wonder_cost_raw"] = float(feats[IDX["wonder_cost_raw_sum"]])
                example["features"]["wonder_cost_manuf"] = float(feats[IDX["wonder_cost_manuf_sum"]])
                example["features"]["wonder_vp_gain"] = float(feats[IDX["wonder_vp_gain"]])
                example["features"]["wonder_delta_shields"] = float(feats[IDX["wonder_delta_shields"]])
                example["features"]["wonder_est_trade_cost"] = float(feats[IDX["wonder_est_trade_cost_coins"]])
            
            analysis["example_actions"].append(example)
    
    return analysis


def main():
    """Run analysis and generate report."""
    print("=" * 80)
    print("ACTION SPACE ANALYSIS FOR 7 WONDERS BOTS")
    print("=" * 80)
    print()
    
    # Print schema overview
    print("FEATURE SCHEMA OVERVIEW")
    print("-" * 80)
    print(f"Total features per action: {K}")
    print(f"Action space size: 28 (4 action types × 7 hand slots)")
    print(f"Action types: {', '.join(ACTION_TYPES)}")
    print()
    
    print("Feature Categories:")
    feature_groups = {
        "Universal Header": ["is_legal", "slot_is_empty", "action_type_*", "card_id_norm", "card_color_*", "card_era_*"],
        "Build Costs": ["card_cost_coins", "card_cost_*_sum", "coins_after_card_cost"],
        "Trade Estimation": ["missing_*_sum", "est_trade_cost_*", "needs_trade", "cannot_afford_with_coins"],
        "Card Effects": ["delta_shields", "delta_science_*", "delta_gain_coins", "delta_vp_fixed", "delta_prod_*"],
        "Strategic Flags": ["adds_discount_*", "adds_extra_*", "is_chain_free_now", "enables_chain_to_count"],
        "Wonder Features": ["wonder_stage_index_norm", "wonder_cost_*", "wonder_vp_gain", "wonder_effect_*"],
        "Discard": ["discard_delta_coins"],
    }
    
    for group, patterns in feature_groups.items():
        print(f"  • {group}: {', '.join(patterns)}")
    print()
    
    # Run a sample game and capture observations
    print("RUNNING SAMPLE GAME...")
    print("-" * 80)
    
    env = SevenWondersEnv(
        repo_root=str(REPO_ROOT),
        agent_seat=0,
        seed=42
    )
    
    obs, info = env.reset(seed=999)
    
    # Capture observations at different turns
    observations = []
    turn_num = 0
    
    # Analyze initial state
    analysis = analyze_observation(obs, info, turn_num)
    observations.append(analysis)
    
    # Run for a few turns
    done = False
    max_turns = 5
    while not done and turn_num < max_turns:
        mask = obs["action_mask"]
        legal = np.flatnonzero(mask == 1)
        
        if len(legal) == 0:
            break
        
        # Pick a random legal action
        action = env._rng.choice(legal)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        turn_num += 1
        
        if not done:
            analysis = analyze_observation(obs, info, turn_num)
            observations.append(analysis)
    
    env.close()
    
    print("✓ Game completed\n")
    
    # Generate detailed report
    print("DETAILED TURN-BY-TURN ANALYSIS")
    print("=" * 80)
    print()
    
    for obs_data in observations:
        print(f"Turn {obs_data['turn']} (Era {obs_data['era']})")
        print("-" * 80)
        print(f"Wonder: {obs_data['wonder_name']} (Stage {obs_data['wonder_stage']})")
        print(f"Hand size: {obs_data['hand_size']}")
        print(f"Cards in hand: {', '.join(obs_data['cards_in_hand'])}")
        print(f"Coins: {obs_data['coins']}")
        print(f"Legal actions: {obs_data['num_legal_actions']}")
        print()
        
        print("Legal action types:")
        for action_type, count in obs_data['legal_action_types'].items():
            print(f"  • {action_type}: {count}")
        print()
        
        print("Example actions:")
        for example in obs_data['example_actions']:
            print(f"  [{example['action_index']:2d}] {example['action_type']:20s} slot={example['slot']}")
            if example['card_name']:
                print(f"      Card: {example['card_name']}")
            
            for feat_name, feat_val in example['features'].items():
                if isinstance(feat_val, dict):
                    print(f"      {feat_name}: {feat_val}")
                elif isinstance(feat_val, bool):
                    if feat_val:
                        print(f"      {feat_name}: True")
                elif isinstance(feat_val, float):
                    if abs(feat_val) > 0.001:
                        print(f"      {feat_name}: {feat_val:.2f}")
                else:
                    print(f"      {feat_name}: {feat_val}")
        print()
        print()
    
    # Generate summary statistics
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()
    
    action_type_totals = {}
    for obs_data in observations:
        for action_type, count in obs_data['legal_action_types'].items():
            if action_type not in action_type_totals:
                action_type_totals[action_type] = []
            action_type_totals[action_type].append(count)
    
    print("Average legal actions per turn by type:")
    for action_type, counts in action_type_totals.items():
        avg = np.mean(counts)
        print(f"  • {action_type}: {avg:.1f} (min={min(counts)}, max={max(counts)})")
    print()
    
    # Feature name reference
    print("COMPLETE FEATURE NAME REFERENCE")
    print("=" * 80)
    print()
    print("All 80 features available per action:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  [{i:2d}] {name}")
    print()
    
    print("=" * 80)
    print("Analysis complete!")
    print()
    print("Key insights for policy design:")
    print("  1. Each turn you see 28 actions (4 types × 7 slots)")
    print("  2. Action mask filters to only legal actions (typically 7-21)")
    print("  3. Each action has 80 features describing costs, effects, and strategy")
    print("  4. Features include: costs, resource deltas, VP gains, trade estimates")
    print("  5. Chain-building and wonder progression tracked explicitly")
    print("  6. Different action types have different relevant features")
    print()


if __name__ == "__main__":
    main()
