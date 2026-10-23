"""Observation building logic for 7 Wonders environment.

This module separates feature extraction from the environment step logic.
All observations are built with player rotation: (self, left, right) relative to agent.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal
import numpy as np
from sevenwonders_rl.features.action_features import build_action_feats

N_SLOTS = 7
N_ACTIONS = 28


def _seat_order(agent_seat: int) -> List[int]:
    """Return player seats in order: [self, left, right].
    
    In 7 Wonders, left neighbor is (agent_seat - 1) mod 3,
    right neighbor is (agent_seat + 1) mod 3.
    """
    return [agent_seat, (agent_seat - 1) % 3, (agent_seat + 1) % 3]


def compute_action_mask(status: Dict[str, Any], seat_id: int) -> np.ndarray:
    """Compute action mask for a given seat.
    
    Action encoding: action_index = action_type * 7 + slot
    - action_type 0: build_structure
    - action_type 1: build_wonder
    - action_type 2: build_hand_free
    - action_type 3: discard
    
    Args:
        status: game_status.json dict
        seat_id: player seat (0, 1, or 2)
        
    Returns:
        Binary mask of shape (28,) where 1 = legal action
    """
    if status.get("game", {}).get("phase") == "finished":
        return np.zeros((N_ACTIONS,), dtype=np.int8)
    p = status["players"][str(seat_id)]
    hand: List[str] = p["cards_hand"]
    playable = set(p["cards_playable"])
    can_wonder = bool(p["can_build_wonder"])
    can_free = bool(p["can_build_hand_free"])

    mask = np.zeros((N_ACTIONS,), dtype=np.int8)

    # action_index = action_type*7 + slot
    # 0 build_structure, 1 build_wonder, 2 build_hand_free, 3 discard
    for slot in range(min(N_SLOTS, len(hand))):
        card = hand[slot]

        # Discard always legal for real slots
        mask[3 * 7 + slot] = 1
        
        # Build structure only if playable
        if card in playable:
            mask[0 * 7 + slot] = 1
            
        # Build wonder if can_build_wonder flag set
        if can_wonder:
            mask[1 * 7 + slot] = 1
            
        # Build hand free if can_build_hand_free flag set
        if can_free:
            mask[2 * 7 + slot] = 1

    return mask


def build_table_features(status: Dict[str, Any], agent_seat: int) -> Dict[str, np.ndarray]:
    """Build table-wide features with player rotation (self, left, right).
    
    Returns dict with:
        - table_basic: [coins, shields, gear, compass, tablet] for each player (3, 5)
        - table_prod: [wood, ore, clay, stone, loom, glass, papyrus] for each player (3, 7),
          including the wonder's starting resource in the engine resource totals
        - wonder_stage: current wonder stage for each player (3,)
        - table_points: [civilian, commercial, guild, military, scientific, wonder, total] (3, 7)
        - table_amount: [civilian, commercial, guild, military, scientific, raw_material, manufactured_goods] (3, 7)
    """
    order = _seat_order(agent_seat)

    # Basic table: [coins, shields, gear, compass, tablet]
    basic = np.zeros((3, 5), dtype=np.int32)

    # Production by resource: [wood, ore, clay, stone, loom, glass, papyrus]
    prod = np.zeros((3, 7), dtype=np.int32)

    # Wonder stages
    stages = np.zeros((3,), dtype=np.int32)

    # Points breakdown: [civilian, commercial, guild, military, scientific, wonder, total]
    points = np.zeros((3, 7), dtype=np.int32)

    # Amount breakdown: [civilian, commercial, guild, military, scientific, raw_material, manufactured_goods]
    amount = np.zeros((3, 7), dtype=np.int32)

    for i, sid in enumerate(order):
        p = status["players"][str(sid)]
        r = p["resources"]
        pt = p["points"]
        am = p["amount"]

        basic[i, :] = np.array(
            [r["coins"], r["shields"], r["gear"], r["compass"], r["tablet"]],
            dtype=np.int32
        )

        prod[i, :] = np.array(
            [r["wood"], r["ore"], r["clay"], r["stone"], r["loom"], r["glass"], r["papyrus"]],
            dtype=np.int32
        )

        stages[i] = int(p["wonder_stage"])

        points[i, :] = np.array(
            [pt["civilian"], pt["commercial"], pt["guild"], pt["military"], pt["scientific"], pt["wonder"], pt["total"]],
            dtype=np.int32
        )

        amount[i, :] = np.array(
            [am["civilian"], am["commercial"], am["guild"], am["military"], am["scientific"], am["raw_material"], am["manufactured_goods"]],
            dtype=np.int32
        )

    return {
        "table_basic": basic,
        "table_prod": prod,
        "wonder_stage": stages,
        "table_points": points,
        "table_amount": amount,
    }


def build_hand_ids(status: Dict[str, Any], agent_seat: int) -> np.ndarray:
    """Extract agent's hand card IDs, padded to N_SLOTS with -1.
    
    Returns:
        Array of shape (7,) with card IDs, -1 for empty slots
    """
    hand_ids = list(status["players"][str(agent_seat)]["cards_hand_id"])
    padded = hand_ids[:N_SLOTS] + [-1] * max(0, N_SLOTS - len(hand_ids))
    return np.array(padded, dtype=np.int32)


def build_obs(
    status: Dict[str, Any], 
    agent_seat: int,
    feature_version: Literal["v2"] = "v2"
) -> Dict[str, Any]:
    """Build full observation dict for the agent.
    
    This is the main entry point for observation construction.
    
    Args:
        status: game_status.json dict
        agent_seat: player seat (0, 1, or 2)
        feature_version: "v2" for per-resource features (92 dims). Legacy "v1" no longer supported.
    
    Returns observation dict with keys:
        - global_table: (3, 7) int32 - backward compatible summary
        - wonder_stage: (3,) int32 - wonder stages rotated
        - hand_ids: (7,) int32 - agent's hand card IDs
        - action_mask: (28,) int8 - legal actions for agent
        - action_feats: (28, 92) float32 - per-action features with per-resource breakdown
        - table_points: (3, 7) int32 - points breakdown per player
        - table_prod: (3, 7) int32 - resource production per player
        - table_amount: (3, 7) int32 - card counts per player
        - game_meta: (3,) int32 - [era, turn, clockwise]
    """
    feat = build_table_features(status, agent_seat)
    mask = compute_action_mask(status, agent_seat)
    hand_ids = build_hand_ids(status, agent_seat)
    
    # Use v2 action features (92 dimensions per action)
    action_feats = build_action_feats(status, agent_seat, mask)

    # Backward compatible global_table: [coins, shields, gear, compass, tablet, raw_total, manu_total]
    basic = feat["table_basic"]  # (3,5)
    prod = feat["table_prod"]    # (3,7)
    
    # Raw materials: wood, ore, clay, stone (columns 0-3)
    raw_total = prod[:, 0:4].sum(axis=1)
    # Manufactured goods: loom, glass, papyrus (columns 4-6)
    manu_total = prod[:, 4:7].sum(axis=1)
    
    global_table = np.concatenate(
        [basic, raw_total[:, None].astype(np.int32), manu_total[:, None].astype(np.int32)],
        axis=1
    )

    obs = {
        # Backward compatible keys (keep for existing tests)
        "global_table": global_table.astype(np.int32),
        "wonder_stage": feat["wonder_stage"],
        "hand_ids": hand_ids,
        "action_mask": mask,
        
        # Per-action features (Step 4)
        "action_feats": action_feats,

        # New richer summaries
        "table_points": feat["table_points"],
        "table_prod": feat["table_prod"],
        "table_amount": feat["table_amount"],

        # Game metadata
        "game_meta": np.array(
            [status["game"]["era"], status["game"]["turn"], int(bool(status["game"]["clockwise"]))],
            dtype=np.int32
        ),
    }
    return obs
