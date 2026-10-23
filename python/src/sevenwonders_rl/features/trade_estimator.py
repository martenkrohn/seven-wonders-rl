"""Trade cost estimation for per-action features.

This module provides a fast approximation of trade costs when resources are missing.
It handles both raw and manufactured goods, accounts for discount flags, and
treats neighbor choice production as exclusive tokens using min-cost max-flow.
"""

from typing import Dict, Tuple, NamedTuple, List
from dataclasses import dataclass
from heapq import heappush, heappop
import numpy as np
from sevenwonders_rl.data.mechanics import (
    get_trade_costs, 
    get_manufactured_discount_flag, 
    get_raw_discount_flags,
    resource_kind
)

RESOURCE_ORDER = ['wood', 'ore', 'clay', 'stone', 'loom', 'glass', 'papyrus']


@dataclass
class _Edge:
    to: int
    rev: int
    cap: int
    cost: int


def _add_edge(g: List[List[_Edge]], fr: int, to: int, cap: int, cost: int) -> None:
    """Add directed edge with reverse edge for flow network."""
    g[fr].append(_Edge(to=to, rev=len(g[to]), cap=cap, cost=cost))
    g[to].append(_Edge(to=fr, rev=len(g[fr]) - 1, cap=0, cost=-cost))


def _min_cost_max_flow(g: List[List[_Edge]], s: int, t: int, maxf: int) -> Tuple[int, int]:
    """
    Successive shortest augmenting path algorithm with potentials.
    Returns (flow_sent, min_cost).
    """
    n = len(g)
    INF = 10**18
    h = [0] * n  # potentials
    prevv = [0] * n
    preve = [0] * n
    flow = 0
    cost = 0

    while maxf > 0:
        dist = [INF] * n
        dist[s] = 0
        pq: List[Tuple[int, int]] = [(0, s)]
        while pq:
            d, v = heappop(pq)
            if d != dist[v]:
                continue
            for i, e in enumerate(g[v]):
                if e.cap <= 0:
                    continue
                nd = d + e.cost + h[v] - h[e.to]
                if nd < dist[e.to]:
                    dist[e.to] = nd
                    prevv[e.to] = v
                    preve[e.to] = i
                    heappush(pq, (nd, e.to))

        if dist[t] == INF:
            break  # cannot send more flow

        for v in range(n):
            if dist[v] < INF:
                h[v] += dist[v]

        d = maxf
        v = t
        while v != s:
            e = g[prevv[v]][preve[v]]
            d = min(d, e.cap)
            v = prevv[v]

        maxf -= d
        flow += d
        cost += d * h[t]

        v = t
        while v != s:
            e = g[prevv[v]][preve[v]]
            e.cap -= d
            g[v][e.rev].cap += d
            v = prevv[v]

    return flow, cost



class TradeEstimate(NamedTuple):
    """Result of trade estimation."""
    total_cost: int  # Total coins to cover missing resources
    cost_left: int   # Cost from left neighbor
    cost_right: int  # Cost from right neighbor
    missing_raw: int  # Missing raw resources after own production
    missing_manuf: int  # Missing manufactured after own production
    plan_left: Dict[str, int]  # Resources bought from left
    plan_right: Dict[str, int]  # Resources bought from right
    feasible: bool  # True if trade can cover all missing


def _resource_to_idx(resource: str) -> int:
    """Map resource name to index."""
    return RESOURCE_ORDER.index(resource)


def _cost_dict_to_vec(cost_dict: Dict[str, int]) -> np.ndarray:
    """Convert cost dict to 7-element vector."""
    vec = np.zeros(7, dtype=np.int32)
    for res in RESOURCE_ORDER:
        if res in cost_dict:
            vec[_resource_to_idx(res)] = cost_dict[res]
    return vec


def _vec_to_cost_dict(vec: np.ndarray) -> Dict[str, int]:
    """Convert 7-element vector back to cost dict."""
    result = {}
    for i, res in enumerate(RESOURCE_ORDER):
        if vec[i] > 0:
            result[res] = int(vec[i])
    return result


def _get_neighbor_discount(resource: str, direction: str, player_flags: set[str]) -> int:
    """Get discounted trade cost for a resource from a neighbor.
    
    Args:
        resource: Resource name
        direction: 'left' or 'right'; observation order is engine east/west
        player_flags: Set of discount flags the player has
    
    Returns:
        Coin cost (1 if discounted, 2 if not)
    """
    trade_costs = get_trade_costs()
    base_cost = trade_costs['raw'] if resource_kind(resource) == 'raw' else trade_costs['manuf']
    discount_cost = trade_costs.get('raw_discount', 1) if resource_kind(resource) == 'raw' else trade_costs.get('manuf_discount', 1)
    
    # Check for applicable discount flags
    if resource_kind(resource) == 'raw':
        if direction == 'left' and 'raw_cheap_east' in player_flags:
            return discount_cost
        if direction == 'right' and 'raw_cheap_west' in player_flags:
            return discount_cost
    else:  # manuf
        if 'manuf_cheap' in player_flags:
            return discount_cost
    
    return base_cost


def estimate_trade_cost(
    required: Dict[str, int],
    own_production: Dict[str, int],
    own_choice_resources: list[set[str]],  # List of sets of resources (each card's choices)
    left_supply: Dict[str, int],
    left_choice_supply: list[set[str]],
    right_supply: Dict[str, int],
    right_choice_supply: list[set[str]],
    player_flags: set[str],
) -> TradeEstimate:
    """Estimate minimum trade cost for missing resources.
    
    This uses a two-stage approach:
    1. Use own fixed production to cover cost
    2. Allocate own choice resources
    3. Buy remaining from neighbors (cheapest first, then as needed)
    
    Args:
        required: Dict of required resources {resource: count}
        own_production: Dict of own fixed production
        own_choice_resources: List of choice sets (each set contains 2-3 resource options)
        left_supply: Left neighbor fixed production
        left_choice_supply: Left neighbor choice sets
        right_supply: Right neighbor fixed production
        right_choice_supply: Right neighbor choice sets
        player_flags: Set of player discount flags
    
    Returns:
        TradeEstimate with cost breakdown
    """
    # Convert to vectors for easier manipulation
    required_vec = _cost_dict_to_vec(required)
    own_prod_vec = _cost_dict_to_vec(own_production)
    left_supply_vec = _cost_dict_to_vec(left_supply)
    right_supply_vec = _cost_dict_to_vec(right_supply)
    
    # Stage 1: Use own fixed production
    remaining_vec = np.maximum(required_vec - own_prod_vec, 0).astype(np.int32)
    
    # Stage 2: Allocate own choice production greedily
    # Prioritize filling manufactured goods first if needed, else highest remaining need
    for choice_set in own_choice_resources:
        if remaining_vec.sum() == 0:
            break
        # Find the resource in this choice set with highest remaining need
        best_res = None
        best_remaining = -1
        # Resolve equal needs in schema order, independent of PYTHONHASHSEED.
        for res in RESOURCE_ORDER:
            if res not in choice_set:
                continue
            res_idx = _resource_to_idx(res)
            if remaining_vec[res_idx] > best_remaining:
                best_remaining = remaining_vec[res_idx]
                best_res = res
        
        if best_res and best_remaining > 0:
            res_idx = _resource_to_idx(best_res)
            remaining_vec[res_idx] -= 1
    
    # Stage 3: Buy remaining from neighbors using min-cost flow allocation
    # This respects choice-token exclusivity: each token can only be used once.
    plan_left: Dict[str, int] = {}
    plan_right: Dict[str, int] = {}

    need_total = int(remaining_vec.sum())
    if need_total == 0:
        return TradeEstimate(
            total_cost=0,
            cost_left=0,
            cost_right=0,
            missing_raw=0,
            missing_manuf=0,
            plan_left={},
            plan_right={},
            feasible=True,
        )

    # Build "items" (each usable once):
    # - each fixed neighbor resource unit is an item with a single option
    # - each neighbor choice token is an item with multiple options
    items: List[Tuple[str, List[str]]] = []  # (direction, options)
    
    # Direction ordering matters for deterministic tie-breaking; add left first.
    for res_idx, cnt in enumerate(left_supply_vec):
        res = RESOURCE_ORDER[res_idx]
        for _ in range(int(cnt)):
            items.append(("left", [res]))
    for choice_set in left_choice_supply:
        opts = [r for r in RESOURCE_ORDER if r in choice_set]
        if opts:
            items.append(("left", opts))

    for res_idx, cnt in enumerate(right_supply_vec):
        res = RESOURCE_ORDER[res_idx]
        for _ in range(int(cnt)):
            items.append(("right", [res]))
    for choice_set in right_choice_supply:
        opts = [r for r in RESOURCE_ORDER if r in choice_set]
        if opts:
            items.append(("right", opts))

    # Graph nodes:
    # 0 = source
    # 1..M = item nodes
    # M+1..M+7 = resource nodes (one per resource)
    # M+8 = sink
    M = len(items)
    R = len(RESOURCE_ORDER)
    SRC = 0
    ITEM0 = 1
    RES0 = ITEM0 + M
    SNK = RES0 + R
    g: List[List[_Edge]] = [[] for _ in range(SNK + 1)]

    # source -> items (each item has capacity 1)
    for i in range(M):
        _add_edge(g, SRC, ITEM0 + i, cap=1, cost=0)

    # items -> resources (each item can provide one of its options)
    # Keep handles so we can decode which edges were used
    handles: List[Tuple[int, int, str, str]] = []  # (item_node, edge_idx, direction, resource)
    for i, (direction, options) in enumerate(items):
        inode = ITEM0 + i
        for res in options:
            rnode = RES0 + _resource_to_idx(res)
            unit_cost = _get_neighbor_discount(res, direction, player_flags)
            edge_idx = len(g[inode])
            _add_edge(g, inode, rnode, cap=1, cost=int(unit_cost))
            handles.append((inode, edge_idx, direction, res))

    # resources -> sink (capacity = remaining need for each resource)
    for res_idx, need in enumerate(remaining_vec):
        if int(need) > 0:
            _add_edge(g, RES0 + res_idx, SNK, cap=int(need), cost=0)

    sent, min_cost = _min_cost_max_flow(g, SRC, SNK, need_total)

    # Decode which item->resource edges were used
    # (edge cap goes from 1 to 0 when used)
    for inode, edge_idx, direction, res in handles:
        if edge_idx < len(g[inode]) and g[inode][edge_idx].cap == 0:
            if direction == "left":
                plan_left[res] = plan_left.get(res, 0) + 1
            else:
                plan_right[res] = plan_right.get(res, 0) + 1

    # Compute costs split by direction
    cost_left = sum(v * _get_neighbor_discount(r, "left", player_flags) for r, v in plan_left.items())
    cost_right = sum(v * _get_neighbor_discount(r, "right", player_flags) for r, v in plan_right.items())
    total_cost = int(cost_left + cost_right)

    # Update remaining_vec to reflect what we bought
    bought_vec = np.zeros_like(remaining_vec)
    for res, v in plan_left.items():
        bought_vec[_resource_to_idx(res)] += int(v)
    for res, v in plan_right.items():
        bought_vec[_resource_to_idx(res)] += int(v)
    remaining_vec = (remaining_vec - bought_vec).clip(min=0).astype(np.int32)

    # Count missing raw/manuf AFTER trade
    missing_raw = int(sum(remaining_vec[_resource_to_idx(res)] for res in RESOURCE_ORDER if resource_kind(res) == 'raw'))
    missing_manuf = int(sum(remaining_vec[_resource_to_idx(res)] for res in RESOURCE_ORDER if resource_kind(res) == 'manuf'))

    return TradeEstimate(
        total_cost=int(total_cost),
        cost_left=int(cost_left),
        cost_right=int(cost_right),
        missing_raw=missing_raw,
        missing_manuf=missing_manuf,
        plan_left=plan_left,
        plan_right=plan_right,
        feasible=(sent == need_total),
    )


__all__ = [
    'TradeEstimate',
    'estimate_trade_cost',
    'RESOURCE_ORDER',
]
