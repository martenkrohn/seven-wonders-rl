"""Seat-local, detached visible memory for decision-v1 (never a raw state log).

Logical nested records use dotted keys in the flat Gym/PPO tensor dictionary.
Only the allowlisted public fields below and the observing seat's own offer/hand
are read. Transport identities are used solely for acknowledgement and deduping.
"""

from functools import lru_cache
import hashlib
import json

import numpy as np

from .decision_features import (
    COLORS,
    COUNTS,
    FEATURE_NAMES,
    FREE,
    K,
    KINDS,
    N_OPTIONS,
    RESOURCES,
    SCIENCE,
    candidate_features,
    dataset,
    public_table,
)

C, W, S, H, J = 80, 14, 4, 32, 64
ABILITIES = (
    "first_color_free",
    "first_age_card_free",
    "last_age_card_free",
    "play_seventh_card",
    "discard_build_for_free",
)
STAGE_EFFECT_NAMES = (
    "vp",
    "coins",
    "shields",
    "science_joker",
    *("choice_" + r for r in RESOURCES[:7]),
    "choice_tradable",
    *ABILITIES,
)
TIME_NAMES = (
    "age",
    "round",
    "turn",
    "pass_left",
    "primary_regular",
    "primary_seventh",
    "primary_none",
    "own_regular",
    "own_seventh",
    "own_discard",
    "own_none",
    "public_index",
    "followup_seq",
    "own_index",
)
PHASE_NAMES = ("initial", "primary", "discard", "age_end", "terminal")
PUBLIC_SHAPES = {
    "public_table": (3, 24),
    "built_counts": (3, C),
    "copied_guild_counts": (3, C),
    "wonder_side": (3, W),
    "stage_exists": (3, S),
    "stage_built": (3, S),
    "stage_costs": (3, S, 8),
    "stage_effects": (3, S, 17),
    "ability_unlocked": (3, 5),
    "ability_available": (3, 5),
    "ability_available_known": (3, 5),
    "choice_sources": (3, C + S, 9),
    "choice_source_mask": (3, C + S),
}
PUBLIC_MASKS = {
    "stage_exists",
    "stage_built",
    "ability_available_known",
    "choice_source_mask",
}
CURRENT_SHAPES = {
    "hand_counts": (C,),
    "hand_visible": (),
    "discard_offer_counts": (C,),
    "discard_offer_visible": (),
    "time": (14,),
    "discard_count": (),
}
CURRENT_MASKS = {"hand_visible", "discard_offer_visible"}


@lru_cache(maxsize=1)
def vocabulary():
    data = dataset()
    cards, wonders = tuple(sorted(data["cards"])), tuple(sorted(data["wonders"]))
    if len(cards) != C or len(wonders) != W:
        raise ValueError(
            "Memory vocabulary changed; a new observation contract is required"
        )
    allowed = {
        "production",
        "choice",
        "science",
        "coins",
        "shields",
        "vp",
        "vp_per",
        "coins_per",
        "count",
        "target",
        "condition",
        "tradable",
        "includes_self",
    } | {"discount"}
    for card in data["cards"].values():
        effect = card["effects"]
        if set(effect) - allowed:
            raise ValueError("Unknown card effect; update the observation contract")
        domains = {
            "science": (*SCIENCE, "joker"),
            "discount": ("raw_left", "raw_right", "manufactured_both"),
            "count": COUNTS,
            "target": ("self", "neighbors", "self_and_neighbors"),
            "condition": ("wonder_complete",),
        }
        if any(
            k in effect and effect[k] not in values for k, values in domains.items()
        ):
            raise ValueError(
                "Unknown card effect value; update the observation contract"
            )
        if (
            set(card["cost"]) - set(RESOURCES)
            or set(effect.get("production", {})) - set(RESOURCES[:7])
            or set(effect.get("choice", [])) - set(RESOURCES[:7])
        ):
            raise ValueError("Unknown card resource")
    # Validate all future stages as well, before accepting a run.
    for wonder in data["wonders"].values():
        if len(wonder["stages"]) > S:
            raise ValueError("Wonder exceeds memory stage capacity")
        for stage in wonder["stages"]:
            stage_vector(stage)
    return cards, wonders


def card_index(card):
    key = (card["type_id"], card["age"])
    try:
        return vocabulary()[0].index(key)
    except ValueError as exc:
        raise ValueError(f"Unknown visible card key: {key}") from exc


def counts(cards):
    result = np.zeros(C, np.float32)
    for card in cards:
        result[card_index(card)] += 1
    return result


def stage_vector(stage):
    result = np.zeros(17, np.float32)
    result[0] = stage["vp"]
    for effect in stage["effects"]:
        kind = effect["type"]
        if kind in ("add_coins", "add_shields"):
            result[1 if kind == "add_coins" else 2] += effect["amount"]
        elif kind == "add_science":
            if effect.get("amount", 1) != 1:
                raise ValueError("Unsupported science quantity")
            result[3] += 1
        elif kind in ("choose_raw_nontradable", "choose_manuf_nontradable"):
            if effect.get("amount", 1) != 1 or result[4:11].any():
                raise ValueError("Unsupported wonder choice quantity/combination")
            result[4:8] = kind == "choose_raw_nontradable"
            result[8:11] = kind == "choose_manuf_nontradable"
        elif kind in ABILITIES:
            result[12 + ABILITIES.index(kind)] = 1
        else:
            raise ValueError(f"Unknown wonder effect: {kind}")
    if set(stage["cost"]) - set(RESOURCES):
        raise ValueError("Unknown stage cost resource")
    return result


def public_snapshot(state, seat):
    """No hands, decisions, events, grants or internal allocations are read here."""
    data = dataset()
    _, wonder_keys = vocabulary()
    p = {
        key: np.zeros(shape, bool if key in PUBLIC_MASKS else np.float32)
        for key, shape in PUBLIC_SHAPES.items()
    }
    p["public_table"] = public_table(state, seat)
    terminal = state["game"]["phase"] == "finished"
    primary_open = state["game"]["phase"] == "collect_actions"
    round_ = state["game"]["round_in_age"] or 0
    for role, s in enumerate((seat, (seat + 2) % 3, (seat + 1) % 3)):
        player = state["players"][str(s)]
        p["built_counts"][role] = counts(player["built_cards"])
        p["copied_guild_counts"][role] = counts(
            [dict(type_id=c["type_id"], age=3) for c in player.get("copied_guilds", [])]
        )
        try:
            wi = wonder_keys.index(player["wonder_id"])
        except ValueError as exc:
            raise ValueError("Unknown visible wonder side") from exc
        p["wonder_side"][role, wi] = 1
        stages = data["wonders"][player["wonder_id"]]["stages"]
        built = player["wonder_stage"]
        if not 0 <= built <= len(stages):
            raise ValueError("Invalid public wonder stage")
        for n, stage in enumerate(stages):
            p["stage_exists"][role, n] = True
            p["stage_built"][role, n] = n < built
            p["stage_costs"][role, n] = [stage["cost"].get(r, 0) for r in RESOURCES]
            effects = stage_vector(stage)
            p["stage_effects"][role, n] = effects
            if n < built:
                p["ability_unlocked"][role] = np.maximum(
                    p["ability_unlocked"][role], effects[12:]
                )
                if effects[4:11].any():
                    p["choice_sources"][role, C + n] = [*effects[4:11], 1, effects[11]]
                    p["choice_source_mask"][role, C + n] = True
        for ci, amount in enumerate(p["built_counts"][role]):
            if not amount:
                continue
            effect = data["cards"][vocabulary()[0][ci]]["effects"]
            if effect.get("choice"):
                if set(effect["choice"]) - set(RESOURCES[:7]):
                    raise ValueError("Unknown choice resource")
                p["choice_sources"][role, ci] = [
                    *(float(r in effect["choice"]) for r in RESOURCES[:7]),
                    amount,
                    float(effect.get("tradable", False)),
                ]
                p["choice_source_mask"][role, ci] = True
        unlocked = p["ability_unlocked"][role].astype(bool)
        # These availabilities follow solely from public rules and table state.
        p["ability_available_known"][role, :4] = True
        colors = {
            data["cards"][(c["type_id"], c["age"])]["color"]
            for c in player["built_cards"]
        }
        p["ability_available"][role, :4] = [
            unlocked[0] and len(colors) < len(COLORS) and primary_open,
            unlocked[1] and round_ == 1 and primary_open,
            unlocked[2] and round_ == 6 and primary_open,
            unlocked[3] and round_ == 7 and primary_open,
        ]
        # Do not infer pending grants from hidden transport bookkeeping.
        # Only publicly declared active rights are known; absence is unknown
        # for a seat which has acquired a discard-build ability.
        active = any(
            a["code"] == "discard_build_for_free" and a["availability"] == "decision"
            for a in player.get("available_abilities", [])
        )
        p["ability_available"][role, 4] = active and not terminal
        p["ability_available_known"][role, 4] = active or not unlocked[4] or terminal
    return p


def field_specs():
    """Stable flat wire layout: field -> (shape, dtype)."""
    result = {}
    for key, shape in PUBLIC_SHAPES.items():
        dtype = bool if key in PUBLIC_MASKS else np.float32
        result["public." + key] = (shape, dtype)
        result["decision_history.public." + key] = ((H, *shape), dtype)
        result["public_history.public." + key] = ((J, *shape), dtype)
    for key, shape in CURRENT_SHAPES.items():
        dtype = bool if key in CURRENT_MASKS else np.float32
        result[key] = (shape, dtype)
        result["decision_history." + key] = ((H, *shape), dtype)
    result.update(
        {
            "decision_history.applied_action_feats": ((H, K), np.float32),
            "decision_history.applied_card": ((H, C), np.float32),
            "decision_history.applied_mask": ((H,), bool),
            "decision_history.valid": ((H,), bool),
            "public_history.source_turn": ((J,), np.float32),
            "public_history.followup_seq": ((J,), np.float32),
            "public_history.phase_flags": ((J, 5), np.float32),
            "public_history.discard_count": ((J,), np.float32),
            "public_history.valid": ((J,), bool),
            "action_feats": ((N_OPTIONS, K), np.float32),
            "action_mask": ((N_OPTIONS,), bool),
        }
    )
    return result


def memory_contract(ruleset):
    cards, wonders = vocabulary()
    contract = {
        "format": "decision-ppo-v2",
        "observation": "decision-observation-memory-v1",
        "memory": "visible-decision-history-v1",
        "protocol_version": 3,
        "ruleset": ruleset,
        "source_hash": dataset()["source_hash"],
        "dataset_hash": dataset()["dataset_hash"],
        "card_keys": [list(k) for k in cards],
        "wonder_keys": list(wonders),
        "seats": ["self", "left", "right"],
        "resources": list(RESOURCES),
        "public_field_order": list(PUBLIC_SHAPES),
        "hand_field_order": [
            "hand_counts",
            "discard_offer_counts",
            "hand_visible",
            "discard_offer_visible",
            "time",
            "discard_count",
        ],
        "abilities": list(ABILITIES),
        "stage_effect_names": list(STAGE_EFFECT_NAMES),
        "time_names": list(TIME_NAMES),
        "phase_names": list(PHASE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "action_kinds": list(KINDS),
        "public_feature_names": [
            *("resource_" + r for r in RESOURCES[:7]),
            "coins",
            "shields",
            *("science_" + s for s in SCIENCE),
            "jokers",
            "wonder_stage",
            "wonder_id",
            *("color_" + c for c in COLORS),
            "points_total",
            "wonder_complete",
        ],
        "free_modes": list(FREE),
        "n_options": N_OPTIONS,
        "decision_capacity": H,
        "public_capacity": J,
        "fields": {
            k: {"shape": list(shape), "dtype": np.dtype(dtype).name}
            for k, (shape, dtype) in field_specs().items()
        },
        "scaling": {
            "counts": "raw",
            "time": "raw",
            "stage_costs_effects": "raw",
            "public_table_divisors": [
                *([4] * 7),
                20,
                6,
                4,
                4,
                4,
                2,
                4,
                13,
                *([6] * 7),
                100,
                1,
            ],
            "action_feature_divisors": action_divisors(),
        },
        "masks": {
            "padding": "zero-and-valid-false",
            "unknown": "zero-and-known-false",
            "hand": "own-offered-decision-only",
            "discard": "own-discard-offer-only",
        },
        "visibility": "own-hand-own-offer-public-table-confirmed-own-actions-v1",
        "time_unit": "simultaneous-primary-resolution-and-observed-discard-followups-v1",
        "history": "predecision-snapshot-plus-applied-action; public-deltas-at-every-wait",
        "overflow": "error",
        "flattening": "dotted-logical-record-paths",
    }
    canonical = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    contract["schema_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    return contract


def action_divisors():
    result = []
    for name in FEATURE_NAMES:
        divisor = 1
        if name.startswith("card_cost_"):
            divisor = 4
        if any(
            name.startswith(p)
            for p in ("card_production_", "effect_production_", "effect_count_")
        ):
            divisor = 2
        if name in ("card_vp", "effect_vp"):
            divisor = 8
        if name in ("card_coins", "effect_coins"):
            divisor = 9
        if name in (
            "card_shields",
            "effect_shields",
            "effect_vp_per",
            "effect_coins_per",
            "card_age",
        ):
            divisor = 3
        if name == "card_type_id":
            divisor = 78
        if name.startswith("payment_"):
            divisor = 10
        if name == "science_gain":
            divisor = 20
        if name == "next_stage":
            divisor = 4
        result.append(divisor)
    return result


class DecisionMemory:
    def __init__(self, seat):
        if seat not in range(3):
            raise ValueError("Invalid memory seat")
        self.seat = seat
        self.game_id = None
        self.public = []
        self.decisions = []
        self.pending = {}
        self.last_turn = None
        self.followup = 0

    def observe(self, state, before=None):
        """Ingest one acknowledged status, before any following policy is called."""
        if self.game_id is None:
            self.game_id = state["game_id"]
        elif self.game_id != state["game_id"]:
            raise ValueError("Memory cannot cross games; reset is required")
        current = public_snapshot(state, self.seat)
        discard = state["discard_count"]
        terminal = state["game"]["phase"] == "finished"
        turn = state["game"]["turn"]
        initial = not self.public
        changed = initial or any(
            not np.array_equal(v, self.public[-1]["public"][k])
            for k, v in current.items()
        )
        changed = (
            changed
            or discard != self.public[-1]["discard_count"]
            or turn != self.last_turn
        )
        if changed:
            if len(self.public) >= J:
                raise ValueError(
                    "Public memory capacity exceeded; refusing to truncate"
                )
            source_turn = before["turn"] if before is not None else turn
            is_discard = before is not None and before["phase"] == "discard_choice"
            self.followup = self.followup + 1 if is_discard else 0
            flags = [
                initial,
                before is not None and not is_discard,
                is_discard,
                before is not None
                and (terminal or state["game"]["age"] != before["age"]),
                terminal,
            ]
            self.public.append(
                {
                    "public": current,
                    "discard_count": discard,
                    "source_turn": source_turn,
                    "followup_seq": self.followup,
                    "phase_flags": np.asarray(flags, np.float32),
                }
            )
        if before is not None and before["turn"] != turn:
            self.followup = 0
        self.last_turn = turn
        for request_id, pending in list(self.pending.items()):
            receipt = next(
                (r for r in state["receipts"] if r.get("request_id") == request_id),
                None,
            )
            if receipt is None or receipt["status"] == "accepted":
                continue
            if receipt["status"] != "applied":
                raise ValueError("Unapplied decision cannot enter visible memory")
            if len(self.decisions) >= H:
                raise ValueError(
                    "Decision memory capacity exceeded; refusing to truncate"
                )
            self.decisions.append(pending)
            del self.pending[request_id]

    def observation(self, state, options):
        if self.game_id != state["game_id"] or not self.public:
            raise ValueError("Observe the game before building a policy input")
        obs = {k: np.zeros(shape, dtype) for k, (shape, dtype) in field_specs().items()}
        # Compute current public view directly too: useful for pure feature tests,
        # and prevents a stale cache from silently changing current features.
        for k, value in public_snapshot(state, self.seat).items():
            obs["public." + k][...] = value
        decision = next(
            (d for d in state["decisions"] if d["actor"] == self.seat), None
        )
        kind = decision["kind"] if decision else None
        if kind not in (None, "regular", "seventh", "discard"):
            raise ValueError("Unknown own decision kind")
        if decision is not None:
            obs["hand_counts"][:] = counts(
                state["players"][str(self.seat)]["hand_cards"]
            )
            obs["hand_visible"][...] = True
        if kind == "discard":
            unique = {o["card"]["instance_id"]: o["card"] for o in decision["options"]}
            obs["discard_offer_counts"][:] = counts(unique.values())
            obs["discard_offer_visible"][...] = True
        game = state["game"]
        terminal = game["phase"] == "finished"
        primary = (
            "none"
            if terminal
            else ("seventh" if game["round_in_age"] == 7 else "regular")
        )
        time = obs["time"]
        time[:4] = [
            min(game["age"], 3),
            game["round_in_age"] or 0,
            game["turn"],
            not terminal and game["age"] != 2,
        ]
        time[4 + ("regular", "seventh", "none").index(primary)] = 1
        time[7 + ("regular", "seventh", "discard", None).index(kind)] = 1
        time[11:] = [
            len(self.public) - 1,
            self.followup + (kind == "discard"),
            len(self.decisions),
        ]
        obs["discard_count"][...] = state["discard_count"]
        obs.update(candidate_features(state, self.seat, options))
        for i, row in enumerate(self.public):
            obs["public_history.valid"][i] = True
            for k, value in row["public"].items():
                obs["public_history.public." + k][i] = value
            for k in ("source_turn", "followup_seq", "phase_flags", "discard_count"):
                obs["public_history." + k][i] = row[k]
        for i, row in enumerate(self.decisions):
            obs["decision_history.valid"][i] = True
            obs["decision_history.applied_mask"][i] = True
            for k, value in row.items():
                obs["decision_history." + k][i] = value
        return obs

    def remember(self, request, obs, action, option):
        """Freeze predecision fields now; only observe(applied) publishes them."""
        if request["actor"] != self.seat or request["request_id"] in self.pending:
            raise ValueError("Wrong seat or duplicate pending decision")
        if len(self.decisions) + len(self.pending) >= H:
            raise ValueError("Decision memory capacity exceeded; refusing to truncate")
        row = {
            k: v.copy()
            for k, v in obs.items()
            if k.startswith("public.") or k in CURRENT_SHAPES
        }
        row["applied_action_feats"] = obs["action_feats"][action].copy()
        row["applied_card"] = np.zeros(C, np.float32)
        row["applied_card"][card_index(option["card"])] = 1
        self.pending[request["request_id"]] = row

    @staticmethod
    def decision_view(obs):
        """Private predecision data, independent of any observation given to a caller.

        History tensors dominate observation size but are not used by remember().
        Keep candidate features too: an untrusted policy may mutate its input
        before returning the chosen index.
        """
        return {
            k: v.copy()
            for k, v in obs.items()
            if k.startswith("public.") or k in CURRENT_SHAPES or k == "action_feats"
        }
