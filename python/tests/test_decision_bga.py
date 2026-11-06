"""Active BGA data, observation privacy, decisions and real PPO contracts."""

from copy import deepcopy
from pathlib import Path
import json
import math
import subprocess
import numpy as np
import pytest
import torch
from sevenwonders_rl.envs import DecisionEnv, SevenWondersEnv
from sevenwonders_rl.features.decision_features import (
    dataset,
    observation,
    optimal_science,
    science_inventory,
    feature_contract,
    INDEX,
    N_OPTIONS,
    K,
)
from sevenwonders_rl.utils.decision_ppo import (
    train,
    load_checkpoint,
    DecisionActorCritic,
)

ROOT = Path(__file__).resolve().parents[2]


def priority(obs, rng):
    # Public features only; no raw engine state reaches this policy.
    from sevenwonders_rl.features.decision_features import INDEX

    for kind in ("build_discard_free", "build_wonder", "build_structure", "discard"):
        for n in np.flatnonzero(obs["action_mask"]):
            if obs["action_feats"][n, INDEX["kind_" + kind]]:
                return int(n)
    raise AssertionError("Missing legal option")


def make(seat=0, **kwargs):
    return DecisionEnv(
        agent_seat=seat,
        allow_development=True,
        opponent_policies={s: priority for s in range(3) if s != seat},
        **kwargs,
    )


def test_new_default_and_neutral_data_are_explicit():
    assert SevenWondersEnv is DecisionEnv
    d = dataset()
    assert len(d["cards"]) == 80 and len(d["wonders"]) == 14
    assert (73, 3) not in d["cards"]
    assert d["cards"][(76, 3)]["counts_by_players"]["3"] == 0
    assert d["cards"][(27, 3)]["cost"] == {"stone": 2, "ore": 1, "glass": 1}
    assert d["cards"][(15, 1)]["counts_by_players"]["5"] == 1
    assert d["cards"][(15, 2)]["counts_by_players"]["5"] == 2
    assert len(d["wonders"][0]["stages"]) == 3 and len(d["wonders"][8]["stages"]) == 2
    subprocess.run(
        ["python3", str(ROOT / "python/scripts/export_bga_data.py"), "--check"],
        check=True,
    )


def test_science_optimizer_against_independent_recursion():
    def oracle(g, c, t, j):
        if not j:
            return g * g + c * c + t * t + 7 * min(g, c, t)
        return max(
            oracle(g + 1, c, t, j - 1),
            oracle(g, c + 1, t, j - 1),
            oracle(g, c, t + 1, j - 1),
        )

    for g in range(5):
        for c in range(5):
            for t in range(5):
                for j in range(3):
                    assert optimal_science((g, c, t), j) == oracle(g, c, t, j)


def test_private_opponent_hands_never_change_observation_and_padding_is_rejected():
    e = make()
    try:
        obs, info = e.reset(seed=8)
        state = deepcopy(e.state)
        for s in (1, 2):
            state["players"][str(s)]["hand_cards"] = [{"private": "unavailable"}] * 99
        for d in state["decisions"]:
            if d["actor"] != 0:
                d["options"] = [{"private": "unavailable"}]
        other = observation(state, 0, e.options)
        for key in obs:
            np.testing.assert_array_equal(obs[key], other[key])
        assert "status" not in info
        original = deepcopy(e.state)
        with pytest.raises(ValueError):
            e.step(N_OPTIONS - 1)
        assert e.state == original and not (e.client.exchange / "ready.json").exists()
    finally:
        e.close()


def test_actual_wonder_features_describe_stage_and_foregone_hand_card():
    e = make()
    try:
        obs, _ = e.reset(seed=8)
        state = deepcopy(e.state)
        own = state["players"]["0"]
        own["wonder_id"] = 8
        own["wonder_stage"] = 1
        card = {"type_id": 30, "age": 3, "instance_id": "999", "name": "Palace"}
        option = {
            "card": card,
            "kind": "build_wonder",
            "payment": {"bank": 0, "to_seats": {"0": 0, "1": 0, "2": 0}},
            "free_source": None,
        }
        x = observation(state, 0, [option])["action_feats"][0]
        assert x[INDEX["effect_joker"]] == 1 and x[INDEX["card_vp"]] == 1
        assert x[INDEX["effect_vp"]] == 0 and x[INDEX["next_stage"]] == 0.5
        own["wonder_id"] = 0
        own["wonder_stage"] = 0
        own["built_cards"] = [
            {"type_id": i, "age": age}
            for i, age in ((55, 1), (58, 2), (54, 1), (74, 3))
        ]
        tablet = deepcopy(option)
        tablet["kind"] = "build_structure"
        tablet["card"] = {"type_id": 59, "age": 2}
        assert (
            observation(state, 0, [tablet])["action_feats"][0, INDEX["science_gain"]]
            * 20
            == 5
        )
        own["wonder_id"] = 8

        # Inventory includes both an existing guild joker and the target Babylon stage.
        own["built_cards"] = [{"type_id": 74, "age": 3}]
        own["wonder_stage"] = 2
        symbols, j = science_inventory(own)
        assert j == 2 and optimal_science(symbols, j) == 4
    finally:
        e.close()


def test_real_games_every_seat_special_decisions_and_receipt_history():
    seen = set()
    histories = []
    for seat, seed in ((0, 0), (1, 7), (2, 8), (2, 6), (2, 9)):
        e = make(seat)
        try:
            obs, info = e.reset(seed=seed)
            before = [v.copy() for v in obs.values()]
            steps = 0
            while True:
                assert e.observation_space.contains(obs)
                # Prefer visible free modes, then wonder construction.
                free = [
                    n
                    for n, o in enumerate(e.options)
                    if str(o["free_source"]).startswith("olympia_")
                ]
                action = free[0] if free else priority(obs, None)
                prior = deepcopy(obs)
                obs, reward, done, truncated, info = e.step(action)
                steps += 1
                assert (
                    not truncated
                    and info["transition"]["receipt"]["status"] == "applied"
                )
                seen.add(info["transition"]["option"]["kind"])
                seen.add(info["transition"]["option"]["free_source"])
                assert "hand_cards" not in json.dumps(info)
                if done:
                    assert not obs[
                        "action_mask"
                    ].any() and e.observation_space.contains(obs)
                    totals = info["final_totals"]
                    assert reward == totals[seat] - max(
                        t for s, t in enumerate(totals) if s != seat
                    )
                    assert len(e.history) == steps and 18 <= steps <= 24
                    histories.append(
                        [
                            (h["turn_before"], h["turn_after"], h["option"]["kind"])
                            for h in e.history
                        ]
                    )
                    break
                for player in e.state["players"].values():
                    symbols, jokers = science_inventory(player)
                    assert (
                        optimal_science(symbols, jokers)
                        == player["points"]["scientific"]
                    )
                assert steps < 30
        finally:
            e.close()
    assert {
        "build_discard_free",
        "olympia_first_color",
        "olympia_first_age_card",
        "olympia_last_age_card",
    }.issubset(seen), seen
    assert any(a == b for trace in histories for a, b, _ in trace), (
        "No same-turn follow-up decision exercised"
    )


def test_seed_replay_is_independent_of_game_identity():
    traces = []
    for _ in range(2):
        e = make(2)
        try:
            obs, info = e.reset(seed=8)
            game = info["game_id"]
            trace = []
            while True:
                action = priority(obs, None)
                o, r, d, _, i = e.step(action)
                trace.append(
                    (
                        i["turn"],
                        i["transition"]["option"]["card"]["type_id"],
                        i["transition"]["option"]["kind"],
                        r,
                    )
                )
                obs = o
                if d:
                    break
            traces.append((game, trace))
        finally:
            e.close()
    assert traces[0][0] != traces[1][0] and traces[0][1] == traces[1][1]


def test_ppo_real_updates_reload_resume_and_legacy_rejection(tmp_path):
    e = make()
    try:
        model, metrics = train(e, 64, 42, tmp_path / "first", rollout_size=32, epochs=2)
        assert (
            metrics["actually_applied"] == 64
            and metrics["completed_games"] >= 2
            and math.isfinite(metrics["loss_mean"])
        )
        loaded, data = load_checkpoint(
            tmp_path / "first/checkpoint.pt", metrics["contract"]
        )
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
        _, resume = train(
            e,
            16,
            43,
            tmp_path / "resume",
            tmp_path / "first/checkpoint.pt",
            rollout_size=32,
            epochs=2,
        )
        assert resume["steps"] == 80
        legacy = tmp_path / "legacy.pt"
        torch.save(
            {"state_dict": model.state_dict(), "meta": {"protocol_version": 2}}, legacy
        )
        with pytest.raises(ValueError, match="contract mismatch"):
            load_checkpoint(legacy, metrics["contract"])
        bad = deepcopy(metrics["contract"])
        bad["protocol_version"] = 2
        with pytest.raises(ValueError, match="contract mismatch"):
            load_checkpoint(tmp_path / "first/checkpoint.pt", bad)
        torch.manual_seed(42)
        initial = DecisionActorCritic()
        assert any(
            not torch.equal(v, initial.state_dict()[k])
            for k, v in model.state_dict().items()
        )
    finally:
        e.close()


def test_fresh_ppo_seed_repeats_without_game_identity_leak():
    states = []
    for _ in range(2):
        e = make()
        try:
            model, metrics = train(e, 32, 71, rollout_size=32, epochs=1)
            states.append({k: v.clone() for k, v in model.state_dict().items()})
        finally:
            e.close()
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


pytestmark = pytest.mark.integration
