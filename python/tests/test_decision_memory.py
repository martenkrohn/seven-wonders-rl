"""Acceptance tests for protocol-3 visible memory, including real engine games."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

from sevenwonders_rl.envs import DecisionEnv
from sevenwonders_rl.features.decision_features import (
    dataset,
    feature_contract,
    observation,
)
from sevenwonders_rl.features.decision_memory import (
    ABILITIES,
    C,
    H,
    J,
    DecisionMemory,
    card_index,
    counts,
    stage_vector,
)
from sevenwonders_rl.utils.decision_ppo import (
    DecisionActorCritic,
    tensors,
    train,
    load_checkpoint,
)


def priority(obs, rng=None):
    from sevenwonders_rl.features.decision_features import INDEX

    for kind in ("build_discard_free", "build_wonder", "build_structure", "discard"):
        choices = np.flatnonzero(
            obs["action_mask"] & (obs["action_feats"][:, INDEX["kind_" + kind]] > 0)
        )
        if len(choices):
            return int(choices[0])
    raise AssertionError("No legal candidate")


def equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@pytest.fixture
def live():
    env = DecisionEnv(opponent_policies={1: priority, 2: priority})
    try:
        obs, _ = env.reset(seed=8)
        yield env, obs
    finally:
        env.close()


def test_full_hand_counts_order_invariance_and_duplicate_options(live):
    env, obs = live
    state = deepcopy(env.state)
    own = state["players"]["0"]
    assert obs["hand_counts"].sum() == len(own["hand_cards"]) == 7
    assert obs["action_mask"].sum() > obs["hand_counts"].sum()
    own["hand_cards"].reverse()
    equal(obs, observation(state, 0, env.options))
    # Hand context comes from hand_cards, even if no offered build describes it.
    extra = dict(type_id=30, age=3, instance_id="extra")
    own["hand_cards"] += [extra, dict(extra, instance_id="another")]
    other = observation(state, 0, env.options)
    assert (
        other["hand_counts"][card_index(extra)]
        == obs["hand_counts"][card_index(extra)] + 2
    )
    np.testing.assert_array_equal(other["action_feats"], obs["action_feats"])
    assert (
        not other["discard_offer_visible"] and not other["discard_offer_counts"].any()
    )
    # Candidate permutation permutes logits only, not common hand/value context.
    torch.set_num_threads(1)
    model = DecisionActorCritic()
    a, va = model(tensors([obs]))
    reverse = observation(env.state, 0, env.options[::-1])
    b, vb = model(tensors([reverse]))
    n = len(env.options)
    torch.testing.assert_close(a[:, :n].flip(-1), b[:, :n])
    torch.testing.assert_close(va, vb)


def test_hidden_fields_transport_and_physical_ids_do_not_reach_model(live):
    env, obs = live
    state = deepcopy(env.state)
    state["game"]["engine_seed"] = "secret seed changed"
    state["revision"] = 999999
    state["waiting_for"] = ["transport-only"] * 50
    state["events"] = [{"card_instance_id": "buried secret", "kind": "regular_applied"}]
    state["grants"] = ["not a public log"]
    for s in (1, 2):
        state["players"][str(s)]["hand_cards"] = [{"secret": object()}]
    for d in state["decisions"]:
        if d["actor"] != 0:
            d["options"] = [{"secret": object()}]
    state["buried_cards"] = ["different hidden identities"]
    other = observation(state, 0, env.options)
    equal(obs, other)
    for s in range(3):
        for c in state["players"][str(s)]["built_cards"]:
            c["instance_id"] = "irrelevant"
    for c in state["players"]["0"]["hand_cards"]:
        c["instance_id"] = "new-" + c["instance_id"]
    equal(obs, observation(state, 0, env.options))
    model = DecisionActorCritic()
    for a, b in zip(model(tensors([obs])), model(tensors([other]))):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_discard_offer_is_separate_visible_and_physically_deduplicated(live):
    env, _ = live
    state = deepcopy(env.state)
    d = next(d for d in state["decisions"] if d["actor"] == 0)
    d["kind"] = "discard"
    first = deepcopy(d["options"][0])
    first["card"] = dict(type_id=30, age=3, instance_id="pile1")
    second = deepcopy(first)
    second["card"]["instance_id"] = "pile2"
    second["selectable"] = False
    d["options"] = [first, deepcopy(first), second]
    state["players"]["0"]["hand_cards"] = []
    obs = observation(state, 0, [])
    assert obs["hand_visible"] and obs["hand_counts"].sum() == 0
    assert obs["discard_offer_visible"] and obs["discard_offer_counts"].sum() == 2
    assert obs["time"][9] == 1 and obs["time"][12] == 1
    assert not obs["action_mask"].any()


def test_receipt_gating_snapshot_immutability_and_rejected_actions(live):
    env, obs = live
    memory = DecisionMemory(0)
    memory.observe(env.state)
    request = {"actor": 0, "request_id": "own-request"}
    memory.remember(request, obs, 0, env.options[0])
    frozen = deepcopy(memory.pending["own-request"])
    obs["hand_counts"][:] = 99
    state = deepcopy(env.state)
    state["receipts"] = [dict(request_id="own-request", status="accepted")]
    memory.observe(state)
    assert not memory.decisions
    equal(memory.pending["own-request"], frozen)
    state["receipts"][0]["status"] = "applied"
    memory.observe(state)
    assert len(memory.decisions) == 1 and not memory.pending
    memory.observe(state)
    assert len(memory.decisions) == 1 and len(memory.public) == 1
    equal(memory.decisions[0], frozen)
    memory.remember(
        dict(request, request_id="bad"),
        observation(state, 0, env.options, memory),
        0,
        env.options[0],
    )
    state["receipts"].append(dict(request_id="bad", status="rejected"))
    with pytest.raises(ValueError, match="Unapplied"):
        memory.observe(state)
    assert len(memory.decisions) == 1


def test_public_details_stage_effects_choice_sources_and_unknown_ids(live):
    env, _ = live
    state = deepcopy(env.state)
    p = state["players"]["0"]
    p["wonder_id"] = 12  # Alexandria B: raw and manufactured choice stages.
    p["wonder_stage"] = 2
    state["decisions"] = []
    obs = observation(state, 0, [])
    assert obs["public.wonder_side"][0, 12] == 1
    assert obs["public.stage_built"][0].sum() == 2
    assert obs["public.stage_exists"][0].sum() == len(
        dataset()["wonders"][12]["stages"]
    )
    assert obs["public.choice_source_mask"][0, C : C + 2].all()
    assert obs["public.choice_sources"][0, C, :7].sum() == 4
    assert obs["public.choice_sources"][0, C + 1, :7].sum() == 3
    assert not obs["public.choice_sources"][0, C : C + 2, 8].any()
    assert not obs["hand_visible"]
    # Known absent right differs from an unlocked but currently unproven grant.
    p["wonder_id"] = 13  # Halikarnassos B.
    p["wonder_stage"] = 1
    p["available_abilities"] = []
    obs = observation(state, 0, [])
    assert obs["public.ability_unlocked"][0, ABILITIES.index("discard_build_for_free")]
    assert not obs["public.ability_available_known"][0, 4]
    p["available_abilities"] = [
        dict(code="discard_build_for_free", availability="decision")
    ]
    obs = observation(state, 0, [])
    assert (
        obs["public.ability_available_known"][0, 4]
        and obs["public.ability_available"][0, 4]
    )
    with pytest.raises(ValueError, match="Unknown wonder effect"):
        stage_vector(dict(vp=0, cost={}, effects=[dict(type="future_unknown")]))
    with pytest.raises(ValueError, match="Unknown visible card"):
        counts([dict(type_id=999, age=1)])


def test_history_changes_actor_and_critic_gradients_order_and_padding(live):
    env, before = live
    obs, *_ = env.step(priority(before))
    torch.set_num_threads(1)
    torch.manual_seed(37)
    model = DecisionActorCritic()
    changed = deepcopy(obs)
    changed["decision_history.hand_counts"][0] = np.roll(
        changed["decision_history.hand_counts"][0], 13
    )
    a, va = model(tensors([obs]))
    b, vb = model(tensors([changed]))
    valid = obs["action_mask"]
    assert not torch.equal(a[:, valid], b[:, valid])
    assert not torch.equal(va, vb)
    for target in ("actor", "critic"):
        model.zero_grad()
        x, v = model(tensors([obs]))
        (x[:, valid].sum() if target == "actor" else v.sum()).backward()
        for encoder in (
            model.hand_encoder,
            model.decision_encoder,
            model.public_encoder,
            model.public_history_encoder,
            model.decision_history_encoder,
        ):
            assert (
                sum(
                    p.grad.abs().sum().item()
                    for p in encoder.parameters()
                    if p.grad is not None
                )
                > 0
            )
    junk = deepcopy(obs)
    for prefix in ("decision_history.", "public_history."):
        invalid = ~obs[prefix + "valid"]
        for k in junk:
            if k.startswith(prefix) and k != prefix + "valid":
                junk[k][invalid] = 1 if junk[k].dtype == bool else 19
    for x, y in zip(model(tensors([obs])), model(tensors([junk]))):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    # Same rows in another order must not be treated as an unordered bag.
    reordered = deepcopy(obs)
    for k in reordered:
        if k.startswith("public_history."):
            reordered[k][[0, 1]] = reordered[k][[1, 0]]
    x, v = model(tensors([reordered]))
    assert not torch.equal(a[:, valid], x[:, valid]) and not torch.equal(va, v)


@pytest.mark.parametrize("seat,seed", [(0, 0), (1, 7), (2, 8), (2, 6), (2, 9)])
def test_real_games_memories_all_seats_specials_reset_and_advance(seat, seed):
    shared_policy = priority
    env = DecisionEnv(
        agent_seat=seat,
        opponent_policies={s: shared_policy for s in range(3) if s != seat},
    )
    try:
        obs, _ = env.reset(seed=seed)
        first = deepcopy(obs)
        first_hands = {
            s: counts(env.state["players"][str(s)]["hand_cards"]) for s in range(3)
        }
        all_obs = []
        while True:
            saved = deepcopy(obs)
            all_obs.append(obs)
            prior_public = len(env.memories[seat].public)
            old_turn = env.state["game"]["turn"]
            obs, _, done, _, info = env.step(priority(obs))
            equal(all_obs[-1], saved)
            assert env.observation_space.contains(obs)
            assert obs["decision_history.valid"].sum() == len(env.history)
            assert obs["decision_history.applied_mask"].sum() == len(env.history)
            row = len(env.history) - 1
            np.testing.assert_array_equal(
                obs["decision_history.hand_counts"][row], saved["hand_counts"]
            )
            np.testing.assert_array_equal(
                obs["decision_history.time"][row], saved["time"]
            )
            np.testing.assert_array_equal(
                obs["decision_history.public.built_counts"][row],
                saved["public.built_counts"],
            )
            if prior_public < len(env.memories[seat].public):
                assert (
                    env.memories[seat].public[prior_public]["source_turn"] == old_turn
                )
            if done:
                break
        for s, memory in enumerate(env.memories):
            assert not memory.pending
            np.testing.assert_array_equal(
                memory.decisions[0]["hand_counts"], first_hands[s]
            )
            assert len(memory.decisions) <= 25 and len(memory.public) < J
            assert memory.public[-1]["phase_flags"][-1]
        # Every single applied opponent-only discard choice gets a public group.
        groups = env.memories[seat].public
        for prev, nxt in zip(groups, groups[1:]):
            if nxt["phase_flags"][2]:
                assert (
                    nxt["public"]["built_counts"].sum()
                    == prev["public"]["built_counts"].sum() + 1
                )
        if seed == 8:
            assert any(g["phase_flags"][2] for g in groups)
        equal(first, all_obs[0])
        previous_game = env.memories[seat].game_id
        fresh, _ = env.reset(seed=seed)
        assert env.memories[seat].game_id != previous_game
        assert not env.memories[seat].decisions
        assert fresh["decision_history.valid"].sum() == 0
        equal(first, fresh)
    finally:
        env.close()


def test_simultaneous_public_group_and_memory_overflow(live):
    env, _ = live
    state = deepcopy(env.state)
    for s in range(3):
        state["players"][str(s)]["built_cards"] = [
            dict(type_id=1 + s, age=1, instance_id=str(s))
        ]
    state["game"]["turn"] += 1
    state["game"]["round_in_age"] += 1
    a, b = DecisionMemory(0), DecisionMemory(0)
    a.observe(env.state)
    b.observe(env.state)
    a.observe(state, env.state["game"])
    state["events"] = [dict(actor=s) for s in (2, 1, 0)]
    state["players"] = dict(reversed(list(state["players"].items())))
    b.observe(state, env.state["game"])
    assert len(a.public) == len(b.public) == 2
    equal(a.observation(state, env.options), b.observation(state, env.options))
    a.public = [deepcopy(a.public[0]) for _ in range(J)]
    with pytest.raises(ValueError, match="Public memory capacity"):
        a.observe(state, env.state["game"])
    a.decisions = [{} for _ in range(H)]
    with pytest.raises(ValueError, match="Decision memory capacity"):
        a.remember(dict(actor=0, request_id="overflow"), {}, 0, env.options[0])
    other = deepcopy(state)
    other["game_id"] = "another-game"
    with pytest.raises(ValueError, match="cannot cross games"):
        b.observe(other)


def test_contract_hash_checkpoint_rejection_and_fresh_resume_seeds(live, tmp_path):
    env, _ = live
    contract = feature_contract(env.engine_metadata["ruleset"])
    unhashed = {k: v for k, v in contract.items() if k != "schema_hash"}
    raw = json.dumps(
        unhashed, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert hashlib.sha256(raw).hexdigest() == contract["schema_hash"]
    model, metrics = train(env, 24, 42, tmp_path / "first", rollout_size=16, epochs=1)
    loaded, saved = load_checkpoint(tmp_path / "first/checkpoint.pt", contract)
    obs = observation(
        env.state, env.agent_seat, env.options, env.memories[env.agent_seat]
    )
    if obs["action_mask"].any():
        for x, y in zip(model(tensors([obs])), loaded(tensors([obs]))):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert saved["format"] == "decision-ppo-v2"
    assert json.loads((tmp_path / "first/manifest.json").read_text())["provenance"][
        "engine_matches_pin"
    ]
    _, resumed = train(
        env, 2, 42, tmp_path / "resume", tmp_path / "first/checkpoint.pt", epochs=1
    )
    assert resumed["episode_seeds"][-1] > max(metrics["episode_seeds"])
    assert len(env.memories[env.agent_seat].decisions) == 2
    assert resumed["steps"] == 26
    for key, bad_value in [
        ("memory", "older"),
        ("schema_hash", "bad"),
        ("dataset_hash", "bad"),
    ]:
        bad = deepcopy(contract)
        bad[key] = bad_value
        with pytest.raises(ValueError, match="contract mismatch"):
            load_checkpoint(tmp_path / "first/checkpoint.pt", bad)
    saved["format"] = "decision-ppo-v1"
    torch.save(saved, tmp_path / "v1.pt")
    with pytest.raises(ValueError, match="contract mismatch"):
        load_checkpoint(tmp_path / "v1.pt", contract)


pytestmark = pytest.mark.integration
