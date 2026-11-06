"""Runtime optimizations preserve inputs, temporal positions and gradients."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from sevenwonders_rl.envs import DecisionEnv
from sevenwonders_rl.envs import decision_env as module
from sevenwonders_rl.features.decision_memory import DecisionMemory
from sevenwonders_rl.utils.decision_ppo import DecisionActorCritic, tensors


def test_observation_mutation_cannot_change_confirmed_memory_or_contract(monkeypatch):
    calls = []
    original = module.observation
    saved = {}

    def observe(*args):
        calls.append(args[1])
        return original(*args)

    monkeypatch.setattr(module, "observation", observe)

    def opponent(seat):
        def choose(obs, rng):
            saved[seat] = DecisionMemory.decision_view(obs)
            action = int(np.flatnonzero(obs["action_mask"])[0])
            for value in obs.values():
                value[...] = 0
            return action

        return choose

    env = DecisionEnv(opponent_policies={1: opponent(1), 2: opponent(2)})
    try:
        obs, info = env.reset(seed=23)
        expected = DecisionMemory.decision_view(obs)
        action = int(np.flatnonzero(obs["action_mask"])[0])
        info["feature_contract"]["fields"]["time"]["shape"].append(999)
        for value in obs.values():
            value[...] = 0
        calls.clear()
        _, _, _, _, new_info = env.step(action)
        assert calls.count(0) == 1  # Only the following observation is built.
        assert new_info["feature_contract"] == env.contract
        saved[0] = expected
        for seat in range(3):
            confirmed = env.memories[seat].decisions[0]
            for key, value in saved[seat].items():
                if key != "action_feats":
                    np.testing.assert_array_equal(confirmed[key], value)
            selected = action if seat == 0 else 0
            np.testing.assert_array_equal(
                confirmed["applied_action_feats"], saved[seat]["action_feats"][selected]
            )
    finally:
        env.close()


@pytest.mark.parametrize("sparse", [False, True])
def test_trimmed_history_matches_full_forward_gradient_and_padding(sparse):
    torch.set_num_threads(1)
    env = DecisionEnv()
    observations = []
    try:
        obs, _ = env.reset(seed=24)
        for _ in range(9):
            observations.append(obs)
            obs, _, done, _, _ = env.step(int(np.flatnonzero(obs["action_mask"])[0]))
            if done:
                break
    finally:
        env.close()
    examples = deepcopy([observations[0], observations[-1]])
    if sparse:
        for prefix in ("decision_history.", "public_history."):
            for key, value in examples[-1].items():
                if key.startswith(prefix):
                    value[-1] = value[0]
            examples[-1][prefix + "valid"][-1] = True
    batch = tensors(examples)
    trimmed = DecisionActorCritic()
    full = deepcopy(trimmed)
    full._history_length = lambda valid, positions: len(positions)
    actual = trimmed(batch)
    expected = full(batch)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
    for model, (logits, value) in ((trimmed, actual), (full, expected)):
        (logits[batch["action_mask"].bool()].mean() + value.square().mean()).backward()
    for a, b in zip(trimmed.parameters(), full.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-5)
    junk = {k: v.clone() for k, v in batch.items()}
    for prefix in ("decision_history.", "public_history."):
        valid = junk[prefix + "valid"].bool()
        for key, value in junk.items():
            if key.startswith(prefix) and key != prefix + "valid":
                value[~valid] = 999
    for a, b in zip(actual, trimmed(junk)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not any(
        k.startswith(("_decision_positions", "_public_positions"))
        for k in trimmed.state_dict()
    )
    restored = DecisionActorCritic()
    restored.load_state_dict(trimmed.state_dict(), strict=True)
    for key, value in tensors([examples[0]]).items():
        torch.testing.assert_close(
            value,
            torch.tensor(np.stack([examples[0][key]]), dtype=torch.float32),
            rtol=0,
            atol=0,
        )


pytestmark = pytest.mark.integration
