"""Observable history, actor/critic information parity and end-to-end PPO use."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from sevenwonders_rl.envs import LegacySevenWondersEnv as SevenWondersEnv, SeatPolicy
from sevenwonders_rl.features.obs_builder import build_obs
from sevenwonders_rl.features.observation_history import (
    ObservationHistory, MEMORY_KEY, MEMORY_DIM, BASE_GLOBAL_DIM, ACTION_DIM,
    FRAME_DIM, HISTORY_LENGTH, CARD_IDS, CURRENT_HAND, HAND, HAND_KNOWN,
    PUBLIC, OWN_CARD, OWN_KIND, ACK, MASK, ACTION_FEATURES, FRAMES, VALID,
    card_counts, hand_is_observable, hide_skipped_hand)
from sevenwonders_rl.utils.ppo_model import (
    ActorCritic, MultiHeadActorCritic, extend_observation_memory)
from sevenwonders_rl.utils.ppo_utils import flatten_global, action_features, select_action_from_obs
from sevenwonders_rl.utils.checkpoints import save_checkpoint, load_model


def first_status(fixture):
    status = copy.deepcopy(fixture)
    status['game'].update(turn=0, era=1, phase='playing')
    return status


def payload(status, seat=0):
    obs = build_obs(status, seat, public_information=True)
    history = ObservationHistory()
    history.observe(0, obs)
    return obs, history, history.payload(obs)


@pytest.mark.component
def test_history_is_observation_only_and_hand_order_independent(status_age1_turn0):
    status = first_status(status_age1_turn0)
    original, history, before = payload(status)
    changed = copy.deepcopy(status)
    for seat in ('1', '2'):
        changed['players'][seat].update(cards_hand=['private-invalid'], cards_hand_id=[-999],
            cards_playable=[], can_build_wonder=False, can_build_hand_free=False)
    changed['last_actions'] = {'1': {'argument': 'private-invalid', 'status': 'applied'}}
    _, _, other = payload(changed)
    np.testing.assert_array_equal(before, other)
    reordered = copy.deepcopy(status)
    for key in ('cards_hand', 'cards_hand_id'):
        reordered['players']['0'][key] = list(reversed(reordered['players']['0'][key]))
    _, reordered_history, _ = payload(reordered)
    np.testing.assert_array_equal(history.frames, reordered_history.frames)
    # All own cards are recorded, not just currently affordable builds.
    np.testing.assert_array_equal(before[CURRENT_HAND][:-1], card_counts(original['hand_ids']))


@pytest.mark.component
def test_acknowledgements_cannot_mutate_past_training_inputs(status_age1_turn0):
    status = first_status(status_age1_turn0)
    obs, history, before = payload(status)
    saved = before.copy()
    own_card = int(obs['hand_ids'][0])
    history.acknowledge(0, own_card, 1, 'applied')
    history.observe(0, obs)  # Repeated feature/inference requests do not append.
    after = history.payload(obs)
    np.testing.assert_array_equal(before, saved)
    assert not before.flags.writeable
    frames = after[FRAMES].reshape(HISTORY_LENGTH, FRAME_DIM)
    np.testing.assert_array_equal(frames[0, OWN_CARD], card_counts([own_card]))
    assert frames[0, OWN_KIND.start + 1] == 1 and frames[0, ACK.start] == 1
    assert after[VALID].sum() == 1
    with pytest.raises(ValueError, match='already acknowledged'):
        history.acknowledge(0, own_card, 1, 'applied')
    with pytest.raises(ValueError, match='missing an earlier'):
        history.observe(2, obs)


@pytest.mark.component
def test_history_preserves_what_was_seen_not_a_guessed_current_hand(status_age1_turn0):
    status = first_status(status_age1_turn0)
    obs, history, _ = payload(status)
    seen = card_counts(obs['hand_ids'])
    history.acknowledge(0, int(obs['hand_ids'][0]), 3, 'applied')
    next_status = copy.deepcopy(status)
    next_status['game']['turn'] = 1
    next_status['players']['0']['cards_hand'] = status['players']['1']['cards_hand'][:-1]
    next_status['players']['0']['cards_hand_id'] = status['players']['1']['cards_hand_id'][:-1]
    current = build_obs(next_status, 0, public_information=True)
    history.observe(1, current)
    frames = history.payload(current)[FRAMES].reshape(HISTORY_LENGTH, FRAME_DIM)
    np.testing.assert_array_equal(frames[0, HAND], seen)
    np.testing.assert_array_equal(frames[1, HAND], card_counts(current['hand_ids']))


@pytest.mark.component
def test_shared_context_is_invariant_to_hand_slot_order(status_age1_turn0):
    torch.manual_seed(101)
    obs, history, memory = payload(first_status(status_age1_turn0))
    obs[MEMORY_KEY] = memory
    model = extend_observation_memory(ActorCritic(BASE_GLOBAL_DIM, ACTION_DIM, hidden=32,
                                                public_information=True))
    with torch.no_grad():
        model.information_encoder.projection.weight.fill_(.01)
    permutation = np.asarray([4, 2, 6, 0, 1, 5, 3])
    actions = np.concatenate([permutation + 7 * kind for kind in range(4)])
    changed = {**obs, 'hand_ids': obs['hand_ids'][permutation],
        'action_feats': obs['action_feats'][actions],
        'wonder_action_feats': obs['wonder_action_feats'][actions],
        'action_mask': obs['action_mask'][actions]}
    changed[MEMORY_KEY] = history.payload(changed)
    def predict(o):
        return model.forward_batch(torch.from_numpy(flatten_global(o, model)).unsqueeze(0),
                                   torch.from_numpy(action_features(o, model)).unsqueeze(0))
    logits, value = predict(obs)
    other_logits, other_value = predict(changed)
    torch.testing.assert_close(other_logits, logits[:, actions], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(other_value, value, rtol=1e-5, atol=1e-5)


@pytest.mark.component
@pytest.mark.parametrize('heads', [1, 4])
def test_neutral_migration_roundtrip_and_schema_validation(status_age1_turn0, tmp_path, heads):
    torch.manual_seed(30)
    cls = ActorCritic if heads == 1 else MultiHeadActorCritic
    kwargs = {} if heads == 1 else dict(num_heads=heads, prior_strength=32)
    source = cls(BASE_GLOBAL_DIM, ACTION_DIM, hidden=16, public_information=True, **kwargs)
    rng = torch.get_rng_state().clone()
    model = extend_observation_memory(source)
    assert torch.equal(rng, torch.get_rng_state())
    obs, _, memory = payload(first_status(status_age1_turn0))
    obs[MEMORY_KEY] = memory
    old_g = torch.from_numpy(flatten_global(obs, source)).unsqueeze(0)
    g = torch.from_numpy(flatten_global(obs, model)).unsqueeze(0)
    a = torch.from_numpy(action_features(obs, model)).unsqueeze(0)
    head_args = ([2],) if heads > 1 else ()
    expected = source.forward_batch(old_g, a, *head_args)
    actual = model.forward_batch(g, a, *head_args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.forward_values(g, *head_args), actual[1], rtol=0, atol=0)
    if heads > 1:
        for key, value in source.priors.state_dict().items():
            assert torch.equal(value, model.priors.state_dict()[key])
    path = tmp_path / 'memory.pt'
    save_checkpoint(path, model, None, 1,
        dict(global_dim=g.shape[-1], action_feat_dim=ACTION_DIM, hidden=16),
        {'feature_semantics_version': 3})
    restored = load_model(path)
    assert restored.memory_enabled and restored.num_heads == heads
    torch.testing.assert_close(restored.forward_batch(g, a, *head_args), actual, rtol=0, atol=0)
    bad_obs = {k: v for k, v in obs.items() if k != MEMORY_KEY}
    with pytest.raises(ValueError, match='requires an environment'):
        flatten_global(bad_obs, model)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint['meta']['observation_memory_schema_hash'] = 'different'
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='memory schema'):
        load_model(path, allow_legacy=True)


@pytest.mark.component
@pytest.mark.parametrize('output', ['actor', 'critic'])
def test_both_outputs_can_use_the_same_hand_history_and_action_information(status_age1_turn0, output):
    torch.manual_seed(100)
    obs, _, memory = payload(first_status(status_age1_turn0))
    obs[MEMORY_KEY] = memory
    model = extend_observation_memory(ActorCritic(BASE_GLOBAL_DIM, ACTION_DIM, hidden=32,
                                                public_information=True))
    with torch.no_grad():
        model.information_encoder.projection.weight.fill_(.01)
    g = torch.from_numpy(flatten_global(obs, model)).unsqueeze(0).requires_grad_()
    a = torch.from_numpy(action_features(obs, model)).unsqueeze(0)
    logits, values = model.forward_batch(g, a)
    (logits[0, 0] if output == 'actor' else values[0]).backward()
    gradient = g.grad[0, BASE_GLOBAL_DIM:]
    assert gradient[CURRENT_HAND].abs().sum() > 0
    assert gradient[MASK].abs().sum() > 0
    assert gradient[ACTION_FEATURES].abs().sum() > 0
    assert gradient[FRAMES].reshape(HISTORY_LENGTH, FRAME_DIM)[0, HAND].abs().sum() > 0
    # Unobserved future slots cannot influence either output, even with nonzero padding.
    future = g.detach().clone()
    future[0, BASE_GLOBAL_DIM + FRAMES.start + FRAME_DIM:BASE_GLOBAL_DIM + FRAMES.stop] = 999
    torch.testing.assert_close(model.forward_batch(future, a), (logits.detach(), values.detach()), rtol=0, atol=0)


@pytest.mark.integration
def test_real_game_has_isolated_seat_histories_no_seventh_card_peek_and_clean_reset(require_engine, tmp_path):
    env = SevenWondersEnv(repo_root=str(Path(__file__).parents[2]),
        io_dir=str(tmp_path / 'io'), observation_memory=True)
    try:
        obs, info = env.reset(seed=601)
        first_payload = obs[MEMORY_KEY].copy()
        assert env.observation_space.contains(obs)
        for turn in range(21):
            own = env.observation_for_seat(0)
            other = env.observation_for_seat(1)
            assert env._observation_histories[0] is not env._observation_histories[1]
            np.testing.assert_array_equal(own[MEMORY_KEY][CURRENT_HAND][:-1], card_counts(own['hand_ids']))
            np.testing.assert_array_equal(other[MEMORY_KEY][CURRENT_HAND][:-1], card_counts(other['hand_ids']))
            if turn % 7 == 6:  # Our learner always discards, so never earns seventh-card play.
                assert own[MEMORY_KEY][CURRENT_HAND].sum() == 0
                assert not own['action_feats'].any() and np.all(own['hand_ids'] == -1)
                assert own['action_mask'].sum() == 1 and own['action_mask'][21]
            action = int(np.flatnonzero(own['action_mask'][21:] > 0)[0]) + 21
            obs, _, done, truncated, info = env.step(action)
            assert not truncated
        assert done and info['status']['game']['phase'] == 'finished'
        frames = obs[MEMORY_KEY][FRAMES].reshape(HISTORY_LENGTH, FRAME_DIM)
        assert frames[:, ACK].sum() == 21
        for turn in (6, 13, 20):
            assert not frames[turn, HAND].any() and frames[turn, HAND_KNOWN] == 0
            assert not frames[turn, OWN_CARD].any()
        again, _ = env.reset(seed=601)
        np.testing.assert_array_equal(again[MEMORY_KEY], first_payload)
        assert again[MEMORY_KEY][VALID].sum() == 1
    finally:
        env.close()


@pytest.mark.integration
def test_memory_ppo_training_checkpoint_opponents_and_detailed_evaluation(require_engine, tmp_path):
    root = Path(__file__).parents[1]
    output = tmp_path / 'training'
    result = subprocess.run([sys.executable, 'scripts/ppo_training/train_unified.py',
        '--num-heads', '4', '--prior-strength', '32',
        '--total-timesteps', '128', '--n-envs', '2', '--rollout-horizon', '32',
        '--batch-size', '64', '--snapshot-every', '64', '--eval-every', '128',
        '--eval-games-per-matchup', '1', '--seed', '603', '--out-dir', str(output)],
        cwd=root, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = torch.load(output / 'latest.pt', weights_only=True)
    model = load_model(output / 'latest.pt')
    assert model.memory_enabled and model.num_heads == 4
    assert checkpoint['config']['initialization']['memory_input_migration']
    updates = [json.loads(line) for line in (output / 'updates.jsonl').read_text().splitlines()]
    assert updates and all(np.isfinite(row['value_loss']) for row in updates)
    assert model.information_encoder.projection.weight.abs().sum() > 0
    initial = load_model(output / 'checkpoint_0.pt')
    for part in ('hand', 'opportunities', 'frame', 'sequence'):
        assert any(not torch.equal(before, after) for before, after in zip(
            getattr(initial.information_encoder, part).parameters(),
            getattr(model.information_encoder, part).parameters()))
    for key, value in initial.priors.state_dict().items():
        assert torch.equal(value, model.priors.state_dict()[key])
    # Exercise the same model as learner and as shared cached opponents, in both paths.
    from sevenwonders_rl.evaluation.runner import evaluate_model
    from sevenwonders_rl.evaluation.selfplay import reference
    opponent = reference(output / 'checkpoint_64.pt')
    specs = [dict(engine_seed=800 + seat, agent_seat=seat, left=opponent, right=opponent)
             for seat in range(3)]
    records = []
    for mode, workers in ((True, 1), (False, 1), (True, 2)):
        path = tmp_path / f'eval-{mode}-{workers}.jsonl'
        report = evaluate_model(model, 'cpu', specs, root.parent, tmp_path / f'io-{mode}', path,
            action_mode='stochastic', detailed=True, batch_opponents=mode, workers=workers)
        assert report['completed'] == 3 and report['failed'] == 0
        records.append([json.loads(line) for line in path.read_text().splitlines()])
    for rows in zip(*records):
        for key in ('totals', 'actions', 'head_id', 'opponent_head_ids'):
            assert rows[0][key] == rows[1][key] == rows[2][key]
