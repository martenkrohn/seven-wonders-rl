"""Behavioral contracts for runtime optimizations."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sevenwonders_rl.features.obs_builder import ObservationContext, build_obs
from sevenwonders_rl.selfplay.opponent_pool import OpponentPool
from sevenwonders_rl.utils.ppo_model import ActorCritic


@pytest.mark.unit
def test_throughput_preset_keeps_samples_per_update_and_validates_counts():
    from test_training_contracts import trainer_module
    trainer = trainer_module()
    args = trainer.parse_args(['--throughput-preset'])
    assert args.n_envs == 8 and args.rollout_horizon == 64
    assert args.n_envs * args.rollout_horizon == 512
    with pytest.raises(SystemExit):
        trainer.parse_args(['--eval-workers', '0'])
    with pytest.raises(SystemExit):
        trainer.parse_args(['--opponent-cache-size', '0'])


@pytest.mark.component
def test_default_opponent_cache_covers_more_than_sixteen_snapshots(tmp_path, monkeypatch):
    from collections import OrderedDict
    from test_training_contracts import trainer_module
    trainer = trainer_module()
    loaded = []
    def load(path, **kwargs):
        loaded.append(path)
        return object()
    monkeypatch.setattr(trainer, 'PPOCheckpointPolicy', load)
    pool = OpponentPool(str(tmp_path), max_size=20)
    paths = [str(tmp_path / f'checkpoint_{i}.pt') for i in range(20)]
    sequence = iter(paths + paths[:2])
    monkeypatch.setattr(pool, 'sample_path', lambda *args: next(sequence))
    cache = OrderedDict()
    for _ in range(10):
        trainer.sample_selfplay_opponents(pool, np.random.default_rng(0), 20, 0, cache)
    first = cache[paths[0]]
    policies = trainer.sample_selfplay_opponents(pool, np.random.default_rng(0), 20, 0, cache)
    assert len(loaded) == len(cache) == 20
    assert policies[1].policy is first


@pytest.mark.component
def test_shared_observations_rotate_without_mutating_engine_status(status_age3_turn0, monkeypatch):
    import sevenwonders_rl.features.action_features as features
    original = copy.deepcopy(status_age3_turn0)
    calls = []
    resolve = features.built_ids
    def counted(player):
        calls.append(player)
        return resolve(player)
    monkeypatch.setattr(features, 'built_ids', counted)
    context = ObservationContext(status_age3_turn0)
    shared = [context.observation(seat) for seat in range(3)]
    assert len(calls) == 3
    assert context.observation(1) is shared[1]
    for seat, obs in enumerate(shared):
        separate = build_obs(status_age3_turn0, seat)
        for key in obs:
            np.testing.assert_array_equal(obs[key], separate[key])
        assert obs['global_table'][0, 0] == original['players'][str(seat)]['resources']['coins']
    assert status_age3_turn0 == original
    updated = copy.deepcopy(original)
    updated['players']['1']['resources']['coins'] += 7
    newer = ObservationContext(updated)
    assert newer.observation(1)['global_table'][0, 0] == shared[1]['global_table'][0, 0] + 7
    with pytest.raises(ValueError, match='another status'):
        build_obs(updated, 0, context=context)


@pytest.mark.unit
@pytest.mark.parametrize('batch', [1, 2, 8])
def test_critic_only_is_exact_and_does_not_evaluate_actor(batch):
    model = ActorCritic(102, 99, hidden=16)
    inputs = torch.randn(batch, 102)
    expected = model.forward_batch(inputs, torch.randn(batch, 28, 99))[1]
    def forbid(*args):
        pytest.fail('The actor must not run during bootstrap')
    model.action_mlp.register_forward_hook(forbid)
    torch.testing.assert_close(model.forward_values(inputs), expected, rtol=0, atol=0)


@pytest.mark.unit
def test_pool_index_registration_refresh_and_sampling_match_scan(tmp_path, monkeypatch):
    for step in (0, 4, 8):
        (tmp_path / f'checkpoint_{step}.pt').touch()
    indexed = OpponentPool(str(tmp_path), max_size=3, min_step_lag=2, auto_refresh=False)
    scanned = OpponentPool(str(tmp_path), max_size=3, min_step_lag=2)
    assert indexed.eligible(10) == scanned.eligible(10)
    path = tmp_path / 'checkpoint_12.pt'
    path.touch()
    indexed.register(path)
    a, b = np.random.default_rng(17), np.random.default_rng(17)
    assert [indexed.sample_path(a, t) for t in range(4, 20)] == [scanned.sample_path(b, t) for t in range(4, 20)]
    monkeypatch.setattr(Path, 'glob', lambda *a, **kw: pytest.fail('Registered snapshots must not trigger rescans'))
    indexed.sample_path(a, 20)
    monkeypatch.undo()
    (tmp_path / 'checkpoint_16.pt').touch()
    assert indexed.total_size() == 4
    indexed.refresh()
    assert indexed.total_size() == 5


@pytest.mark.integration
@pytest.mark.parametrize('action_mode,constraint', [('greedy', 'none'), ('stochastic', 'green_priority')])
def test_parallel_evaluation_matches_ordered_serial_records(require_engine, tmp_path, action_mode, constraint):
    from sevenwonders_rl.evaluation.runner import evaluate_model
    model = ActorCritic(102, 99, hidden=16)
    model.train()
    rng = torch.get_rng_state().clone()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    specs = [dict(engine_seed=300 + i, agent_seat=i % 3, left='baseline', right='mad_scientist') for i in range(4)]
    root = Path(__file__).parents[2]
    results = []
    for workers in (1, 2):
        output = tmp_path / f'games-{workers}.jsonl'
        summary = evaluate_model(model, 'cpu', specs, root, tmp_path / f'io-{workers}', output,
                                 workers=workers, detailed=True, action_mode=action_mode,
                                 action_constraint=constraint)
        assert summary['completed'] == len(specs) and summary['failed'] == summary['truncated'] == 0
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        for row in rows:
            row.pop('duration_seconds')
        results.append(rows)
    assert results[0] == results[1]
    assert model.training
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    assert not list(tmp_path.glob('evaluation-parts-*'))


@pytest.mark.integration
def test_parallel_evaluation_retains_finished_games_when_another_worker_raises(require_engine, tmp_path):
    from sevenwonders_rl.evaluation.runner import evaluate_model
    specs = [dict(engine_seed=501, left='baseline', right='baseline'),
             dict(engine_seed=502, left={'checkpoint': str(tmp_path / 'missing.pt'), 'sha256': 'missing'},
                  right='baseline')]
    output = tmp_path / 'partial.jsonl'
    with pytest.raises(FileNotFoundError):
        evaluate_model(ActorCritic(102, 99, hidden=8), 'cpu', specs, Path(__file__).parents[2],
                       tmp_path / 'io', output, workers=2)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['game_id'] == 0 and rows[0]['status'] == 'completed'
    assert not list(tmp_path.glob('evaluation-parts-*'))
