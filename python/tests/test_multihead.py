"""Episode identity, PPO likelihoods, frozen priors and population contracts."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from sevenwonders_rl.utils.ppo_model import ActorCritic, MultiHeadActorCritic
from sevenwonders_rl.utils.checkpoints import save_checkpoint, load_model, file_hash
from sevenwonders_rl.utils.ppo_utils import masked_logits
from sevenwonders_rl.utils.timing import TimingBuckets
from sevenwonders_rl.selfplay.opponent_pool import FrozenOpponentPool, read_population
from test_training_contracts import trainer_module
from test_selfplay import checkpoint
from test_collector import observation


@pytest.mark.component
def test_warm_start_all_heads_match_source_and_loading_preserves_rng(tmp_path):
    source = tmp_path / 'source.pt'
    expected = checkpoint(source)
    trainer = trainer_module()
    args = trainer.parse_args(['--no-observation-memory', '--init-checkpoint', str(source), '--num-heads', '4'])
    rng = torch.get_rng_state().clone()
    model, _ = trainer.initialize_model(args, 102, 99)
    assert torch.equal(rng, torch.get_rng_state())
    global_obs, features = torch.randn(5, 102), torch.randn(5, 28, 99)
    for h in range(4):
        torch.testing.assert_close(model.forward_batch(global_obs, features, [h]*5),
                                   expected.forward_batch(global_obs, features), rtol=0, atol=0)
    target = tmp_path / 'model.pt'
    save_checkpoint(target, model, None, 0, dict(global_dim=102, action_feat_dim=99, hidden=8),
                    {'feature_semantics_version': 3})
    before = torch.get_rng_state().clone()
    restored = load_model(target)
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(restored.forward_batch(global_obs, features, [0, 1, 2, 3, 0]),
                               model.forward_batch(global_obs, features, [0, 1, 2, 3, 0]))
    payload = torch.load(target, weights_only=True)
    payload['config']['num_heads'] = 3
    torch.save(payload, target)
    with pytest.raises(RuntimeError):
        load_model(target)


@pytest.mark.unit
@pytest.mark.parametrize('strength', [0.0, 0.7])
def test_only_selected_heads_learn_and_prior_function_remains_fixed(strength):
    model = MultiHeadActorCritic(3, 2, hidden=8, num_heads=4, prior_strength=strength)
    prior_before = copy.deepcopy(model.priors.state_dict())
    global_obs, features = torch.randn(5, 3), torch.randn(5, 4, 2)
    logits, values = model.forward_batch(global_obs, features, [1]*5)
    (-logits.log_softmax(-1)[:, 0].mean() + values.square().mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.policy_heads[1].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.value_heads[1].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.global_mlp.parameters())
    for h in (0, 2, 3):
        assert all(p.grad is None for p in model.policy_heads[h].parameters())
        assert all(p.grad is None for p in model.value_heads[h].parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.priors.parameters())
    torch.optim.Adam((p for p in model.parameters() if p.requires_grad)).step()
    for key, tensor in prior_before.items():
        assert torch.equal(tensor, model.priors.state_dict()[key])
    torch.testing.assert_close(model.forward_values(global_obs, [1]*5),
                               model.forward_batch(global_obs, features, [1]*5)[1])
    with pytest.raises(ValueError, match='explicit'):
        model.forward_batch(global_obs, features)


@pytest.mark.component
def test_collector_preserves_heads_across_batches_and_bootstraps_before_reset():
    class Model:
        num_heads = 4
        def forward_batch(self, obs, features, ids):
            return torch.zeros(len(obs), 28), self.forward_values(obs, ids)
        def forward_values(self, obs, ids):
            return obs[:, 0] + ids.float()*100
    class Env:
        io_dir = 'fake'
        def __init__(self, truncated=False):
            self.steps = 0
            self.truncated = truncated
        def send_action(self, action, **kwargs):
            self.steps += 1
            return 0
        def wait_next(self, previous):
            boundary = self.steps % 3 == 0
            return observation(10), 1., boundary and not self.truncated, boundary and self.truncated, {}
    ids = [1, 2]
    def reset(i, *args):
        ids[i] = 3
        return observation(1000)
    trainer = trainer_module()
    envs = [Env(), Env(truncated=True)]
    roll, obs = trainer.collect_rollout(Model(), envs, [observation(0)]*2, 2,
        torch.Generator().manual_seed(0), reset, TimingBuckets(), head_ids=ids)
    assert roll['head_ids'].tolist() == [[1, 2], [1, 2]]
    roll, _ = trainer.collect_rollout(Model(), envs, obs, 2,
        torch.Generator().manual_seed(0), reset, TimingBuckets(), head_ids=ids)
    assert roll['head_ids'].tolist() == [[1, 2], [3, 3]]
    assert roll['next_values'][0].tolist() == [0, 210]  # truncated game uses OLD critic
    assert roll['values'][1].tolist() == [1300, 1300]


@pytest.mark.component
def test_ppo_recomputes_likelihood_with_recorded_head():
    trainer = trainer_module()
    model = MultiHeadActorCritic(3, 2, hidden=8, num_heads=4, prior_strength=.5)
    global_obs, features = torch.randn(2, 4, 3), torch.randn(2, 4, 5, 2)
    ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    masks = torch.ones(2, 4, 5)
    with torch.no_grad():
        logits, values = model.forward_batch(global_obs.flatten(0,1), features.flatten(0,1), ids.flatten())
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        roll = dict(global_=global_obs, features=features, masks=masks, head_ids=ids,
            actions=actions.reshape(2,4), values=values.reshape(2,4),
            logprobs=dist.log_prob(actions).reshape(2,4), rewards=torch.randn(2,4),
            next_values=torch.zeros(2,4), terminated=torch.ones(2,4,dtype=torch.bool),
            truncated=torch.zeros(2,4,dtype=torch.bool))
        roll['global'] = roll.pop('global_')
    args = trainer.parse_args(['--num-heads', '4', '--batch-size', '8', '--minibatch-epochs', '1'])
    result = trainer.ppo_update(model, torch.optim.Adam(model.parameters(), lr=0), roll, args,
                                np.random.default_rng(0))
    assert abs(result['approx_kl']) < 1e-7
    assert result['head_transitions'] == [2,2,2,2]


@pytest.mark.component
def test_population_weights_are_per_group_not_checkpoint_count(tmp_path):
    paths = [tmp_path / f'{i}.pt' for i in range(4)]
    for p in paths:
        p.touch()
    spec = tmp_path / 'population.json'
    spec.write_text(json.dumps({'groups': [
        {'name':'standard', 'weight':2, 'checkpoints':list(map(str, paths[:3]))},
        {'name':'science', 'weight':1, 'checkpoints':[str(paths[3])]}]}))
    trainer = trainer_module()
    args = trainer.parse_args(['--opponent-population', str(spec)])
    assert len(args.frozen_opponents) == 4
    pool = FrozenOpponentPool(tuple(map(str, paths)), (2/3,2/3,2/3,1))
    rng = np.random.default_rng(0)
    proportion = sum(pool.sample_path(rng, 999) == str(paths[3]) for _ in range(10000))/10000
    assert abs(proportion - 1/3) < .02
    bad = {'groups':[{'name':'bad', 'weight':0, 'checkpoints':[str(paths[0])]}]}
    spec.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        read_population(spec)
    with pytest.raises(SystemExit):
        trainer.parse_args(['--num-heads', '1', '--prior-strength', '1'])


@pytest.mark.integration
def test_training_population_and_per_head_holdout(require_engine, tmp_path):
    source = tmp_path / 'standard.pt'
    checkpoint(source)
    other = MultiHeadActorCritic(102, 99, hidden=8, num_heads=4, prior_strength=.4)
    second = tmp_path / 'other.pt'
    save_checkpoint(second, other, None, 100, dict(global_dim=102, action_feat_dim=99, hidden=8),
                    {'feature_semantics_version':3})
    spec = tmp_path / 'population.json'
    spec.write_text(json.dumps({'groups':[
        {'name':'standard','weight':2,'checkpoints':[str(source)]},
        {'name':'other','weight':1,'checkpoints':[str(second)]}]}))
    root = Path(__file__).parents[1]
    output = tmp_path / 'train'
    command = [sys.executable, 'scripts/ppo_training/train_unified.py', '--num-heads', '4',
        '--prior-strength', '.4', '--init-checkpoint', str(source), '--opponent-population', str(spec),
        '--total-timesteps', '256', '--n-envs', '4', '--rollout-horizon', '16',
        '--snapshot-every', '64', '--save-every', '64', '--eval-every', '256',
        '--eval-games-per-matchup', '1', '--out-dir', str(output)]
    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True, timeout=90)
    config = json.loads((output/'config.json').read_text())
    assert config['reward_profile'] == 'standard'
    assert config['num_heads'] == 4 and config['prior_strength'] == .4
    episodes = [json.loads(s) for s in (output/'episodes.jsonl').read_text().splitlines()]
    assert episodes
    for episode in episodes:
        assert 0 <= episode['head_id'] < 4
        for opponent in episode['opponents'].values():
            assert opponent['group'] in ('standard', 'other')
            assert Path(opponent['checkpoint']).parent.name == 'frozen_opponents'
            if opponent['group'] == 'other':
                assert 0 <= opponent['head_id'] < 4
    initial, final = load_model(output/'checkpoint_0.pt'), load_model(output/'latest.pt')
    assert any(not torch.equal(initial.policy_heads[h][0].weight, final.policy_heads[h][0].weight)
               for h in range(4))
    for key, tensor in initial.priors.state_dict().items():
        assert torch.equal(tensor, final.priors.state_dict()[key])
    holdout = tmp_path / 'holdout'
    subprocess.run([sys.executable, 'scripts/eval_selfplay.py', '--run-dir', str(output),
        '--population-from-run', '--all-opponent-pairs', '--evaluate-heads', '--blocks', '1',
        '--workers', '2', '--out-dir', str(holdout)], cwd=root, check=True,
        capture_output=True, text=True, timeout=90)
    summary = json.loads((holdout/'summary.json').read_text())
    assert set(summary['models']) == {'initial','candidate','head_0','head_1','head_2','head_3'}
    assert len(json.loads((holdout/'exploration.json').read_text())) == 6
    for name in summary['models']:
        assert summary[name]['completed'] == 12
        rows = [json.loads(s) for s in (holdout/f'{name}.jsonl').read_text().splitlines()]
        if name.startswith('head_'):
            assert {r['head_id'] for r in rows} == {int(name[-1])}
    assert 'Head 0' in (holdout/'report.txt').read_text()
    # Re-run two games involving the multi-head opponent through serial inference.
    # Results must agree with the parallel, batched path, including head choices.
    from sevenwonders_rl.evaluation.runner import evaluate_model
    candidate_rows = [json.loads(s) for s in (holdout/'candidate.jsonl').read_text().splitlines()]
    chosen = candidate_rows[3:5]
    specs = [dict(engine_seed=r['engine_seed'], agent_seat=r['agent_seat'],
                  left=r['opponents']['left'], right=r['opponents']['right']) for r in chosen]
    destination = tmp_path/'serial.jsonl'
    evaluate_model(final, 'cpu', specs, root.parent, tmp_path/'serial-io', destination,
                   action_mode='stochastic', batch_opponents=False, detailed=True)
    serial_rows = [json.loads(s) for s in destination.read_text().splitlines()]
    for actual, expected in zip(serial_rows, chosen):
        for key in ('head_id', 'opponent_head_ids', 'totals', 'action_counts', 'actions'):
            assert actual[key] == expected[key]


@pytest.mark.component
def test_shared_opponent_heads_batch_and_serial_use_explicit_game_assignments(tmp_path, status_age1_turn0):
    from types import SimpleNamespace
    from sevenwonders_rl.envs import SeatPolicy
    from sevenwonders_rl.features.obs_builder import build_obs
    from sevenwonders_rl.policies.ppo_checkpoint_policy import PPOCheckpointPolicy
    from sevenwonders_rl.policies.opponent_batch import opponent_actions_batch
    model = MultiHeadActorCritic(102, 99, hidden=8, num_heads=4, prior_strength=.7)
    target = tmp_path/'multi.pt'
    save_checkpoint(target, model, None, 100, dict(global_dim=102, action_feat_dim=99, hidden=8),
                    {'feature_semantics_version':3})
    policy = PPOCheckpointPolicy(str(target))
    envs = [SimpleNamespace(batch_opponents=True, _opponent_sampler=None, agent_seat=0,
        _last_status=status_age1_turn0,
        opponent_policies=[SeatPolicy('ppo_snapshot', policy=policy)]*3,
        _opponent_head_ids={1:i, 2:3-i},
        _seat_rngs=[np.random.default_rng(i*10+s) for s in range(3)]) for i in range(2)]
    expected = []
    for i, env in enumerate(envs):
        expected.append({s: policy.select_action(build_obs(status_age1_turn0, s),
            np.random.default_rng(i*10+s), head_id=env._opponent_head_ids[s]) for s in (1,2)})
    assert opponent_actions_batch(envs) == expected
    with pytest.raises(ValueError, match='explicit'):
        policy.select_action(build_obs(status_age1_turn0, 1), np.random.default_rng(0))
