"""Public observability, neutral migration, mixed populations and durable reports."""
import copy
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from sevenwonders_rl.features.obs_builder import build_obs
from sevenwonders_rl.features.public_state import (
    PLAYER_NAMES, STAGE_NAMES, PUBLIC_GLOBAL_DIM, PUBLIC_ACTION_DIM, stage_vector)
from sevenwonders_rl.data.wonders import wonder_by_id, load_wonder_db
from sevenwonders_rl.utils.ppo_model import MultiHeadActorCritic, extend_public_inputs
from sevenwonders_rl.utils.ppo_utils import flatten_global, action_features
from sevenwonders_rl.utils.checkpoints import save_checkpoint, load_model
from sevenwonders_rl.selfplay.opponent_pool import OpponentPool, MixedOpponentPool
from sevenwonders_rl.evaluation.milestones import milestone_steps, fixed_panel, MilestoneEvaluations
from sevenwonders_rl.utils.phase_resume import runtime_snapshot, restore_runtime


@pytest.mark.component
def test_wonder_database_matches_current_cpp_costs_rewards_and_abilities():
    from sevenwonders_rl.features.public_state import ABILITIES
    root = Path(__file__).resolve().parents[2]
    source = (root / 'src/wonder.cpp').read_text()
    header = (root / 'include/wonder.h').read_text().split('namespace EFFECT')[0]
    header = re.sub(r'//[^\n]*', '', header)
    ids = re.search(r'enum\s*\{(.*?)\}', header, re.S).group(1)
    names = [x.strip() for x in ids.split(',') if x.strip()]
    aliases = {'choose_raw': 'choose_raw_nontradable', 'choose_manuf': 'choose_manuf_nontradable',
               'add_vp_discard': 'discard_build_for_free'}
    assignments = r'this->cost\[RESOURCE::(\w+)\]\s*=\s*(\d+)'
    for wid, name in enumerate(names):
        board = wonder_by_id(wid)
        assert board['id_name'] == name
        cls = name.capitalize()
        constructor = re.search(rf'{cls}::{cls}\(\)\s*\{{(.*?)\n\}}', source, re.S).group(1)
        assert re.search(r'production\s*=\s*RESOURCE::(\w+)', constructor).group(1) == board['starting_resource']
        cost = {r: int(n) for r, n in re.findall(assignments, constructor)}
        effects = re.findall(r'EFFECT::(\w+)', constructor)
        block = source.split(f'bool {cls}::AddStage', 1)[1].split(f'bool {cls}::CanAddStage', 1)[0]
        cases = re.findall(r'case (\d+):(.*?)break;', block, re.S)
        assert len(cases) == len(board['stages']) == len(effects)
        for (index, code), effect, stage in zip(cases, effects, board['stages']):
            resources = re.findall(r'ProduceResource\(RESOURCE::(\w+)', code)
            assert stage['cost'] == {r: cost[r] for r in resources}, (name, index)
            vp = re.search(r'wonder_points\s*\+=\s*(\d+)', code)
            assert stage['vp'] == (int(vp.group(1)) if vp else 0), (name, index)
            vector = stage_vector(stage)
            for resource in ('coins', 'shields'):
                additions = re.findall(rf'AddResource\(RESOURCE::{resource},\s*(\d+)\)', code)
                assert vector[STAGE_NAMES.index(resource)] == sum(map(int, additions)), (name, index)
            ability = aliases.get(effect, effect)
            assert {a for a in ABILITIES if vector[STAGE_NAMES.index(a)]} == ({ability} if ability in ABILITIES else set())
            cost.update({r: int(n) for r, n in re.findall(assignments, code)})
    assert len(names) == len(load_wonder_db()) == 14


@pytest.mark.component
def test_public_state_seat_rotation_and_private_hand_isolation(status_age1_turn0):
    status = copy.deepcopy(status_age1_turn0)
    for seat in range(3):
        status['players'][str(seat)]['wonder_id'] = seat  # ID zero is valid.
        status['players'][str(seat)]['wonder_stage'] = 0
    observations = [build_obs(status, i, public_information=True) for i in range(3)]
    for seat, obs in enumerate(observations):
        rows = obs['public_state'].reshape(3, -1)
        for row, player in zip(rows, [seat, (seat - 1) % 3, (seat + 1) % 3]):
            assert row[PLAYER_NAMES.index('wonder_' + str(player))] == 1
        old = build_obs(status, seat)
        np.testing.assert_array_equal(flatten_global(obs), flatten_global(old))
        np.testing.assert_array_equal(action_features(obs), old['action_feats'])
        assert flatten_global(obs, public_information=True).shape == (102 + PUBLIC_GLOBAL_DIM,)
    changed = copy.deepcopy(status)
    for seat in ('1', '2'):
        changed['players'][seat]['cards_hand'] = ['private, deliberately invalid']
        changed['players'][seat]['cards_hand_id'] = [-999]
        changed['players'][seat]['cards_playable'] = []
        changed['players'][seat]['can_build_wonder'] = False
    other = build_obs(changed, 0, public_information=True)
    for key in observations[0]:
        np.testing.assert_array_equal(observations[0][key], other[key])


@pytest.mark.component
def test_wonder_actions_describe_the_next_stage(status_can_build_wonder):
    status = copy.deepcopy(status_can_build_wonder)
    player = status['players']['0']
    player.update(wonder_id=0, wonder_stage=0, can_build_wonder=True)
    obs = build_obs(status, 0, public_information=True)
    row = obs['wonder_action_feats'][7]
    assert row[0] == 1
    assert row[1 + STAGE_NAMES.index('cost_stone')] == 2
    assert row[1 + STAGE_NAMES.index('vp')] == 3
    assert not obs['wonder_action_feats'][:7].any()
    # Halikarnassos B bundles VP and discard access in its first two stages.
    for stage in wonder_by_id(13)['stages']:
        assert stage_vector(stage)[STAGE_NAMES.index('discard_build_for_free')] == 1


@pytest.mark.component
def test_neutral_migration_preserves_outputs_priors_and_learns_new_columns(tmp_path):
    torch.manual_seed(4)
    source = MultiHeadActorCritic(102, 99, hidden=16, num_heads=4, prior_strength=32)
    before_rng = torch.get_rng_state().clone()
    extended = extend_public_inputs(source)
    assert torch.equal(before_rng, torch.get_rng_state())
    g, a = torch.randn(4, 102), torch.randn(4, 28, 99)
    new_g = torch.cat((g, torch.randn(4, PUBLIC_GLOBAL_DIM)), -1)
    new_a = torch.cat((a, torch.randn(4, 28, PUBLIC_ACTION_DIM)), -1)
    old_logits, old_values = source.forward_batch(g, a, [0, 1, 2, 3])
    logits, values = extended.forward_batch(new_g, new_a, [0, 1, 2, 3])
    torch.testing.assert_close(logits, old_logits, atol=5e-6, rtol=1e-6)
    torch.testing.assert_close(values, old_values, atol=1e-6, rtol=1e-6)
    for key, tensor in source.priors.state_dict().items():
        assert torch.equal(tensor, extended.priors.state_dict()[key])
    (values.square().mean() - logits.log_softmax(-1)[:, 0].mean()).backward()
    assert extended.global_mlp[0].weight.grad[:, 102:].abs().sum() > 0
    assert extended.action_mlp[0].weight.grad[:, 99:].abs().sum() > 0
    path = tmp_path / 'public.pt'
    save_checkpoint(path, extended, None, 0,
        dict(global_dim=102 + PUBLIC_GLOBAL_DIM, action_feat_dim=99 + PUBLIC_ACTION_DIM, hidden=16),
        {'feature_semantics_version': 3})
    restored = load_model(path)
    assert restored.public_information and restored.prior_global_dim == 102 and restored.prior_action_dim == 99
    torch.testing.assert_close(restored.forward_batch(new_g, new_a, [0, 1, 2, 3]), (logits, values))
    payload = torch.load(path, weights_only=True)
    payload['meta']['public_state_schema_hash'] = 'changed'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='public observation'):
        load_model(path)


@pytest.mark.unit
def test_mixed_league_fixed_weights_lag_archive_and_fallback(tmp_path):
    league = OpponentPool(str(tmp_path), min_step_lag=1500, auto_refresh=False)
    groups = [dict(name='anchors', weight=.20, paths=['v1', 'v2']),
              dict(name='standard', weight=.05, paths=['baseline', 'credit']),
              dict(name='science', weight=.05, paths=['science']),
              dict(name='military', weight=.05, paths=['military'])]
    pool = MixedOpponentPool(league, groups)
    rng = np.random.default_rng(1)
    fallback = [pool.sample_entry(rng, 0) for _ in range(100)]
    assert all(path in ('v1', 'v2') for path, m in fallback if m['fallback_to_anchors'])
    for step in range(0, 13000, 1000):
        path = tmp_path / f'checkpoint_{step}.pt'
        path.touch()
        pool.register(path)
    members = pool.dynamic_members(12000)
    assert {Path(p).stem for p in members['recent']} == {f'checkpoint_{s}' for s in range(5000, 11000, 1000)}
    assert not set(members['recent']).intersection(members['archive'])
    assert len(members['archive']) == 4
    from collections import Counter
    counts = Counter(pool.sample_entry(rng, 12000)[1]['group'] for _ in range(10000))
    for group, weight in [('recent', .5), ('archive', .15)] + [(g['name'], g['weight']) for g in groups]:
        assert abs(counts[group] / 10000 - weight) < .02


@pytest.mark.unit
def test_second_phase_archive_is_populated_from_start_and_remains_temporally_spread(tmp_path):
    historical = tuple(str(tmp_path / f'previous_{i}.pt') for i in range(6))
    league = OpponentPool(str(tmp_path), min_step_lag=100, auto_refresh=False)
    pool = MixedOpponentPool(league, [dict(name='anchors', weight=.2, paths=['end', 'quarter'])],
        recent_weight=.6, archive_weight=.2, historical_paths=historical)
    assert pool.dynamic_members(0)['archive'] == list(historical)
    rng = np.random.default_rng(3)
    entries = [pool.sample_entry(rng, 0) for _ in range(10000)]
    from collections import Counter
    counts = Counter(m['group'] for _, m in entries)
    for name, expected in [('recent', .6), ('archive', .2), ('anchors', .2)]:
        assert abs(counts[name] / 10000 - expected) < .02
    assert all(not m['fallback_to_anchors'] and path in historical
               for path, m in entries if m['group'] == 'archive')
    for step in range(100, 2000, 100):
        path = tmp_path / f'checkpoint_{step}.pt'
        path.touch()
        pool.register(path)
    members = pool.dynamic_members(2000)
    assert len(members['archive']) == 6
    assert members['archive'][0] == historical[0]
    assert members['archive'][-1].endswith('checkpoint_1300.pt')
    assert not set(members['archive']).intersection(members['recent'])


@pytest.mark.unit
def test_quarters_and_panel_are_fixed_and_all_seats_are_covered(tmp_path):
    assert milestone_steps(2097152, 8) == [524288, 1048576, 1572864, 2097152]
    with pytest.raises(ValueError):
        milestone_steps(100, 8)
    panel = tmp_path / 'panel.json'
    panel.write_text(json.dumps({'pairs': [['anchors:0', 'military:0']]}))
    refs = [dict(group=g, checkpoint=g, sha256=g) for g in ('anchors', 'military')]
    specs = fixed_panel(refs, panel, 119001, 8)
    assert len(specs) == 24
    assert len({s['engine_seed'] for s in specs}) == 8
    assert {s['agent_seat'] for s in specs} == {0, 1, 2}


@pytest.mark.unit
def test_resume_restores_rngs_and_moves_beyond_active_games():
    env, heads = [np.random.default_rng(1)], [np.random.default_rng(2)]
    mb, actions = np.random.default_rng(3), torch.Generator().manual_seed(4)
    state = copy.deepcopy(runtime_snapshot(10, [7], {0: {'engine_seed': 11}}, env, heads, mb, actions))
    expected = (env[0].integers(100000), heads[0].integers(100000), mb.integers(100000),
                torch.rand(3, generator=actions))
    episodes = [999]
    restore_runtime(state, episodes, env, heads, mb, actions)
    assert episodes == [8]
    assert (env[0].integers(100000), heads[0].integers(100000), mb.integers(100000)) == expected[:3]
    assert torch.equal(torch.rand(3, generator=actions), expected[3])


@pytest.mark.integration
@pytest.mark.parametrize('interrupt_final', [False, True])
def test_milestones_publish_before_completion_and_survive_resume(tmp_path, require_engine, monkeypatch, interrupt_final):
    from test_training_contracts import trainer_module
    from test_selfplay import checkpoint
    from sevenwonders_rl.utils.checkpoints import file_hash
    trainer = trainer_module()
    anchor, military = tmp_path / 'anchor.pt', tmp_path / 'military.pt'
    checkpoint(anchor)
    checkpoint(military)
    population, panel = tmp_path / 'population.json', tmp_path / 'panel.json'
    population.write_text(json.dumps({'groups': [
        dict(name='anchors', weight=.30, checkpoints=[str(anchor)]),
        dict(name='military', weight=.05, checkpoints=[str(military)])]}))
    panel.write_text(json.dumps({'pairs': [['anchors:0', 'military:0']]}))
    output = tmp_path / 'run'
    argv = ['--out-dir', str(output), '--init-checkpoint', str(anchor),
        '--total-timesteps', '1024', '--n-envs', '2', '--rollout-horizon', '64',
        '--batch-size', '128', '--minibatch-epochs', '1', '--num-heads', '4',
        '--public-information', '--opponent-population', str(population), '--mixed-league',
        '--milestone-evaluations', '--eval-panel', str(panel), '--eval-games-per-matchup', '1',
        '--snapshot-every', '256', '--pool-lag-steps', '128', '--save-every', '128']
    if interrupt_final:
        evaluation_population = tmp_path / 'evaluation_population.json'
        evaluation_population.write_text(population.read_text())
        population.write_text(json.dumps({'groups': [
            dict(name='anchors', weight=.2, checkpoints=[str(anchor)])]}))
        history = tmp_path / 'historical.pt'
        checkpoint(history)
        argv += ['--mixed-recent-weight', '.6', '--mixed-archive-weight', '.2',
                 '--historical-checkpoints', str(history),
                 '--evaluation-population', str(evaluation_population)]
    original = trainer.collect_rollout
    original_eval = trainer.MilestoneEvaluations.evaluate
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('deliberate test interruption after 25%')
        return original(*args, **kwargs)
    def interrupt_evaluation(self, model, step):
        if step == 1024:
            directory = self.output / 'evaluations' / 'step_0001024'
            directory.mkdir()
            (directory / 'candidate.jsonl').write_text('deliberately incomplete evaluation')
            raise RuntimeError('deliberate test interruption during final evaluation')
        return original_eval(self, model, step)
    with monkeypatch.context() as m:
        if interrupt_final:
            m.setattr(trainer.MilestoneEvaluations, 'evaluate', interrupt_evaluation)
        else:
            m.setattr(trainer, 'collect_rollout', interrupt)
        with pytest.raises(RuntimeError, match='deliberate'):
            trainer.main(argv)
    html = (output / 'dashboard.html').read_text()
    entries = json.loads(html.split('<script id="evaluation-data" type="application/json">')[1].split('</script>')[0])
    assert [e['report']['training_percent'] for e in entries] == ([75, 50, 25, 0] if interrupt_final else [25, 0])
    assert not (output / 'evaluations' / 'step_0001024' / 'summary.json').exists()
    initial_hash = file_hash(output / 'checkpoint_0.pt')
    saved = torch.load(output / 'latest.pt', weights_only=True)
    assert saved['timestep'] == (1024 if interrupt_final else 256) and saved['optimizer_state'] and saved['runtime_state']
    trainer.main(argv + ['--resume'])
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'completed' and result['timestep'] == 1024
    assert result['resumed_from_step'] == (1024 if interrupt_final else 256)
    assert file_hash(output / 'checkpoint_0.pt') == initial_hash
    html = (output / 'dashboard.html').read_text()
    entries = json.loads(html.split('<script id="evaluation-data" type="application/json">')[1].split('</script>')[0])
    assert [e['report']['training_percent'] for e in entries] == [100, 75, 50, 25, 0]
    assert len({e['report']['plan_id'] for e in entries}) == 1
    assert all(e['report']['models']['candidate']['completed'] == 3 for e in entries)
    updates = [json.loads(line)['timestep'] for line in (output / 'updates.jsonl').read_text().splitlines()]
    assert updates == list(range(128, 1025, 128))
    resume = json.loads((output / 'resume_events.jsonl').read_text().splitlines()[0])
    assert not resume['bit_exact'] and len(resume['abandoned_games']) == 2
    if interrupt_final:
        draws = [json.loads(line) for line in (output / 'opponents.jsonl').read_text().splitlines()]
        assert all(d['group'] in ('recent', 'archive', 'anchors') for r in draws for d in r['opponents'].values())
        assert all('evaluation_opponents' not in d['checkpoint'] for r in draws for d in r['opponents'].values())
        saved_plan = json.loads((output / 'evaluation_plan.json').read_text())['plan']
        assert all('evaluation_opponents' in s['left']['checkpoint'] and
                   s['right']['group'] == 'military' for s in saved_plan)
