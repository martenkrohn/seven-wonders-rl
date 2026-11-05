"""Calibration and offline comparisons must measure the intended policies."""
import importlib.util
from pathlib import Path

import pytest
import torch


def script(name):
    path = Path(__file__).parents[1]/'scripts/ppo_training'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_calibration_zero_prior_and_illegal_actions():
    metrics = script('calibrate_priors').distribution_metrics
    logits = torch.tensor([[0.,0.,1000.],[2.,0.,1000.]])
    prior = torch.tensor([[[1.,-1.,10000.],[1.,-1.,10000.]],
                          [[-1.,1.,-10000.],[-1.,1.,-10000.]]])
    masks = torch.tensor([[1,1,0],[1,1,0]])
    baseline = metrics(logits,prior,masks,0)
    assert baseline['mean_total_variation'] == 0
    assert baseline['kl_from_baseline'] == 0
    assert baseline['greedy_change_fraction'] == 0
    measurement = metrics(logits,prior,masks,1)
    p = torch.softmax(logits[:,:2],-1)
    q = torch.softmax(logits[:,:2]+prior[:,:,:2],-1)
    expected = .5*(q-p).abs().sum(-1).mean()
    assert measurement['mean_total_variation'] == pytest.approx(float(expected))
    assert measurement['head_js_divergence'] > 0
    assert 0 < measurement['greedy_change_fraction'] < 1


@pytest.mark.unit
def test_cross_run_comparison_matches_hashes_and_rejects_changed_plans():
    paired = script('compare_multihead_runs').paired
    def row(seed, path, score):
        return dict(engine_seed=seed,agent_seat=0,action_selection_mode='stochastic',
            opponents={'left':dict(checkpoint=path,sha256='a'),
                       'right':dict(checkpoint=path,sha256='b')},
            status='completed',vp_diff=score,win_share=float(score>0))
    before = [row(1,'old.pt',-1),row(2,'old.pt',-2)]
    after = [row(2,'new.pt',1),row(1,'new.pt',2)]
    result = paired(after,before)
    assert result['margin_gain'] == 3
    assert result['win_share_gain'] == 1
    assert result['margin_ci95'] == [3,3]
    after[0]['opponents']['right']['sha256'] = 'different'
    with pytest.raises(ValueError,match='plans differ'):
        paired(after,before)
