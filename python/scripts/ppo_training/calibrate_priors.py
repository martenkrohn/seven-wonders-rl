"""Measure fixed-prior strength on states collected without priors or learning.

All strengths see exactly the same states. Scores from the collection games are
not used to choose a strength; this measures policy perturbation, not performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from sevenwonders_rl.envs import SevenWondersEnv, SeatPolicy
from sevenwonders_rl.evaluation.runner import run_game
from sevenwonders_rl.evaluation.selfplay import evaluation_plan
from sevenwonders_rl.features.action_feature_schema import IDX
from sevenwonders_rl.policies.ppo_checkpoint_policy import PPOCheckpointPolicy
from sevenwonders_rl.utils.checkpoints import file_hash, load_model
from sevenwonders_rl.utils.ppo_utils import flatten_global, masked_logits, select_action_from_obs
from sevenwonders_rl.utils.runtime import atomic_json, exchange_io_root, seat_order
from sevenwonders_rl.utils.seeding import derive_seed


def distribution_metrics(base_logits, prior_logits, masks, strength):
    """Prior logits [heads, states, actions]; masks are [states, actions]."""
    p = masked_logits(base_logits, masks).softmax(-1)
    q = masked_logits(base_logits.unsqueeze(0) + strength * prior_logits, masks.unsqueeze(0)).softmax(-1)
    p_log, q_log = p.clamp_min(1e-30).log(), q.clamp_min(1e-30).log()
    average = q.mean(0).clamp_min(1e-30)
    tv = .5 * (q - p).abs().sum(-1)
    flips = (q.argmax(-1) != p.argmax(-1)).float()
    return dict(
        strength=float(strength),
        mean_total_variation=float(tv.mean()),
        p90_total_variation=float(torch.quantile(tv.flatten(), .9)),
        greedy_change_fraction=float(flips.mean()),
        kl_from_baseline=float((p * (p_log - q_log)).sum(-1).mean()),
        mean_entropy=float(-(q * q_log).sum(-1).mean()),
        mean_max_probability=float(q.max(-1).values.mean()),
        head_js_divergence=float((q * (q_log - average.log())).sum(-1).mean()),
        per_head=[dict(total_variation=float(tv[h].mean()), greedy_change_fraction=float(flips[h].mean()))
                  for h in range(len(q))])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('runs/multihead-v1-seed31'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=91001)
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--strengths', type=float, nargs='+', default=[0, .5, 1, 2, 4, 8, 16, 32])
    args = parser.parse_args(argv)
    if args.seed < 0 or args.blocks < 1 or any(not np.isfinite(s) or s < 0 for s in args.strengths):
        parser.error('Use nonnegative seeds/strengths and positive blocks')
    torch.set_num_threads(1)
    config = json.loads((args.run_dir / 'config.json').read_text())
    model = load_model(args.run_dir / 'checkpoint_0.pt')
    if model.num_heads < 2 or model.prior_strength != 0:
        parser.error('Calibration requires a multi-head starting checkpoint with priors disabled')
    refs = config['frozen_opponent_references']
    for ref in refs:
        if file_hash(ref['checkpoint']) != ref['sha256']:
            parser.error('A frozen opponent changed')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('Calibration output must be empty')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    io_root = exchange_io_root(None, args.out_dir / 'exchange', 'prior-calibration')
    cache = {ref['checkpoint']: PPOCheckpointPolicy(ref['checkpoint']) for ref in refs}
    plan = evaluation_plan(refs, args.seed, args.blocks, all_pairs=True)
    states, records = [], []
    try:
        for game_id, spec in enumerate(plan):
            agent, left, right = seat_order(spec['agent_seat'])
            opponents = [SeatPolicy('random') for _ in range(3)]
            for seat, side in ((left, 'left'), (right, 'right')):
                opponents[seat] = SeatPolicy('ppo_snapshot', policy=cache[spec[side]['checkpoint']])
            env = SevenWondersEnv(repo_root=str(root), agent_seat=agent,
                io_dir=str(io_root / f'game_{game_id:04d}' / 'io'),
                opponent_policies=opponents, invalid_action_fallback='raise', batch_opponents=True)
            rng = np.random.default_rng(derive_seed(spec['engine_seed'], 'policy', agent))
            def observe(obs, action, info):
                ack = info.get('status', {}).get('last_actions', {}).get(str(agent), {})
                if ack.get('status') == 'applied':
                    states.append((flatten_global(obs), obs['action_feats'].copy(),
                                   obs['action_mask'].copy(), int(obs['game_meta'][0])))
            record, _ = run_game(env, lambda obs: select_action_from_obs(
                obs, model, 'cpu', rng=rng, head_id=0),
                dict(game_id=game_id, engine_seed=spec['engine_seed'], agent_seat=agent,
                     opponents={s:spec[s] for s in ('left','right')}), on_step=observe)
            records.append(record)
            if record['status'] != 'completed':
                raise RuntimeError(f'Calibration game failed: {record}')
            shutil.rmtree(env.io_dir.parent)
            if (game_id+1) % 24 == 0:
                print(f'Collected {game_id+1}/{len(plan)} games', flush=True)
    finally:
        shutil.rmtree(io_root)
        with (args.out_dir / 'games.jsonl').open('x') as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
    global_obs = torch.from_numpy(np.stack([s[0] for s in states]))
    features = torch.from_numpy(np.stack([s[1] for s in states]))
    masks = torch.from_numpy(np.stack([s[2] for s in states]))
    ages = torch.tensor([s[3] for s in states])
    logits, priors = [], []
    with torch.no_grad():
        for start in range(0, len(states), 128):
            g, f = global_obs[start:start+128], features[start:start+128]
            base, _ = model.forward_batch(g, f, [0]*len(g))
            raw = torch.cat([f, g.unsqueeze(1).expand(-1, f.size(1), -1)], -1)
            logits.append(base)
            priors.append(torch.stack([prior(raw).squeeze(-1) for prior in model.priors]))
        logits, priors = torch.cat(logits), torch.cat(priors, dim=1)
        p = masked_logits(logits, masks).softmax(-1)
        builds = ((features[:,:,IDX['action_type_0']] > 0) | (features[:,:,IDX['action_type_2']] > 0))
        grey = builds & (features[:,:,IDX['card_color_grey']] > 0) & (masks > 0)
        early_grey = (ages <= 2) & grey.any(-1)
        measurements = []
        for strength in args.strengths:
            measurement = distribution_metrics(logits, priors, masks, strength)
            measurement['ages'] = {str(a): distribution_metrics(logits[ages==a], priors[:,ages==a], masks[ages==a], strength)
                                   for a in (1,2,3)}
            q = masked_logits(logits.unsqueeze(0)+strength*priors, masks.unsqueeze(0)).softmax(-1)
            measurement['mean_grey_probability_when_available_early'] = (float(
                (q[:,early_grey] * grey[early_grey]).sum(-1).mean()) if early_grey.any() else None)
            measurement['per_head_grey_probability_when_available_early'] = ((
                q[:,early_grey] * grey[early_grey]).sum(-1).mean(-1).tolist()
                if early_grey.any() else [None]*model.num_heads)
            measurements.append(measurement)
    # Save the exact states, so strength sweeps can be repeated without new games.
    dataset = args.out_dir / 'states.npz'
    np.savez_compressed(dataset, global_obs=global_obs.numpy(), features=features.numpy(),
                        masks=masks.numpy(), ages=ages.numpy())
    report = dict(seed=args.seed, blocks=args.blocks, games=len(records), states=len(states),
        collection_policy='Starting standard model; stochastic; no learning or priors',
        initialization=config['initialization'], model_seed=config['seed'],
        initial_checkpoint_sha256=file_hash(args.run_dir/'checkpoint_0.pt'),
        dataset_sha256=file_hash(dataset), num_heads=model.num_heads, references=refs,
        early_states_with_legal_grey=int(early_grey.sum()),
        baseline_mean_max_probability=float(p.max(-1).values.mean()),
        measurements=measurements,
        limitations='States come from the baseline policy. This does not estimate returns under priors '
                    'or measure generalization, learned diversity or success of grey-resource investment.')
    atomic_json(args.out_dir/'summary.json', report)
    for measurement in measurements:
        print({k:measurement[k] for k in ('strength','mean_total_variation','greedy_change_fraction',
              'head_js_divergence','mean_grey_probability_when_available_early')}, flush=True)
    return report


if __name__ == '__main__':
    main()
