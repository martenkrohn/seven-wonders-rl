"""Neural reference plans and paired, seed-block learning diagnostics."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from sevenwonders_rl.utils.checkpoints import file_hash
from sevenwonders_rl.utils.seeding import derive_seed


def reference(path):
    path = Path(path).resolve()
    return {"checkpoint": str(path), "sha256": file_hash(path)}


def rolling_references(directory, current_step):
    """Latest strictly older trained snapshot plus a roughly half-age reference.

    Never evaluate against the current learner or the untrained step-zero model.
    One trained snapshot is enough; no eligible snapshot means defer evaluation.
    """
    snapshots = []
    for path in Path(directory).glob('checkpoint_*.pt'):
        suffix = path.stem.removeprefix('checkpoint_')
        if suffix.isdigit() and 0 < int(suffix) < current_step:
            snapshots.append((int(suffix), path))
    snapshots.sort()
    if not snapshots:
        return []
    latest = snapshots[-1]
    others = snapshots[:-1]
    selected = [latest]
    if others:
        selected.append(min(others, key=lambda pair: (abs(pair[0] - current_step / 2), pair[0])))
    return [{**reference(path), 'step': step} for step, path in selected]


def plan_identity(plan, action_mode):
    import hashlib
    return hashlib.sha256(json.dumps({'plan': plan, 'action_mode': action_mode},
                                    sort_keys=True).encode()).hexdigest()


def evaluation_plan(references, seed, blocks, *, all_pairs=False):
    """Each seed block covers all candidate seats and both mixed-opponent orders."""
    pairs = ([(left, right) for left in references for right in references] if all_pairs else
             [(ref, ref) for ref in references])
    if not all_pairs and len(references) >= 2:
        pairs.extend(((references[0], references[-1]), (references[-1], references[0])))
    return [{"left": left, "right": right, "agent_seat": seat,
             "engine_seed": derive_seed(seed, "evaluation", block)}
            for block in range(blocks) for left, right in pairs for seat in range(3)]


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def block_significance(values):
    """Two-sided paired sign-flip test of mean seed-block differences.

    The null assumes exchangeable model labels within independent seed blocks
    (symmetric block differences). Seat/matchup rotations are never resampled
    independently. Exact up to 16 blocks; otherwise 65,535 deterministic draws,
    with the conservative plus-one correction. Exploratory, unadjusted p-values.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        return None
    n = len(values)
    exact = n <= 16
    count = 2 ** n if exact else 65535
    extreme = np.zeros(values.shape[1], dtype=np.int64)
    observed = np.abs(values.mean(axis=0))
    tolerance = np.maximum(1., observed) * 1e-12
    rng = np.random.default_rng(1729)
    for start in range(0, count, 1024):
        size = min(1024, count - start)
        if exact:
            bits = (np.arange(start, start + size, dtype=np.uint64)[:, None]
                    >> np.arange(n, dtype=np.uint64)) & 1
        else:
            bits = rng.integers(0, 2, size=(size, n))
        signs = bits.astype(float) * 2 - 1
        permuted = np.abs(np.einsum('bn,nm->bm', signs, values) / n)
        extreme += (permuted >= observed - tolerance).sum(axis=0)
    p = extreme / count if exact else (extreme + 1) / (count + 1)
    return dict(p_values=p.tolist(), method='paired_seed_block_sign_flip',
                exact=exact, permutations=count, adjusted=False)


def paired_progress(candidate_path, initial_path):
    """Bootstrap whole engine-seed blocks, never count seat rotations as independent."""
    candidate, initial = read_records(candidate_path), read_records(initial_path)
    def key(record):
        return (record["engine_seed"], record["agent_seat"],
                json.dumps(record["opponents"], sort_keys=True))
    baseline = {key(record): record for record in initial}
    if (len(baseline) != len(initial) or len({key(r) for r in candidate}) != len(candidate)
            or len(candidate) != len(initial) or {key(r) for r in candidate} != set(baseline)):
        raise ValueError("Paired evaluation plans differ")
    if not candidate or any(r["status"] != "completed" for r in candidate + initial):
        return {"paired_valid": False, "seed_blocks": 0}
    grouped = defaultdict(list)
    for record in candidate:
        previous = baseline[key(record)]
        grouped[record["engine_seed"]].append([
            record["vp_diff"] - previous["vp_diff"],
            record["win_share"] - previous["win_share"]])
    values = np.array([np.mean(rows, axis=0) for rows in grouped.values()])
    intervals = [None, None]
    if len(values) >= 2:
        rng = np.random.default_rng(0)  # local diagnostic RNG; does not affect training
        samples = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
        intervals = np.quantile(samples, [.025, .975], axis=0).T.tolist()
    test = block_significance(values)
    return {"paired_valid": True, "seed_blocks": len(values),
            "vp_diff_gain": float(values[:, 0].mean()), "vp_diff_gain_ci95": intervals[0],
            "win_share_gain": float(values[:, 1].mean()), "win_share_gain_ci95": intervals[1],
            "vp_diff_gain_p_value": test['p_values'][0] if test else None,
            "win_share_gain_p_value": test['p_values'][1] if test else None,
            "significance_test": {k: v for k, v in test.items() if k != 'p_values'} if test else None}
