#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Fail before training if the calibrated starting model or opponents changed.
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from sevenwonders_rl.utils.checkpoints import file_hash
from sevenwonders_rl.selfplay.opponent_pool import read_population

run = Path('runs/multihead-v1-seed31')
config = json.loads((run/'config.json').read_text())
calibration = json.loads(Path('runs/multihead-prior-calibration-20261026/summary.json').read_text())
selection = json.loads(Path('runs/multihead-prior-calibration-20261026/selection.json').read_text())
if selection['selected_strength'] != 32 or selection['calibration_summary_sha256'] != file_hash(
        'runs/multihead-prior-calibration-20261026/summary.json'):
    raise ValueError('Prior calibration or selected strength changed')
if config['seed'] != 31 or config['num_heads'] != 4 or config['prior_strength'] != 0:
    raise ValueError('First experiment configuration changed')
if calibration['initial_checkpoint_sha256'] != file_hash(run/'checkpoint_0.pt'):
    raise ValueError('Calibrated initial checkpoint changed')
if not any(m['strength'] == 32 for m in calibration['measurements']):
    raise ValueError('Strength 32 was not calibrated')
refs = config['frozen_opponent_references']
for ref in refs:
    if file_hash(ref['checkpoint']) != ref['sha256']:
        raise ValueError('A frozen opponent changed')
source = run/'frozen_opponents/opponent_000.pt'
if file_hash(source) != config['initialization']['sha256']:
    raise ValueError('Original standard starting model changed')
population = read_population('configs/multihead_v2_priors_opponents.json')
expected = sorted((r['checkpoint'],r['group'],r['sampling_weight']) for r in refs)
actual = sorted((p,g['name'],g['weight']/len(g['checkpoints']))
                for g in population for p in g['checkpoints'])
if actual != expected:
    raise ValueError('Opponent population differs from the first experiment')
print('Verified calibrated initialization and unchanged opponent population.', flush=True)
PY

.venv/bin/python scripts/ppo_training/run_multihead_experiment.py \
  --init-checkpoint runs/multihead-v1-seed31/frozen_opponents/opponent_000.pt \
  --population configs/multihead_v2_priors_opponents.json \
  --num-heads 4 --prior-strength 32 --seed 31 \
  --total-timesteps 262144 --blocks 8 --eval-seed 89001 --workers 2 \
  --out-dir runs/multihead-v2-priors-seed31

.venv/bin/python scripts/ppo_training/compare_multihead_runs.py \
  --previous-run runs/multihead-v1-seed31 \
  --current-run runs/multihead-v2-priors-seed31
