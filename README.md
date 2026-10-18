# Seven Wonders Reinforcement Learning

A research project combining a C++ simulator for three-player games of 7 Wonders with a Python Gymnasium environment and PyTorch PPO agents. The Python layer provides legal-action masks, structured action features, heuristic opponents and snapshot self-play.

**Status:** working simulator/environment tests; experimental training. The trainer has a confirmed multi-environment advantage-calculation defect, so existing learning curves are not reliable evidence of algorithm quality. See [HANDOVER.md](HANDOVER.md) for the German restart guide and priorities.

## Structure

| Path | Purpose |
| --- | --- |
| `main.cpp`, `src/`, `include/` | Simulator and bundled JSON headers |
| `python/src/sevenwonders_rl/` | Environment, features, policies, PPO model and opponent pool |
| `python/src/sevenwonders_rl/data/` | Card, wonder and mechanics JSON databases |
| `python/scripts/ppo_training/` | Main training and detailed evaluation commands |
| `python/scripts/` | Data extraction, diagnostics, tournaments and earlier training scripts |
| `python/tests/`, `tests/` | Python contracts/integration tests and C++ regression checks |
| `.github/workflows/` | Automated checks |

Generated runs, checkpoints, runtime exchange files, personal notes, old GUI assets and local research archives are excluded from the GitHub snapshot. They remain available in the original working directory.

## Setup and checks

Use Ubuntu/Linux, Python 3.12, a C++17 compiler and CMake. Run from the repository root:

```sh
python3 -m venv python/.venv
. python/.venv/bin/activate
python -m pip install -e './python[dev]'
cmake -S . -B build
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
cd python
python -m pytest -m 'unit or component'
python -m pytest -m integration
```

Integration tests need `build/7Wonders`. Unit/component tests use frozen states. The published fixture pool is a deterministic subset of the complete local capture; the original pool remains local. Tests do not establish complete board-game rule compliance or successful PPO learning.

## Training and evaluation

From `python/`, with the environment active:

```sh
python scripts/ppo_training/train_unified.py --help
python scripts/ppo_training/eval_ppo_descriptives.py --help
```

The trainer supports `--mode fixed` and `--mode selfplay`. Do not start a long run until the GAE issue in HANDOVER.md is fixed. PPO buffers already live in RAM; the simulator still exchanges JSON/actions through files. `--io-root` can place training exchange directories on a RAM filesystem, while logs/checkpoints remain persistent. This does not change the transport protocol or automatically cover evaluation IO.

The detailed evaluator accepts a checkpoint or run directory and opponent mixtures such as `baseline:0.5,science:0.3,military:0.2`. Existing checkpoints may use an older feature schema and must be checked before evaluation. The unified trainer has no resume flag; an old note claiming exact resume support was incorrect.

## Origin and publication scope

The simulator is derived from [DMAG's 7Wonders](https://github.com/dmag-ufsm/7Wonders). The local pre-cleanup base is commit `020869a`; its parent `a0bc333` is the historical license update. The RL package, experiments and local engine changes are extensions in this working tree, not an original implementation of the simulator. Historical notes show use of coding assistants; the precise authorship/contribution statement still needs the maintainer's review.

The inherited [LICENSE](LICENSE) contains GPLv3 and is preserved unchanged. The upstream README says MIT, which conflicts with its license file; this repository does not replace that license or resolve the discrepancy. The bundled nlohmann/json headers retain their original notices. See [NOTICE](NOTICE).

The first GitHub repository is private. Before changing visibility, review simulator/data provenance, the contribution statement and the selected source snapshot. Upstream GUI artwork and reference documents are omitted from the upload. No benchmark or game-publisher affiliation is claimed.
