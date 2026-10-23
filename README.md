# Seven Wonders RL

A work-in-progress reinforcement-learning project for three-player **7 Wonders**. It adds a Python Gymnasium environment and PyTorch PPO agents to an existing C++ game simulator.

The simulator comes from [DMAG/UFSM's 7Wonders](https://github.com/dmag-ufsm/7Wonders). This project adds the RL environment, observation and action features, heuristic opponents, training/evaluation tools and snapshot self-play. It also includes a few targeted simulator fixes; it does not implement the original game engine from scratch.

**Status:** neural-only self-play, corrected PPO/GAE and seeded evaluation are implemented. Two short training runs show learning on separate test seeds against initial and trained intermediate opponents; this is initial proof-of-concept evidence, not a general playing-strength benchmark. RAM exchange and grouped opponent inference are enabled for self-play. See the [experiment guide](docs/SELFPLAY_POC.md), [learning results](docs/SELFPLAY_POC_RESULTS_2026-10-23.md), [validation record](docs/VALIDATION.md) and [remaining limitations](docs/KNOWN_ISSUES.md).

## How it works

```mermaid
flowchart LR
    A[C++ simulator] -->|JSON game state| B[Gymnasium environment]
    B --> C[Observations and legal-action mask]
    C --> D[PPO policy or heuristic opponent]
    D -->|Encoded actions| B
    B -->|Player JSON files and ready signal| A
```

Python starts the simulator, reads its state and submits actions for all three seats. The learner controls one seat; the other two use heuristic policies or frozen policy snapshots. Communication is currently file-based: `game_status.json`, `player_1.json` through `player_3.json`, and `ready.txt`.

The environment exposes 28 action slots: four action types for each of seven hand positions. These cover building a structure, building a wonder stage, using a free-build action and discarding. Masks restrict the available choices. Observation builders rotate player information relative to the learner and provide global state and per-action features. The terminal reward compares the learner's victory points with the strongest opponent, scaled by the training configuration.

## What is inherited and what is added

| Component | Origin |
| --- | --- |
| Core C++ simulator and file protocol | DMAG/UFSM's upstream project |
| Most engine source and bundled JSON headers | Preserved from upstream |
| Engine corrections and protocol v2 | Selectable production, trading, Gizah B, final status, seeded RNG and atomic publication |
| Gymnasium wrapper, masks and feature extraction | RL extension in this project |
| PyTorch actor-critic, PPO scripts, opponent policies and self-play pool | RL extension in this project |
| Python tests, C++ regression checks and CI setup | Downstream development and validation |

The [origin record](docs/ORIGIN.md) pins the upstream commit. The [simulator change log](docs/SIMULATOR_CHANGES.md) documents each C++ correction, with an exact patch and checked file inventory. This distinguishes inherited implementation from downstream work; it does not claim that every line of the extension was written manually. Development has used coding assistants.

## Setup

The supported development environment is **Linux/Ubuntu, including WSL**. Use Python 3.12, a C++17 compiler and CMake. From the repository root:

```sh
python3.12 -m venv python/.venv
source python/.venv/bin/activate
python -m pip install -e './python[dev]'
cmake -S . -B build
cmake --build build --parallel 2
```

The package declares Python >=3.10; Python 3.12 is the tested baseline. Dependencies include Gymnasium, NumPy and PyTorch. Versions are not locked, so the package declaration is not a fully pinned reproduction environment.

## Run the checks

From the repository root, with the environment active:

```sh
ctest --test-dir build --output-on-failure
cd python
python -m pytest -m 'unit or component'
python -m pytest -m integration
```

Unit/component tests use recorded game-state fixtures. Integration tests require `build/7Wonders`. These checks cover the implemented interfaces and selected rule regressions; they do not establish complete board-game rule correctness or successful learning. See the [validation record](docs/VALIDATION.md) for the checks performed during this cleanup and the [architecture notes](docs/ARCHITECTURE.md) for the environment and training interfaces.

## Training and evaluation

The main entry points are in `python/scripts/ppo_training/`. From `python/`:

```sh
python scripts/ppo_training/train_unified.py --help
python scripts/ppo_training/eval_ppo_descriptives.py --help
```

- **Fixed opponents:** `--mode fixed` trains against a mixture of heuristic policies.
- **Neural-only self-play:** `--mode selfplay` starts against its random initial neural checkpoint, trains against frozen same-run snapshots and evaluates against fixed neural references. See the [proof-of-concept guide](docs/SELFPLAY_POC.md) for pilots, separate holdout evaluation and a moderate training budget.
- **Evaluation:** the detailed evaluator records per-game outcomes and action statistics for a checkpoint or run directory.

Training and old experiments remain exploratory. Historical results used incorrect GAE and premature scoring and must be re-evaluated. Current checkpoints carry feature semantics and schema metadata; historical inference requires an explicit `--allow-legacy` override and is not directly comparable. No checkpoints are bundled. The [script index](python/scripts/README.md) separates the main tools from diagnostics and older trainers.

## Repository layout

```text
main.cpp, src/, include/          C++ simulator and vendored JSON headers
tests/                           C++ regression checks
python/src/sevenwonders_rl/
  envs/                          Gymnasium wrapper and engine transport
  features/                      Observations, action features and trade estimates
  policies/                      Heuristics and checkpoint-backed opponents
  evaluation/                    Shared completion and outcome handling
  selfplay/                      Frozen-checkpoint opponent pool
  utils/                         Actor-critic model and PPO utilities
  data/                          Engine-derived card/wonder data and mechanics
python/scripts/ppo_training/     Main training and evaluation tools
python/tests/                   Python tests and a selected fixture pool
docs/                            Origin, architecture and known issues
```

Generated builds, runtime exchange files, logs, training runs and checkpoints are ignored. The original GUI, its artwork and unrelated reference archives are not distributed in this source tree.

## License and attribution

This project is distributed under **GNU GPL version 3**; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The inherited GPLv3 license is retained and also applies to the downstream RL code and simulator modifications. The vendored **nlohmann/json** library retains its MIT license and copyright notices.
