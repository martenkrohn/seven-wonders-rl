# Development

Use Linux/Ubuntu or WSL, Python 3.12, a C++17 compiler and CMake.
From the repository root:

```sh
git submodule update --init --recursive
python3.12 -m venv python/.venv
python/.venv/bin/python -m pip install -c python/constraints.txt -e './engine' -e './python[dev]'
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
python/.venv/bin/python python/scripts/check_engine.py
python/.venv/bin/python python/scripts/export_bga_data.py --check
```

The constraints file records the tested direct dependency versions. It is not
a cross-platform lock of every transitive package. A CPU-only PyTorch wheel can
be installed using the official PyTorch CPU index before installing the project.

Choose local checks by what changed:

| Change | Local check from `python/` |
| --- | --- |
| Console strings/formatting | `.venv/bin/python -m pytest tests/test_decision_console.py` |
| Dashboard rendering | `.venv/bin/python -m pytest tests/test_dashboard.py` |
| GAE or matchmaking mathematics | `.venv/bin/python -m pytest tests/test_gae.py tests/test_pair_pfsp.py` |
| Environment, features or memory | Relevant `test_decision_bga.py`, `test_decision_memory.py`, `test_decision_performance.py` tests |
| Collector or training loop | `test_decision_parallel.py` plus relevant league/resume tests |
| Documentation | Check local links and the commands actually changed; no training campaign |

```sh
cd python
make test-fast          # unit + component; no engine games
make test-integration   # current engine, PPO, league, failure and resume contracts
make test-all           # every Python test, including unmarked future additions
```

CI checks the complete remaining active system. No historical engine is cloned
or built by pytest. Engine source changes are tested in the engine repository;
run `ctest --test-dir build --output-on-failure` after updating its pin. A full
integration pass is appropriate for a trainer refactor or release, not a local
prerequisite for each text edit.

Training outputs and checks belong under `python/runs/` or a temporary directory.
Do not add routine validation logs to documentation. Preserve a compact result
with method, versions and limitations when a measurement answers a real question.

Before publishing, select files explicitly and review `git diff --cached` plus
the entire target tree. `.gitignore` excludes local runs and weights, but cannot
remove already tracked data or older public revisions.
