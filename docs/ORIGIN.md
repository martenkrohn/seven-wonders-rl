# Source origin and downstream changes

## Baseline

The simulator is derived from [dmag-ufsm/7Wonders](https://github.com/dmag-ufsm/7Wonders), using upstream commit [`a0bc333a7a7f21f83156de8a2d522e7baa913f0a`](https://github.com/dmag-ufsm/7Wonders/tree/a0bc333a7a7f21f83156de8a2d522e7baa913f0a), dated 7 October 2021, as the comparison baseline. Its commit message is `Update LICENSE`.

The original local history was inspected on 19 October 2026. The comparison below describes the simulator changes already present before this documentation cleanup. The exact dates and line-by-line authorship of the earlier downstream edits are not asserted.

| Engine file | Difference from upstream baseline |
| --- | --- |
| `main.cpp` | Unchanged game loop and JSON exchange protocol |
| `src/card.cpp`, `src/filer.cpp` | Unchanged |
| `include/card.h`, `include/filer.h`, `include/resources.h`, `include/wonder.h` | Unchanged |
| `include/player.h` | Replaces old extra-production flags with state for wonder choice production |
| `src/player.cpp` | Tracks choice use/reset, removes duplicated extra raw production and restricts the neighbor use of Caravansery production |
| `src/wonder.cpp` | Changes the Gizah B fourth-stage case from `4` to `3` in the build and feasibility paths |
| `CMakeLists.txt` | Adds regression executables and CTest registration |
| `include/nlohmann/` | Vendored upstream JSON library with its original notices |

The four modified engine/build files are marked with downstream notices. These targeted corrections do not constitute a rewrite or a complete rule audit.

## RL extension

The `python/` package and scripts add the Gymnasium bridge, relative-seat observations, action masks/features, PyTorch actor-critic, PPO training, heuristic policies, frozen-checkpoint opponents, evaluation tooling and test fixtures. The C++ regression tests, Python tests and CI setup are also downstream additions.

This is the project's contribution boundary. Coding assistants have been used during development; the original engine is attributed to its upstream authors rather than presented as a new implementation by the downstream maintainer.

## License choice

The pinned upstream source contains a GPLv3 `LICENSE`, and that exact license text is retained here. The combined simulator and downstream RL code are distributed under **GPL-3.0-only**. The Python distribution includes a copy of the same license. No broader "or later" grant is added.

The upstream README has a conflicting MIT sentence. This project follows the explicit GPLv3 license supplied in the source snapshot; it does not claim that the upstream authors issued a separate MIT grant for this snapshot. The discrepancy is recorded rather than silently copied into this project's README.

The embedded MIT notices in `include/nlohmann/` remain intact. Game names and trademarks are not relicensed. Card/wonder data are derived from engine definitions using the included extraction scripts; fixtures are recorded simulator states. GUI artwork, rulebooks and unrelated scraped datasets are excluded.

Relevant primary sources: [upstream LICENSE](https://github.com/dmag-ufsm/7Wonders/blob/a0bc333a7a7f21f83156de8a2d522e7baa913f0a/LICENSE), [GPLv3 terms](https://www.gnu.org/licenses/gpl-3.0.html), [nlohmann/json license](https://github.com/nlohmann/json/blob/v3.7.0/LICENSE.MIT).
