# Origin and contribution boundary

The original C++ simulator comes from [DMAG/UFSM 7Wonders](https://github.com/dmag-ufsm/7Wonders),
upstream baseline `a0bc333a7a7f21f83156de8a2d522e7baa913f0a` (7 October 2021).
Its maintained downstream [engine fork](https://github.com/martenkrohn/seven-wonders-engine)
is included at the exact commit in [engine.lock.json](../engine.lock.json).
Engine source, game/GUI changes, exact patches and inventories live in
[the engine origin record](../engine/docs/ORIGIN.md) and
[engine change history](../engine/docs/SIMULATOR_CHANGES.md).

This repository contributes the Python Gymnasium environment, privacy-bounded
observation/history encoding, actor-critic and PPO training, frozen-opponent
league, evaluation, dashboard, tests and research analysis. It does not claim
the inherited simulator as an implementation from scratch. Development uses
coding assistants; contribution claims describe project scope and decisions,
not exclusive manual authorship of every line.

The inherited GPLv3 source license is retained: [LICENSE](../LICENSE),
[NOTICE](../NOTICE). The engine carries its own bundled-library and asset
notices. Game names and trademarks are not relicensed by this project.
Historical protocol-2 source and reports remain in Git history.
