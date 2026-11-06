# Engine workflow

The simulator, Python transport client, GUI, game definitions and C++ tests live
in the separate [engine repository](https://github.com/martenkrohn/seven-wonders-engine).
`engine/` is a Git submodule. `engine.lock.json` and the Git submodule entry pin
the exact supported commit and rules/protocol contract.

For a new checkout, run `git submodule update --init --recursive` and follow
[setup](../../README.md#setup). Do not point current training at a historical
protocol-2 binary.

To adopt a new engine version: validate it in the engine repository, update the
submodule commit and lock together, regenerate neutral resources using
`python/scripts/export_bga_data.py`, rebuild the binary and run the active
integration tests. `check_engine.py` verifies both the pin and a clean submodule;
`export_bga_data.py --check` verifies exact resource correspondence. Check the
reported engine commit before resuming; rules/feature incompatibilities require
a fresh training run rather than an override.

The root CMake project builds `build/7Wonders` through the submodule. The optional
GUI is documented in [the engine guide](../../engine/docs/GUI.md).
Provenance and engine correction history belong to that repository.
