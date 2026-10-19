# Validation

Checked on 19 October 2026 using Ubuntu/WSL and Python 3.12.3.

| Check | Result |
| --- | --- |
| Source comparison | Compared the engine against upstream `a0bc333a7a7f21f83156de8a2d522e7baa913f0a`; four engine/build files differ before the documentation notices |
| C++ build | Fresh CMake configure/build completed in the reviewed source checkout |
| CTest | Both existing choice-production regression programs passed |
| Python suite | All 36 existing tests passed, including engine integration |
| Documented entry points | Main trainer and detailed evaluator `--help` commands completed |
| Package metadata | Wheel built without dependency downloads; GPLv3 classifier and license file present |
| License preservation | Root license retained; Python distribution receives an identical copy; vendored notices unchanged |
| Documentation | Relative Markdown links checked; references to the private restart guide removed |

The runtime environment, PPO implementation, feature schema and earlier engine corrections were not changed by this documentation cleanup. Source notices were added to the modified engine files. No new training experiment, checkpoint evaluation or benchmark was run. Passing interface tests does not resolve the training issues described in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

No TeX compilation, image review or screenshots were performed.
