# Consolidation, 6 November 2026

The project now supports one three-player Second Edition agent with visible
memory. This supersedes the [pre-cleanup audit](../research/PROJECT_CLEANUP_AUDIT_2026-11-06.md).
The selected source tree has 86 entries (previously 395), about 0.8 MB excluding
the separate engine and Git history; active Python modules fell from 62 to 21.

## Implementation

- One PPO learner loop handles local and process collection. Worker count is a
  runtime setting; one worker does not select a separate trainer. GAE uses the
  shared independently tested routine for each game's trajectory.
- Existing `decision-ppo-v2` model and observation contracts are preserved.
  Unspecified resume settings inherit their checkpoint values; explicit
  incompatible settings are rejected.
- Protocol-2 environments, trainer variants, multihead/prior recipes, old
  heuristics, legacy data exports, obsolete fixtures and their tests are retired.
  The frozen-snapshot inventory and pair-PFSP components remain because the
  active league uses them.
- The active evaluator exports a standalone dashboard. Public data are explicit
  aggregate fields, without raw games, local paths or training manifests.
  The selected public result is the existing complete 196,608-decision,
  40-block / 1,080-game Second Edition evaluation.
- README, architecture, operation and evaluation guides describe this system.
  Historical experiment records are linked by immutable Git revision; unique
  runtime measurements and the literature review are retained separately from
  current instructions. Repeated implementation acceptance logs are removed.
- Fast unit/component checks are separate from active integration tests.
  `make test-all` selects the entire suite. Pytest no longer builds an old engine.
  CI checks the pinned current engine and the remaining active contracts.
- Direct dependency constraints, portable VS Code workspace and package metadata
  support the documented Python 3.12 setup. Active Python files use consistent formatting.

## Data preservation

The previously running training was stopped cleanly at **319,808 decisions** and
**17,469 completed games** before changing the trainer. Its durable checkpoint,
league snapshots and evaluation panels were retained. A disposable copy of that
checkpoint was used to verify real resume compatibility.

Forty-seven old local run/diagnostic directories were moved, without deleting
their contents, to `python/runs/archive/pre-consolidation-20261106/`.
Its local inventory maps former paths to their archive locations. Old generated
build duplicates and coverage output were removed; they can be regenerated.
A private source backup and retired root artifacts are kept in the ignored
`.local-archive/20261106/`. Neither archive is part of the public source tree.

## Validation

- The initial complete active-system run passed 65 tests and exposed one resume
  default mismatch. The API now inherits that value from the checkpoint; its
  targeted regression passes.
- The fast suite, including new dashboard export and inert-JSON checks, passes.
  Independent evaluation and CLI resume were checked after integration changes.
- The pinned engine and exact generated card/wonder resources pass their checks.
  The editable Python package builds and installs as version 0.2.0.
- Browser checks cover desktop/mobile layout, English/German persistence,
  themes, CSV download, current/previous JSON loading and injection isolation.
  The README image is captured from the actual current dashboard.
- Relative documentation file links and the selected publication tree are
  checked before committing. No additional learning experiment is claimed.

These checks support the consolidation and interfaces. They do not establish
complete game-rule correctness, general playing strength or a causal memory benefit.
