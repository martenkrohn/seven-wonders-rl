# Working on Seven Wonders RL

The supported system is the three-player Second Edition agent with explicit
visible memory. Read `docs/README.md` for the current architecture and workflows.
Do not restore retired protocol-2 trainers or compatibility paths just to keep
historical code executable; old source and experiment records remain in Git.

Use verification proportional to the change:

- Console wording or formatting: run the affected console/format tests. Do not
  run training, serial/parallel comparisons, benchmarks or an engine build.
- Documentation: check the changed paths, links and commands. No training tests.
- Dashboard: renderer tests; browser checks when interaction/layout changes.
- Features/environment: the affected observation, privacy and engine contracts.
- Training/GAE/collector: relevant numerical, budget, boundary and short real
  integration tests. Resume/worker changes also need the corresponding recovery tests.
- A release or substantial cross-module refactor: run the complete active suite
  once. Repeat only failing/affected checks unless new changes justify more.

From `python/`, `make test-fast` runs engine-free checks and `make test-all` runs
all Python tests. Engine builds belong to changes in the pinned engine or build
integration. There is one PPO implementation for every worker count; one worker
is a collector setting, not an independently maintained trainer.

Never delete or modify an existing training run for a test. Use a temporary
directory or a disposable copy. Check the training lock before changing active
runtime code; request a clean boundary stop if necessary and preserve the durable
checkpoint, league and referenced evaluation opponents.

Keep runs, weights, raw logs, caches, local source archives and assistant/account
state out of commits. Select publication files deliberately and inspect the
staged diff plus target tree. Current behavior belongs in one guide/contract;
do not add a validation JSON or handover note for each small change.
