# Änderungsprotokoll der Engine

Seit der Trennung vom 5. November 2026 besitzt das eigenständige
[Engine-Repository](https://github.com/martenkrohn/seven-wonders-engine) das
[vollständige Änderungsprotokoll](../engine/docs/SIMULATOR_CHANGES.md), den
[Upstream-Patch](../engine/docs/simulator/upstream.patch) und das
[Prüfsummenverzeichnis](../engine/docs/simulator/manifest.json).

`engine/` ist als Git-Submodule auf einen festen Commit gesetzt; der Vertrag
steht in [engine.lock.json](../engine.lock.json). Die bekannte Einstiegsstelle
funktioniert weiter:

```sh
python3 python/scripts/audit_simulator.py --check
```

Der Tag `pre-engine-split-2026-11-05` bewahrt den vollständigen Stand vor der
Trennung. Die Spielregeln wurden bei der Ausgliederung nicht verändert.
Weitere Pflege: [Engine-Arbeitsweise](guides/ENGINE.md).
