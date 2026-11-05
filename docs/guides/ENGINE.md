# Engine und RL getrennt entwickeln

Die [Engine](https://github.com/martenkrohn/seven-wonders-engine) ist ein
eigenständiger Fork von `dmag-ufsm/7Wonders`. Das RL-Projekt bindet sie unter
`engine/` als Git-Submodule ein. Der Submodule-Eintrag und
[engine.lock.json](../../engine.lock.json) halten denselben vollständigen
Commit fest. `v0.1.0-legacy` ist die unveränderte Spielimplementierung vor der
BGA-Migration. Der Sicherungstag `pre-engine-split-2026-11-05` enthält den
bisherigen gemeinsamen Projektstand.

Der [Prüfnachweis der Trennung](../reference/engine-split-validation.json)
enthält Quell- und Binary-Prüfsummen, sieben identische vollständige
Spielverläufe sowie lokale und GitHub-Prüfergebnisse.

Die alte Tk-GUI ist mit `v0.1.1-legacy` wieder als optionales Frontend verfügbar;
der C++-Regelstand bleibt gleich. [Start und Bedienung](../../engine/docs/GUI.md),
[GUI-Audit](../../engine/docs/GUI_AUDIT.md),
[verbindlicher Arbeitsplan](../../engine/docs/MIGRATION_PLAN.md) und
[nächster Agentenauftrag](../../engine/docs/NEXT_AGENT_TASK.md) liegen in der Engine.

## Bestehenden Stand verwenden

```sh
git submodule update --init --recursive
python3 python/scripts/check_engine.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

Neue Klone verwenden `git clone --recurse-submodules`. Die bekannten
Trainingsskripte verwenden weiterhin `build/7Wonders`. Die Engine benötigt
selbst nur CMake, C++17 und optional Python für ihren Protokolltest.

## Engine ändern und bewusst übernehmen

1. In `engine/` einen Arbeitsbranch erstellen. C++-Änderung, fachliche
   Regressionen, Engine-Vertrag und Herkunftsnachweis dort pflegen. Anleitung:
   [Engine-Entwicklung](../../engine/docs/DEVELOPMENT.md). Vor einem Wechsel
   den lokalen Arbeitsstand prüfen; Submodule-Updates können sonst stören.
2. Eigenständig bauen und testen, dann im Engine-Repository committen,
   veröffentlichen und gegebenenfalls taggen.
3. Im RL-Repository den gewünschten Engine-Commit auschecken. Den vollständigen
   Commit und die Vertragswerte in `engine.lock.json` übernehmen.
4. Bei Datenänderungen die Python-Datenbanken bewusst neu exportieren und
   deren Änderungen prüfen. Feature-, Aktions- und Checkpoint-Verträge bei
   Regeländerungen ebenfalls prüfen. Historische Trainingsstände behalten
   ihren ursprünglichen Engine-Bezug.
5. `git add engine engine.lock.json`, `check_engine.py`, Herkunftsprüfung,
   Build, CTest und Python-Tests ausführen. Erst anschließend den RL-Commit
   erstellen. Ein Engine-Commit muss auf GitHub erreichbar sein, bevor der
   Consumer darauf zeigt.

Die Audit-/Export-/Vergleichsskripte lesen die Quellen jetzt aus `engine/`;
ihre Ausgaben bleiben im RL-Projekt. Alte Auditberichte behalten ihre
historischen Pfadangaben. Die im Editionsaudit belegten Exportfehler
(unter anderem verschobene Wunder-IDs und Produktionsmengen) bestehen weiter;
die gepflegten JSON-Daten wurden bei der Trennung bewusst nicht neu erzeugt.
Vor einer Datenmigration sind diese Exportfehler separat zu korrigieren.
Neue Trainingsmanifeste erfassen zusätzlich den
Engine-Quellcommit, Arbeitsstatus und Regelnamen; der vorhandene Binary-Hash
bleibt erhalten. Ein neu gebautes Binary kann trotz gleicher Regeln einen
anderen Hash besitzen. Die bisherigen Resume-Prüfungen werden nicht umgangen.

## Nächste fachliche Änderung

Die BGA-Quelle ist die Zieldefinition für die neue Edition. Referenzen und
[vollständige Gegenüberstellung](../reference/edition-comparison/README.md)
bleiben hier. Die Trennung selbst migriert weder Karten noch Wunder. Der
nächste Schritt wird als eigener Engine-Branch mit gezielten Regeltests
umgesetzt und erst nach Anpassung der RL-Schnittstelle übernommen.
