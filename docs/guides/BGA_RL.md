# BGA-Entscheidungen und frisches Training

Stand: 6. November 2026. Release: `v0.2.0-bga`, Regeln `bga-second-edition`,
Protokoll 3, `second_edition_implemented=true`. Der aktive Einstieg ist `DecisionEnv`, auch unter
`SevenWondersEnv` exportiert und als `SevenWondersBGA-v0` registriert. Er benötigt
den gemeinsamen Client aus dem Engine-Repository, Protokoll 3 und `decision-v1`.
Das Grundspiel ist auf drei Spieler begrenzt. Eine Entwicklungsengine muss
ausdrücklich mit `allow_development=True` freigegeben werden.

```sh
python -m pip install -e ./engine -e './python[dev]'
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
python python/scripts/check_engine.py
python python/scripts/export_bga_data.py --check
python python/scripts/train_decision_ppo.py --steps 256 --eval-games 3 --output-dir /tmp/bga-training
```

Der Standardstart bevorzugt `engine/build/7Wonders`, falls dort ein Binary liegt,
sonst `build/7Wonders`. Veraltete Binaries werden anhand des Consumer-Locks
zurückgewiesen. Mit `--engine /absoluter/pfad/7Wonders` kann ein separat gebautes
Binary gewählt werden. Jede Partie hat ein eigenes temporäres Arbeitsverzeichnis;
ein Abbruch oder fehlerhafter Austausch erzeugt keinen normalen Lernübergang.

## Daten und Beobachtung

`scripts/export_bga_data.py` übernimmt die neutralen, versionierten Engine-Exporte
bytegleich. Es liest keinen C++-Quelltext. `--check` schreibt nichts. Beide JSONs
werden als Python-Paketressourcen ausgeliefert: 80 Karten-/Zeitalterzeilen und
14 Wunderseiten. Kartenschlüssel sind `(engine_id, age)`, einschließlich grauer
Varianten. Die historischen Datenbanken bleiben für Legacy-Experimente erhalten.

Die Beobachtung enthält `public_table` (eigener Sitz, links, rechts), `context`,
`action_feats` und `action_mask`. Gegnerhände und gegnerische Entscheidungsoptionen
gehen weder an Policies noch ins zurückgegebene `info`. Produktionsmengen,
Science-Bestand mit Jokern, Stufen und öffentliche Kartenfarben sind sichtbar.
Science-Zugewinne werden optimal über alle vorhandenen Joker berechnet.

Die bis zu 128 Kandidaten sind tatsächlich angebotene physische Optionen der
Engine: normaler Bau, Wunderbau, Verkauf und Pflichtbau aus der Ablage, jeweils
mit konkreter Zahlung und explizitem Freibaumodus. Padding ist ungültig. Die
Merkmalsnamen stehen in `features/decision_features.py::FEATURE_NAMES`.
Wunderaktionen beschreiben den nächsten wirklichen Stufeneffekt; zusätzlich
bleibt die dafür aufgegebene Handkarte beschrieben. Das Verfahren errät keine
Fähigkeiten anhand von Wunder-IDs und rekonstruiert keine Legalität.

## Übergänge und Checkpoints

Ein `step()` ist genau eine tatsächlich angewandte Entscheidung des Lerners.
Gegnerentscheidungen werden bis zur nächsten eigenen Entscheidung abgearbeitet.
Eine Pflichtwahl kann im selben globalen Zug folgen. Sitze ohne Entscheidung
erzeugen keine künstlichen Schritte. Die History enthält Angebot, tatsächliche
Bestätigung und Zuggrenzen. Endbelohnung ist eigener Gesamtwert minus bester
Gegnerwert, optional skaliert; Zwischenbelohnungen sind null. Terminalwerte
werden gegen die normale Ergebnisdatei geprüft.

Der neue PPO lernt mit maskierten Kandidaten, GAE und geclipptem Policy-Update.
Checkpoints tragen `decision-ppo-v1`, Regelnamen, Protokollversion,
Feature-Semantik, Dimensionen, Namen sowie Quellen- und Datensatzhash. Laden
verlangt die exakte Übereinstimmung. Alte 28-Aktionen-Checkpoints werden abgewiesen;
alte Lernkurven sind kein Vergleich für diese Semantik. Ein Resume lädt Modell,
Optimizer und Torch-Zufallszustand und startet bewusst eine frische Partie;
es ist keine Wiederherstellung eines laufenden Spiels.

```sh
python python/scripts/train_decision_ppo.py --steps 128 --eval-games 3 \
  --resume /tmp/bga-training/checkpoint.pt --output-dir /tmp/bga-resume
```

Kurze Trainingsläufe prüfen die technische Pipeline, nicht Spielstärke.
`tests/test_decision_bga.py` prüft öffentliche Sichtbarkeit, echte Übergänge,
alle Sitze, Spezialentscheidungen, optimale Science-Marginalwerte, reproduzierbare
Seeds, Gewichtsänderung, Reload/Resume und die Ablehnung alter Verträge.

## Historische Werkzeuge

Die vorhandenen alten Trainings-/Auswertungsskripte importieren ausdrücklich
`LegacySevenWondersEnv` und bleiben Protokoll 2 zugeordnet. Die vollständige
Testsuite baut dafür einen lokalen, unveränderten Checkout des erhaltenen
Engine-Commits `f39c1d7`; sie testet nicht alte Merkmale gegen neue Regeln.
`SEVENWONDERS_LEGACY_ENGINE_ROOT` und `SEVENWONDERS_LEGACY_ENGINE` können einen
bereits gebauten historischen Checkout angeben. Die aktive BGA-Umgebung benutzt
diese Variablen nicht. Neue Experimente starten über `train_decision_ppo.py`.
