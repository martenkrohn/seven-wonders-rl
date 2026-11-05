# Vorbereiteter Lauf: 2.097.152 zusätzliche Übergänge

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Stand: 26. Oktober 2026. Implementiert und mit kurzen Läufen geprüft. Der volle
Trainingslauf wird durch die Vorbereitung nicht gestartet.

## Starten und beobachten

Im Verzeichnis `python/`:

```bash
.venv/bin/python scripts/ppo_training/run_long_training.py
```

Währenddessen in einem zweiten Terminal:

```bash
.venv/bin/python scripts/open_dashboard.py --run-dir runs/long-mixed-wonders-v1-seed131
```

Nach jeder Evaluation die Browserseite neu laden. Dieselbe HTML-Datei enthält
alle abgeschlossenen Kontrollpunkte; der neueste steht zuerst im Auswahlfeld.
Die Tabelle darüber zeigt die Lernkurve einschließlich Punkteabstand ohne den
eingefrorenen Wissenschaftsspezialisten. Dashboard und Berichte werden während
des Trainings geschrieben; kein separater Auswertungsschritt ist erforderlich.

| Stand | Zusätzliche Übergänge | Kontrollspiele |
| --- | ---: | ---: |
| Start | 0 | 288 |
| 25% | 524.288 | 288 |
| 50% | 1.048.576 | 288 |
| 75% | 1.572.864 | 288 |
| Ende | 2.097.152 | 288 |

Jeder Kontrollpunkt verwendet dieselben zwölf geordneten Gegnerpaare, alle drei
Sitze und acht Seed-Blöcke. Zwei Paarungen enthalten Wissenschaftsspezialisten,
zehn kommen ohne sie aus. Die Evaluation pausiert kurz die Trainingssammlung;
danach läuft das Training automatisch weiter. Das Panel gewichtet Paarungen
gleich und bildet nicht die Trainingsverteilung ab. Es ist wiederholte
Trainingskontrolle; ein späterer, größerer Test mit neuen Seeds bleibt sinnvoll.

## Modell und Informationen

Start sind die fertigen v2-Gewichte, vier Köpfe und ihre unveränderten festen
Prior-Netze mit Stärke 32. Kopfwahl einmal pro Partie, Standardreward ohne
Strategieboni, neuer Phasen-Seed 131 und frischer Optimizer. Acht Umgebungen,
Horizont 64, 512 Übergänge je Update, insgesamt 4.096 Updates. Die bisherigen
PPO-Parameter bleiben erhalten.

Zusätzliche öffentliche Eingaben, relativ als selbst/links/rechts:

- Identität aller Wunder einschließlich A/B-Seite und Startressource;
- Plan aller bis zu vier Stufen: Kosten pro Ressource, Punkte, Münzen, Schilde
  und Sonderfähigkeiten, mit Markierungen für vorhandene/gebaute/nächste Stufe;
- Identitäten offen ausgespielter Karten aller Spieler;
- für Wunderbauaktionen: nächste Stufe, deren Kosten/Effekte und geschätzte
  Handelskosten. Handelswerte bleiben Schätzungen; die Engine bestimmt Legalität.

Die Daten entsprechen der vorhandenen Simulator-Edition. Bei Halikarnassos B
enthält das kombinierte `add_vp_discard` auch den Zugriff auf den Ablagestapel.
Die begrenzte RL-Aktionsschnittstelle wird dadurch nicht um neue Aktionen erweitert.
Verdeckte Hände anderer Spieler werden nicht eingelesen.

Globale Eingaben wachsen von 102 auf 645, Aktionsfeatures von 99 auf 121.
Die ursprünglichen Spalten behalten ihre Bedeutung. Neue Encoder-Spalten beginnen
bei null, sodass die Startpolicy bis auf numerische Rundung gleich bleibt.
Die vorhandenen Priors behalten ihre alten 201 Eingaben und sämtliche Gewichte.
Alte eingefrorene Gegner verwenden weiterhin ausschließlich 102/99 Eingaben;
neue Snapshots erhalten die erweiterten Eingaben. Schema und Datenbankinhalt
werden zusätzlich per Hash geprüft.

## Gegnerpool

Je Gegnerplatz unabhängig, Auswahl und Kopf für die ganze Partie eingefroren:

| Anteil | Inhalt |
| --- | --- |
| 50% | Letzte sechs eigene, ausreichend alte Snapshots |
| 15% | Bis zu sechs zeitlich verteilte eigene Snapshots vor diesen letzten sechs |
| 20% | Fertige v1/v2-Referenzen, je 10% |
| 5% | Ursprüngliche Standardmodelle, je 2,5% |
| 5% | Wissenschaftsspezialist |
| 5% | Militärspezialist |

Snapshots alle 131.072 Übergänge, Mindestabstand zum Lerner 65.536 Übergänge.
Fehlen Mitglieder einer dynamischen Gruppe, werden deren Plätze aus den reifen
v1/v2-Referenzen gezogen. Das Archiv ist nach Zeit verteilt; seine strategische
Vielfalt wird nicht automatisch bewertet. Gruppengewichte werden nicht von der
Anzahl gespeicherter Dateien verändert. Alle Quellen werden vor Start gegen
vorbereitete Hashes geprüft und als unveränderliche Kopien im Lauf abgelegt.

## Dateien und Unterbrechungen

Ziel: `python/runs/long-mixed-wonders-v1-seed131/`.

- `dashboard.html`: gemeinsame Ansicht aller fertigen Evaluationen.
- `evaluations/step_XXXXXXX/`: jeweiliges `model.pt`, Rohspiele, `analysis.json`,
  Text-/CSV-Berichte, eigenes Dashboard und abschließendes `summary.json`.
- `evaluation_plan.json`, `learning_curve.json`, `training_status.json`: fester
  Spielplan, Verlauf und aktueller Fortschritt.
- `latest.pt`: atomarer Sicherungsstand mit Optimizer und Zufallszuständen,
  alle 8.192 Schritte sowie an Kontrollpunkten; `evaluated.pt`: zuletzt geprüfter Stand.
- `opponents.jsonl` und `episodes.jsonl`: Gegnergruppen, Fallbacks, Köpfe und Partien.

Nach einer Unterbrechung mit denselben Einstellungen weiterführen:

```bash
.venv/bin/python scripts/ppo_training/run_long_training.py --resume
```

Gewichte, Optimizer, Sampling-Zustände und Updatezähler werden wiederhergestellt.
Bis zu acht am Sicherungsstand laufende Engine-Partien werden ausdrücklich
aufgegeben und mit neuen Episode-Seeds ersetzt; das wird in `resume_events.jsonl`
protokolliert. Es ist keine bitgenaue Fortsetzung dieser Partien. Noch nicht
gesicherte Updates werden zurückgesetzt. Unvollständige Evaluationsordner bleiben
archiviert, fehlende Evaluationen werden nachgeholt, auch bei einer Unterbrechung
der Endevaluation. Vorhandene fertige Messpunkte bleiben erhalten.

## Prüfung

`tests/test_long_training.py` prüft Informationsgrenzen und Sitzrotation,
neutrale Gewichtsmigration, unveränderte Priors, Lernfähigkeit neuer Eingänge,
Schema-Validierung, Poolgewichte/Abstand/Fallback sowie sichtbare Zwischenberichte
und Weiterführung nach Unterbrechungen.

Ein zusätzlicher Probelauf verwendet die tatsächlichen v2-Gewichte und alle sechs
Referenzen mit 4.096 Übergängen und verkürzten Snapshot-Intervallen. Seine kleinen
Evaluationen prüfen den Ablauf und sind kein Urteil über Lernfortschritt.

Der abschließende Probelauf
`runs/long-mixed-wonders-validation-20261026` ist abgeschlossen: fünf Evaluationen
mit jeweils 72 Spielen, insgesamt **360/360 ohne Fehler oder Abbruch**. Alle fünf
Stände stehen im gemeinsamen Dashboard. Die neuen Eingangsgewichte lernen;
sämtliche Prior-Gewichte bleiben bytegleich zum v2-Quellmodell. Zusätzlich prüft
ein Test sämtliche 14 Wunder gegen `src/wonder.cpp`: Identitäten, Startressourcen,
Stufenkosten, feste Punkte-/Münzen-/Schildgewinne und Fähigkeiten.

Gemessen wurden etwa **139 Übergänge/s ohne Evaluationszeit** einschließlich
Initialisierung/Sicherungen im kurzen Lauf (144/s für Sammlung und PPO-Updates).
Für den vollen Lauf grob **4–6 Stunden** einplanen; der größere spätere
Snapshot-Pool und Systemauslastung können die Geschwindigkeit verändern.
Messwerte: [LONG_TRAINING_VALIDATION_2026-10-26.json](LONG_TRAINING_VALIDATION_2026-10-26.json).
