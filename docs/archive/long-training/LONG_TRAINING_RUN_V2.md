# Zweiter langer Lauf: eigene Gegner, 60/20/20

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Aus Ubuntu starten:

```bash
cd /home/marten/dev/7Wonders/python
.venv/bin/python scripts/ppo_training/run_long_training_v2.py
```

Neue Phase aus dem Endmodell des ersten langen Laufs, weitere 2.097.152 Übergänge,
vier Köpfe, unveränderte Priors mit Stärke 32 und öffentliche Informationen.
Neuer Trainingsseed 231, frischer Optimizer; sonst bisherige PPO-Einstellungen.
Kein voller Lauf startet durch die Vorbereitung.

Vorbereitung geprüft: tatsächlicher Modell-Warmstart mit 1.024 Übergängen und
fünf verkürzten Evaluationen (210/210 Spiele ohne Fehler/Abbruch). Trainingslogs
enthalten ausschließlich jüngere/historische eigene Modelle und die beiden Anker;
Evaluationsgegner sind separat kopiert. Probelauf unter
`runs/long-mixed-wonders-v2-command-smoke/`.

Trainingsgegner je Platz: 60% letzte sechs eigene Snapshots (alle 131.072 Schritte,
mindestens 65.536 Schritte Abstand), 20% zeitlich verteiltes Archiv,
20% feste Anker (bisheriges Endmodell und 25%-Stand, je 10%). Das Archiv startet
mit sechs eingefrorenen Entwicklungsständen des vorherigen langen Laufs.
Später werden bis zu sechs Archivmitglieder zeitlich über beide Phasen verteilt
ausgewählt, ohne die jüngsten sechs neuen Snapshots. Fehlende aktuelle Mitglieder
fallen auf die beiden Anker zurück. Keine Spezialisten oder ursprünglichen
Standardmodelle im Trainingspool. Auswahl unabhängig je Gegnerplatz, Köpfe einmal
je Partie. Quellen werden auf vorbereitete Hashes geprüft und kopiert.

Getrennte Evaluation bei 0/25/50/75/100%: die bisherigen zwölf Referenzpaarungen
einschließlich Spezialisten/Standardmodelle plus je eine Paarung gegen zwei
Exemplare des bisherigen Endmodells und zwei Exemplare des bisherigen 25%-Stands.
14 Paarungen × 3 Sitze × 8 Seed-Blöcke = 336 Spiele je Kontrollpunkt.
Die Evaluation verändert den Trainingspool nicht. Kontrollseed 149001;
wiederholte Kontrolle, kein abschließender unabhängiger Holdout.

Ausgabe: `python/runs/long-mixed-wonders-v2-seed231/`.
Dashboard während des Trainings:

```bash
.venv/bin/python scripts/open_dashboard.py --run-dir runs/long-mixed-wonders-v2-seed231
```

Nach Kontrollpunkten Seite neu laden. Nach Unterbrechung:

```bash
.venv/bin/python scripts/ppo_training/run_long_training_v2.py --resume
```

Resume stellt den letzten gesicherten vollständigen PPO-Update samt Optimizer
und Samplingzuständen wieder her. Unfertige Engine-Partien werden protokolliert
aufgegeben und mit neuen Episode-Seeds ersetzt, wie im ersten langen Lauf.
