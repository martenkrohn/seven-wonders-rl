# Frischer Trainingsstart mit einer Liga aus eigenen Zwischenständen

Stand: 29. Oktober 2026. Die neue Variante ist eingerichtet und technisch
prüfbar. Der lange Lauf wurde noch nicht gestartet. Das
[Versuchsprotokoll](../experiments/001-fresh-pair-pfsp/PROTOCOL.md) beschreibt
die wissenschaftliche Fragestellung und Grenzen.

Für Training ohne festes Endbudget mit VS-Code-Start-/Stop-Tasks siehe
[fortlaufendes Training](CONTINUOUS_TRAINING.md). Die folgenden Befehle
beschreiben weiterhin die begrenzte Variante mit 2.097.152 Übergängen.

## Starten unter Ubuntu

Die folgenden Befehle werden im Verzeichnis `7Wonders/python/` ausgeführt.
Die vorhandene virtuelle Umgebung und die gebaute Engine werden verwendet.

```bash
.venv/bin/python scripts/ppo_training/run_fresh_pair_pfsp.py
```

Damit startet ein neuer Lauf mit **2.097.152 eigenen Spielerübergängen** in
`runs/fresh-pair-pfsp-v1-seed331/`. Die Gewichte sind neu initialisiert,
der Optimizer ist neu und die Trainingsliga enthält anfangs ausschließlich
`checkpoint_0.pt` dieses Laufs. Keine bisherigen Modelle, Spezialisten oder
heuristischen Bots werden als Trainingsgegner verwendet.

Vier Köpfe, feste Actor-Logit-Priors mit Stärke 32, vollständige öffentliche
Informationen und das gemeinsame Beobachtungsgedächtnis bleiben aktiviert.
Die eingefrorenen Gegner wählen wie bisher pro Partie einen Kopf. Die
Auswahlstatistik fasst diese Kopf-Mischungen pro Checkpoint zusammen.

Vorab lässt sich die Konfiguration prüfen, ohne ein Laufverzeichnis anzulegen:

```bash
.venv/bin/python scripts/ppo_training/run_fresh_pair_pfsp.py --dry-run
```

Das Laufverzeichnis muss neu sein. Ein abgebrochener Lauf lässt sich mit
derselben Konfiguration fortsetzen:

```bash
.venv/bin/python scripts/ppo_training/run_fresh_pair_pfsp.py --resume
```

Dabei werden der letzte gespeicherte PPO-Update, Optimizer, Zufallszustände
und Liga-Ergebnisse wiederhergestellt. Unfertige Partien werden aufgegeben
und mit neuen Seeds begonnen; es ist keine bitgenaue Fortsetzung der Engine.
Der Budgetzähler läuft bis zum ursprünglichen Gesamtbudget weiter.

Ein kurzer technischer Test in einem **anderen** Verzeichnis:

```bash
.venv/bin/python scripts/ppo_training/run_fresh_pair_pfsp.py \
  --out-dir runs/fresh-pair-pfsp-technical-check \
  --total-timesteps 256 --snapshot-every 64 --eval-blocks 1
```

Dies ist kein sinnvoller Spielstärkeversuch. Bei einer Wiederaufnahme müssen
dieselben Überschreibungen erneut angegeben werden.

## Auswahlregel und Parameter

Die maßgebliche Rezeptdatei ist
[fresh_pair_pfsp_v1.json](../../python/configs/fresh_pair_pfsp_v1.json).
Alle aufgelösten Trainerargumente werden in `config.json` gespeichert.
Die folgende Regel ist eine PFSP-inspirierte Übertragung auf drei Spieler,
keine Implementierung der vollständigen AlphaStar-Liga und kein JPSRO.

| Einstellung | Wert und Bedeutung |
| --- | --- |
| Neue Gegner | Eigene eingefrorene Kopie alle 65.536 Übergänge |
| Startphase | Zwei Gegner aus der eigenen Anfangskopie, bis weitere Kopien existieren |
| Berechtigung | Veröffentlicht, nicht aus der Zukunft; kein zusätzlicher Schritt-Lag |
| Aktiver Pool | Höchstens 64 neueste Kopien; alle Dateien bleiben erhalten |
| Erwartete Größe dieses Laufs | 33 Kopien einschließlich Start; deshalb keine Verdrängung |
| Kandidaten | Alle geordneten Paare, einschließlich zweimal desselben Checkpoints |
| Priorisierung | Gewicht `(1 − geschätzter eigener Sieganteil)²` |
| Kontrollanteil | 10% gleichverteilte Paar-Auswahl, 90% normalisierte Priorisierung |
| Neutrale Schätzung | Sieganteil 1/3 mit Gewicht von drei Pseudo-Spielen |
| Alterung | Beobachtungsgewicht halbiert sich nach 131.072 weiteren Übergängen |
| Feedback | Nur regulär abgeschlossene Trainingspartien |

Für ein Paar mit diskontierter Spielanzahl `N` und diskontierter Summe der
Sieganteile `S` gilt:

```text
w = (S + 3 × 1/3) / (N + 3)
q(Paar) = 0,90 × (1 − w)² / Summe_aller_Paargewichte + 0,10 / Anzahl_Paare
```

Ein alleiniger Sieg zählt 1, eine geteilte Höchstpunktzahl unter zwei Spielern
1/2, unter drei Spielern 1/3 und eine Niederlage 0. Das entspricht dem
vorhandenen Evaluationsmaß nach Siegpunkten. Neue Paare haben zunächst `N=0`.
Es wird keine unbekannte Stärke als sicher bekannt behandelt.

Beim nächsten Ergebnis werden bisherige Summen mit
`2 ** (−vergangene_Übergänge / Halbwertszeit)` abgewertet und das neue Ergebnis
addiert. Dadurch nähert sich eine veraltete Schätzung wieder dem neutralen
Ausgangspunkt an. Die Zeitmarke ist der Beginn des aktuellen PPO-Updates.
Die Schätzung umfasst Ergebnisse des sich verändernden Lerners und mittelt
über Köpfe, Sitze, Wunder und Austeilungen; sie ist keine exakte Auszahlung
einer festen aktuellen Policy. Kleine Stichproben bleiben unsicher.

Der Kontrollanteil ist **keine zusätzliche Evaluation**: Das sind gewöhnliche
Trainingspartien mit gleichverteilter Gegnerauswahl. Sie lernen ebenfalls mit
und aktualisieren die Schätzung. Die Mindestchance je Paar beträgt damit
`0,10 / Anzahl_Paare`; sie sinkt bei wachsender Liga. Eine vollständige
Abdeckung innerhalb eines endlichen Budgets ist nicht garantiert.
Bei 33 Modellen entstehen bereits 1089 Paare. Die effektiven Stichproben
bleiben deshalb eine wichtige Diagnose; die Anzahl gespeicherter Paare ist
kein Beleg für präzise Schätzungen.

Der Lernreward bleibt `(eigene Punkte − höchster Gegnerpunktestand) / 10`
am Spielende. Der Sieganteil steuert ausschließlich die Gegnerauswahl.
Strategische Vielfalt und ein Leistungsgewinn durch PFSP müssen gemessen werden.

## Evaluation und Dashboard

Die beiden früheren langen Endmodelle werden ausschließlich für den festen
Kontrollplan kopiert. Ihre Quellprüfsummen werden vor dem Start geprüft:

- `long-mixed-wonders-v1-seed131/evaluations/step_2097152/model.pt`
- `long-mixed-wonders-v2-seed231/latest.pt`

Der Plan enthält alle vier geordneten Paare dieser Modelle und alle drei
Sitze, mit acht gemeinsamen Seed-Blöcken: **96 Partien je Kontrollpunkt**.
Evaluationen erfolgen bei 0, 524.288, 1.048.576, 1.572.864 und 2.097.152
Übergängen. Ihre Ergebnisse beeinflussen die Liga-Schätzungen nicht.
Während dieser Evaluationen pausiert die Trainingssammlung.

Die vorbereitete Rezeptdatei aktiviert außerdem die
[feste Dashboard-Referenz](EVALUATION.md#bereinigte-dashboard-auswahl).
Das bisherige Endmodell wird einmal auf diesem Plan ausgewertet; die neuen
Berichte vergleichen direkt damit. Diese einmaligen 96 zusätzlichen Partien
ändern weder den Trainingspool noch den Reward.

```bash
.venv/bin/python scripts/open_dashboard.py \
  --run-dir runs/fresh-pair-pfsp-v1-seed331
```

Nach jedem Kontrollpunkt die Seite neu laden. Frühere Ergebnisse bleiben
auswählbar. Das Dashboard zeigt zusätzlich den gespeicherten Ligastand und
die aktuell häufigsten Gegnerpaare. Das Panel ist eine wiederholt verwendete
Entwicklungskontrolle, kein unberührter abschließender Test.

`league_status.json` enthält alle aktiven Paare mit Auswahlchance,
Sieganteilschätzung, effektiver Spielzahl, roher Spielzahl und letzter
Ergebnis-Zeitmarke. Die rohen Statistiken stehen unter `results`; für noch
unbeobachtete Paare existiert dort kein Eintrag. `opponents.jsonl` protokolliert
die tatsächlichen Auswahlen einschließlich Kontroll-/PFSP-Zweig,
Auswahlchance, Checkpoint-Hashes und Kopfzuordnung. `episodes.jsonl` enthält
die zugehörigen Endergebnisse. Dauerhafte Checkpoints speichern den Liga-Zustand.

Die neue Variante wird mit `--pair-pfsp` im Haupttrainer aktiviert. Historische
Trainingsskripte behalten ihre eigene Poolmethode; sie werden nicht automatisch
auf diese Liga umgestellt.
