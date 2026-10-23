# Self-Play-Prototyp: Ergebnisse vom 23. Oktober 2026

**Ergebnis:** Zwei unabhängig initialisierte neuronale Policies haben in kurzen Self-Play-Läufen messbar nützlicheres Verhalten gelernt. Verbesserungen erscheinen auf separaten Test-Seeds sowohl gegen feste untrainierte Referenzen als auch gegen trainierte Zwischenstände. Das erfüllt den ersten Lernnachweis für den Prototypen; es ist kein Nachweis hoher oder optimaler Spielstärke.

## Änderungen und technische Prüfung

Self-Play startet unmittelbar mit `checkpoint_0.pt`, dem zufälligen Anfangsmodell. Beide Gegner werden ausschließlich aus eingefrorenen neuronalen Zwischenständen dieses neuen Laufs ausgewählt. Es gibt keine Bot-Startphase und keinen stillen Ersatz bei Ladefehlern. Die Lernersitze sind über die Partien ausgeglichen. Gegnerpfade, SHA-256-Prüfsummen und Seeds stehen in Start- und Ergebnisprotokollen.

Die Evaluation verwendet ein festes Referenzpaar aus dem eigenen Anfangsmodell und einem unabhängig initialisierten, ausschließlich für Evaluation verwendeten Modell. Je Seedblock werden beide homogenen Gegnerpaare und beide gemischten Anordnungen über alle drei Lernersitze gespielt. Vor dem ersten Update wird das Anfangsmodell auf demselben Plan gemessen. Modellselektion verwendet den Sieganteil auf diesem unveränderten Plan.

75 Python-Tests und vier CTests bestanden. Der Integrationstest verbietet Bot-Erzeugung ausdrücklich. Zwei erste Pilotversuche deckten einen Simulatorfehler auf und wurden ohne Verwendung fehlerhafter Übergänge abgebrochen. Ein isolierter C++-Test reproduzierte den Fehler vor der Korrektur und bestand danach. Ursache und minimale Änderung sind unter ENG-006 im [Simulatorprotokoll](SIMULATOR_CHANGES.md) dokumentiert. Alle nachfolgenden Ergebnisse stammen aus frischen vollständigen Läufen mit der korrigierten Engine. Die 49 inventarisierten Simulator-/Build-/Testdateien wurden aus dem fixierten Upstream plus gespeichertem Patch bytegenau rekonstruiert.

## Trainingsumfang

Trainings-Seeds 7 und 19; je 8.192 Übergänge, zwei Umgebungen, Rollout-Horizont 128, 32 PPO-Updates, Snapshots alle 1.024 Übergänge. Netzwerkarchitektur v2, bestehende VP-Margin-Endbelohnung, keine Strategie-Zusatzbelohnungen und keine neuen Wunderregeln. Je Lauf endeten 390 Trainingspartien vollständig, jeweils 130 pro Lernersitz. Sämtliche 784 protokollierten Gegnerzuweisungen je Lauf waren `ppo_snapshot`; Startprotokolle enthalten auch die am Budgetende noch laufenden Partien.

Pro Lauf gab es vor dem Training sowie bei 4.096 und 8.192 Übergängen eine Evaluation mit 72 Partien auf sechs Seedblöcken. Beide Läufe liefen parallel; die Wandzeiten lagen einschließlich dieser Evaluationen bei ungefähr 187 beziehungsweise 190 Sekunden. Diese Zeiten sind keine isolierte Hardware-Durchsatzmessung und keine belastbare Prognose für beliebige längere Läufe.

## Separate Test-Seeds

Die fertigen `latest.pt`-Modelle bei Schritt 8.192 wurden vor der Holdout-Prüfung festgelegt. Die Modelle wurden anhand dieser Testergebnisse nicht erneut ausgewählt oder weitertrainiert. Tests verwendeten Seed-Familie 9001, getrennt von der Trainings-Evaluationsfamilie 42. Jeder Kandidat und sein untrainierter Ausgangsstand spielten je 144 Partien auf denselben zwölf Seedblöcken und Gegnerkombinationen.

| Trainings-Seed | Sieganteil vorher | Sieganteil nachher | Gewinn im VP-Vorsprung gegenüber dem stärksten Gegner | Näherungsweises 95%-Intervall des Gewinns |
| --- | ---: | ---: | ---: | --- |
| 7 | 37,8 % | 81,9 % | +13,28 Punkte | +9,85 bis +16,94 |
| 19 | 33,3 % | 90,3 % | +20,97 Punkte | +16,92 bis +24,99 |

Sieganteil teilt einen Sieg bei gleichen höchsten VP auf die beteiligten Spieler auf. Die Auswertung verwendet die bestehende VP-Zieldefinition, keine neu implementierte editionsspezifische Gleichstandsentscheidung. Der untrainierte Ausgangswert muss gegen ein zusätzliches, unabhängig initialisiertes Modell nicht exakt ein Drittel betragen.

## Gegen bereits trainierte Zwischenstände

Weitere getrennte Tests verwendeten die festen Modelle bei 2.048 und 4.096 Übergängen desselben Laufs als Gegner und Seed-Familie 19001. Wieder spielte sowohl der untrainierte Ausgangsstand als auch der fertige Kandidat je 144 Partien über zwölf Seedblöcke.

| Trainings-Seed | Sieganteil des untrainierten Modells | Sieganteil nach 8.192 Übergängen | VP-Vorsprung des fertigen Modells, absolut | Verbesserung gegenüber untrainiert |
| --- | ---: | ---: | ---: | ---: |
| 7 | 11,8 % | 52,8 % | +0,50 Punkte | +15,63 Punkte |
| 19 | 12,2 % | 66,0 % | +6,17 Punkte | +20,82 Punkte |

Die ungefähren 95%-Intervalle der VP-Verbesserung lagen bei +12,74 bis +18,74 beziehungsweise +18,43 bis +23,17 Punkten. Die Kandidaten schlagen damit nicht nur zufällige Anfangsmodelle; sie bestehen auch gegen frühere trainierte Strategien. Die Gegner stammen aber aus derselben Entwicklungslinie. Robustheit gegen unabhängig trainierte starke Gegner bleibt offen.

Alle 1.152 zusätzlichen Holdout-/Zwischenstands-Vergleichspartien wurden vollständig abgeschlossen. Mit den 432 regulären Evaluationspartien wurden insgesamt 1.584 vollständige Evaluationspartien ausgewertet. Technische Abbrüche der vorherigen Fehlersuche sind davon getrennt.

## Aussagekraft und nächster Lauf

Die Intervalle beruhen auf 2.000 Bootstrap-Stichproben ganzer Engine-Seedblöcke. Sitzrotationen und Gegnerpaarungen innerhalb eines Blocks zählen nicht als unabhängige Beobachtungen. Zwölf Blöcke liefern eine begrenzte, näherungsweise Unsicherheitsabschätzung. Zwei Trainings-Seeds und diese Gegnerauswahl reichen für einen positiven Prototypbefund, aber nicht für eine allgemeine Strategie- oder Konvergenzgarantie.

Ein moderater nächster Lauf mit 262.144 Übergängen ist im [Umsetzungsguide](SELFPLAY_POC.md) vorbereitet. Dort werden Evaluationen seltener durchgeführt und anschließend trainierte Modelle bei 65.536 und 131.072 Übergängen als feste Testgegner verwendet. Dieser moderate Lauf wurde hier noch nicht gestartet. Die Wunderregeln bleiben bis zur Entscheidung über weitere Investitionen unverändert. Auch exaktes Fortsetzen und langfristige Ligaoptimierung sind noch nicht implementiert.

Lokale Artefakte: `python/runs/selfplay-poc-20261023-fixed-seed7*` und `python/runs/selfplay-poc-20261023-fixed-seed19*`. Modelle und Einzelpartieprotokolle bleiben lokal. Die [kompakte JSON-Auswertung](SELFPLAY_POC_RESULTS_2026-10-23.json) hält Konfiguration, Kennzahlen und Referenzidentitäten ohne Modellgewichte im Projekt fest.
