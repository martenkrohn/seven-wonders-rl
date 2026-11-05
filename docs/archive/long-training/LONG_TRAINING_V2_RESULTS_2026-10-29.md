# Zweiter langer Trainingslauf: Ergebnis vom 29. Oktober 2026

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Der Lauf `long-mixed-wonders-v2-seed231` ist vollständig abgeschlossen:
2.097.152 zusätzliche Übergänge, 4.096 PPO-Updates, 99.864 abgeschlossene
Trainingspartien, keine Trainingstrunkierungen. Dauer 4 h 16 min 26 s,
136,3 Übergänge/s einschließlich Evaluationen. Alle fünf Kontrollpunkte enthalten
336 abgeschlossene Spiele ohne Fehler/Abbrüche, insgesamt 1.680 Evaluationsspiele.
Endzustände, Aktionsprotokolle und vollständige Zeitalterfolgen sind überall
vorhanden; keine inkonsistenten Punktsummen oder unbekannten gebauten Karten.

## Was gegenüber dem vorherigen Lauf geändert wurde

Startgewichte stimmen tensorweise exakt mit dem Endmodell des ersten langen
Laufs überein. Vier Köpfe, öffentliche Informationen, Standardreward und
Prior-Stärke 32 bleiben erhalten; sämtliche Prior-Gewichte sind am Ende
weiterhin unverändert. Neue Trainingsphase mit frischem Optimizer und Seed 231.

Der Trainingspool enthält ausschließlich eigene Modelle: tatsächlich gezogen
119.809 jüngere Snapshots, 39.959 Archivmitglieder und 39.976 feste Anker,
entsprechend etwa 60/20/20%. Das historische Archiv war von Beginn an gefüllt,
ohne Archiv-Fallbacks. 11.356 frühe Ziehungen aus der Gruppe jüngerer Snapshots
fielen erwartungsgemäß auf feste Anker zurück. Das Archiv griff sowohl auf
vorherige als auch neue Entwicklungsstände zurück. Spezialisten und ursprüngliche
Standardmodelle waren ausschließlich Evaluationsgegner.

Das Kontrollpanel hat 14 statt zuvor zwölf Paarungen: die bisherigen Referenzpaare
plus je zwei Exemplare des vorherigen Endmodells bzw. 25%-Stands. Andere
Kontrollseeds, deshalb frühere globale Sieganteile nicht direkt gegenüberstellen.
Alle Vergleiche unten verwenden Start und Zwischenstände dieses zweiten Laufs
auf seinem unveränderten Spielplan.

## Spielstärke im Kontrollpanel

| Fortschritt | Sieganteil | Eigene Punkte Ø | Abstand zum stärksten Gegner Ø |
| --- | ---: | ---: | ---: |
| Start, Ende des ersten langen Laufs | 47,77% | 48,51 | +0,03 |
| 25% | 47,12% | 47,59 | −0,38 |
| 50% | 54,02% | 49,76 | +0,51 |
| 75% | 45,98% | 49,38 | −0,82 |
| Ende | 53,72% | 50,47 | +1,17 |

Ende gegen Start: +1,14 Abstandspunkte, gespeichertes gepaartes
95%-Bootstrapintervall **[−0,72; +2,98]**; Sieganteil +5,95 Prozentpunkte
**[−2,38; +14,43]**. Die Punktschätzung ist positiv, der Test belegt aber noch
keinen verlässlichen zusätzlichen Gesamtfortschritt. Nur acht unabhängige
Engine-Seed-Blöcke, trotz 336 Spielen je Modell. 50% und Ende lassen sich ebenfalls
nicht klar unterscheiden: Ende gegenüber 50% +0,66 Abstandspunkte,
Intervall ungefähr [−1,25; +2,81]. Den 50%-Stand ebenfalls aufbewahren.

### Gegnerkonstellationen, Ende gegen Start

| Teilpanel | Spiele je Stand | Abstand Start → Ende | Veränderung |
| --- | ---: | ---: | ---: |
| Ohne Wissenschaftsspezialist | 288 | −0,19 → +0,85 | +1,04 |
| Mit Wissenschaftsspezialist | 48 | +1,33 → +3,08 | +1,75 |
| Nur ursprüngliche Standardmodelle | 48 | +1,21 → +4,94 | +3,73 |
| Mit Militärspezialist | 96 | −0,41 → +2,21 | +2,61 |
| Nur ältere v1/v2-Anker | 96 | +0,94 → −0,56 | −1,50 |
| Zwei Exemplare des vorherigen Endmodells | 24 | −4,67 → −3,50 | +1,17 |
| Zwei Exemplare des vorherigen 25%-Stands | 24 | −2,12 → −2,71 | −0,58 |

Der Gewinn ist ungleich verteilt. Gerade gegen die stärkeren eigenen Vorgänger
ist noch kein klarer Durchbruch sichtbar. Die Teilgruppen sind klein und
explorativ; aus einzelnen negativen Paarungen keinen allgemeinen Rückschritt
ableiten. Auch bei einem Modell gegen zwei identische eingefrorene Exemplare
ist der mittlere Abstand zum stärksten Gegner nicht notwendigerweise null.

Zusätzlich berechnete Untergruppenintervalle verwenden 10.000 Bootstrapresamples;
die gespeicherten Hauptintervalle verwenden 2.000. Kleine Unterschiede ihrer
Grenzen sind Monte-Carlo-Abweichungen. Keine Korrektur für mehrere Vergleiche.

## Ressourcenplanung und Strategie

| Verhalten bzw. Punkte | Start | Ende |
| --- | ---: | ---: |
| Graue Bauten in I je Partie | 1,57 | 1,44 |
| Graue Bauten in II je Partie | 0,31 | 0,09 |
| III: Entscheidungen mit legalem Kartenbau | 94,47% | 94,61% |
| III: Abwürfe je Partie, insgesamt | 0,34 | 0,23 |
| III: Abwürfe ohne legalen Kartenbau je Partie | 0,26 | 0,19 |
| I: Abwürfe je Partie | 0,27 | 0,05 |
| Wissenschaftspunkte Ø | 16,29 | 21,44 |
| Blaue Punkte Ø | 9,18 | 15,41 |
| Militärpunkte Ø | 7,88 | 2,42 |
| Gildenpunkte Ø | 5,51 | 2,73 |
| Wunderpunkte Ø | 5,30 | 3,92 |

Die frühere Ressourcenverbesserung geht nicht verloren. Die III-Verfügbarkeit
ist praktisch unverändert auf hohem Niveau, spätes Abwerfen nimmt weiter ab.
Weniger graue Bauten sind daher für sich kein Rückschritt. Auch die zuvor fast
vollständig gemiedenen blauen/gelben Karten in I werden wieder öfter gebaut:
158 blaue und 166 gelbe Bauten statt 25 bzw. zwölf, jeweils in 336 Partien.

Das Modell verschiebt seine Punktequellen stark zu Wissenschaft und Blau.
Das ist eine andere Gewichtung; mehr Wissenschaft allein beweist weder höhere
Gesamtstärke noch mehr situationsabhängige Strategievielfalt. Dass Wissenschaft
ohne expliziten Spezialisten im Trainingspool wächst, zeigt immerhin, dass der
Spezialist für dieses gelernte Verhalten nicht erforderlich blieb.

Die vier Köpfe erhielten 523.132 / 524.876 / 530.569 / 518.575 Übergänge,
also ausgeglichene Trainingsbudgets. Ihre gemessene JS-Divergenz auf denselben
Rolloutzuständen sank im Mittel der ersten/letzten 100 Updates von 0,073 auf
0,047; Greedy-Abweichung zu Kopf 0 von 7,3% auf 5,2% (einschließlich Kopf 0
selbst). Das spricht für ähnlichere Entscheidungen auf diesen Zuständen,
beweist aber keinen vollständigen Kollaps der Köpfe. Zustandsverteilungen
ändern sich während des Trainings mit. Einzelkopf-Sieganteile aus ungleichen
Teilplänen eignen sich nicht zur zuverlässigen Auswahl eines besten Kopfes.

## Einschätzung und nächster Schritt

Technisch sauberer Lauf, erhaltene Ressourcenplanung und positiver Trend im
Gesamtergebnis. Die Größenordnung und Sicherheit des Zugewinns sind deutlich
geringer als nach dem ersten langen Lauf. Daraus folgt weder ein bewiesenes
Plateau noch ein gesicherter Vorteil des neuen 60/20/20-Pools.

Vor einem weiteren langen Lauf Start, 50%-Stand und Ende auf mehr neuen
Seed-Blöcken vergleichen. Ein Referenztest mit 32 Blöcken, denselben 14 Paarungen
und allen Sitzen wären 1.344 Spiele je Stand, insgesamt 4.032. Zusätzlich kann
ein direkter Tischvergleich dieser drei Stände klären, ob sich ihre geänderten
Spielweisen gegenseitig schlagen. Bisher keine neue Evaluation oder Training
für diese Analyse gestartet.

Artefakte: `python/runs/long-mixed-wonders-v2-seed231/dashboard.html` und
`post_training_analysis.json`. Eine Kopie der ergänzenden Kennzahlen liegt unter
[LONG_TRAINING_V2_RESULTS_2026-10-29.json](LONG_TRAINING_V2_RESULTS_2026-10-29.json).
