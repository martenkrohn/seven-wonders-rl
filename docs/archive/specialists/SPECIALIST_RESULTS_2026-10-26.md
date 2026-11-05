# Spezialisten gegen eingefrorene Standardgegner: Auswertung vom 26. Oktober 2026

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

**Ergebnis:** Der Wissenschaftsspezialist hat auf dem gemeinsamen Holdout eine
zuverlässige, siegfähige Wissenschaftsstrategie entwickelt. Der Militärspezialist
erhöht seine Militärpunkte, wird insgesamt aber schwächer. Den Wissenschaftler
würde ich als Kandidaten für einen vielfältigeren Gegnerpool behalten; das finale
Militärmodell erfüllt das bisherige Ziel eines spielstarken Spezialisten noch
nicht. Eine Aufnahme oder ein neuer Trainingslauf wurde hier nicht durchgeführt.

## Was tatsächlich gelaufen ist

Beide Modelle starten von denselben Gewichten des normalen Standardmodells
`science-diagnostic-v1-baseline-seed31/latest.pt`, nach dessen 524.288 ursprünglichen
Übergängen. Je Spezialist folgten 524.288 zusätzliche Übergänge mit Seed 31,
acht Umgebungen und Horizont 64. Bonusgewicht 1, Reward-Skalierung 10:
Wissenschaft `(VP-Abstand + Wissenschafts-VP)/10`, Militär
`(VP-Abstand + Netto-Militär-VP)/10`. Gewertet wurde erst beim echten Spielende.

Beide trainierten ausschließlich gegen zwei eingefrorene Kopien des normalen
Standardmodells. Eigene Zwischenstände wurden gespeichert und nie als Gegner
ausgewählt. Alle protokollierten Gegnerpfade/-Hashes entsprechen der vorgesehenen
Kopie. Der Startmodell-/Gegnerhash lautet
`c81d68ebdf9ec7f4f3cd21ca35014809b612872a0a65ac202818caa81a5fd302`.

| Lauf | Übergänge | PPO-Updates | Abgeschlossene Trainingsspiele | Laufzeit inkl. Trainingsevaluation |
| --- | ---: | ---: | ---: | ---: |
| Wissenschaft | 524.288 | 1.024 | 24.960 | 35 min 30 s |
| Militär | 524.288 | 1.024 | 24.960 | 34 min 1 s |

Je Lauf wurden zusätzlich acht Trainingsevaluationen mit insgesamt 288 gültigen
Spielen ausgeführt. Keine protokollierte Episode ist abgebrochen, kein Update hat
nichtendliche numerische Kennzahlen. Die acht am Budgetende schon gestarteten,
noch laufenden Spiele pro Lauf gehören nicht zu den 24.960 abgeschlossenen
Partien und erhalten keinen nachträglichen Reward. Alle drei Lernerpositionen
wurden praktisch gleich häufig bespielt.

## Gemeinsamer Holdout ohne Trainingsbonus

Je Ausgangs-/Endmodell wurden 480 Spiele auf demselben Plan gespielt:
40 Engine-Seed-Blöcke, drei Sitze und vier geordnete Gegnerpaare. Gegner sind die
normalen Checkpoints 131.072 und 262.144 derselben Standardlinie, einschließlich
beider gemischter Reihenfolgen. Sie unterscheiden sich vom direkten
Trainingsgegner bei Schritt 524.288, sind aber keine unabhängige Modellfamilie.
Alle Modelle spielen stochastisch, ohne erzwungene Kartenwahl. Test-Seed 79001;
kein gespeicherter Holdout-Engine-Seed überschneidet sich mit den neuen
Spezialisten-Trainingsspielen.

Die beiden Auswertungen enthalten je Ausgangsmodell und Spezialist: insgesamt
1.920 vollständige Datensätze, davon 960 für die beiden Spezialisten. Die zweimal
ausgewertete Ausgangspolicy liefert identische Spielverläufe und Ergebnisse;
nur Laufzeiten und die dateibezogenen Checkpoint-IDs unterscheiden sich. Diese
Wiederholung ist **keine** zweite unabhängige Baseline-Stichprobe.

| Kennzahl je Spiel | Ausgangsmodell | Wissenschaftsspezialist | Militärspezialist |
| --- | ---: | ---: | ---: |
| Sieganteil, geteilte Siege anteilig | 41,56 % | **73,75 %** | **31,04 %** |
| Strikte Siege | 40,21 % | 72,29 % | 29,58 % |
| Eigene Gesamt-VP | 39,52 | **57,05** | 37,01 |
| VP-Abstand zum stärksten Gegner | −2,62 | **+5,85** | −5,04 |
| Wissenschafts-VP | 3,84 | **50,39** | 5,38 |
| Militär-VP | 5,58 | −5,94 | **9,40** |
| Blaue VP | 14,65 | 4,27 | 7,98 |
| Wunder-VP | 5,88 | 0,73 | 5,05 |
| Abgeworfene Karten | 2,47 | 1,27 | 4,13 |

Alle Punkte und Sieganteile hier sind echte Engine-Ergebnisse ohne Zusatzreward.
Ein Sieganteil unter einem Drittel ist bei unterschiedlichen Policies nicht an sich
ein formaler Misserfolgstest; maßgeblich ist hier auch der gepaarte Rückgang zum
Ausgangspunkt auf demselben Plan.

Die Rohdaten wurden unabhängig vom fertig gerenderten Bericht geprüft:
Gesamt-VP ergeben sich aus den Kategorien; VP-Abstand, strikte Siege und geteilte
Siege wurden neu berechnet. Die Militärscores jedes ausgewerteten Lerners wurden
anhand der Schildvergleiche der drei Zeitalter und Gewichte 1/3/5 nachgerechnet;
Wissenschaftsscores anhand gebauter Karten, Wunder/Joker und der derzeitigen
Engine-Auswahlregel. Es gab dabei keine Abweichung. Endzustände, alle drei
Zeitalter und Aktionsaufzeichnungen sind durchgehend vorhanden.

## Gepaarte Unterschiede zum Ausgangsmodell

Explorative 95%-Intervalle aus 5.000 Bootstrap-Stichproben **ganzer
Engine-Seed-Blöcke**, nicht einzelner Sitzrotationen. Die vorhandenen Reports
verwenden 2.000 Stichproben und liefern entsprechend geringfügig andere Grenzen.

| Unterschied | Wissenschaft: Mittel [95%-Intervall] | Militär: Mittel [95%-Intervall] |
| --- | --- | --- |
| Sieganteil, Prozentpunkte | **+32,19 [+22,08; +41,15]** | **−10,52 [−17,50; −3,33]** |
| VP-Abstand | +8,47 [+5,97; +10,80] | −2,42 [−3,70; −1,14] |
| Eigene VP | +17,53 [+15,28; +19,65] | −2,50 [−3,47; −1,54] |
| Wissenschafts-VP | +46,55 [+43,90; +49,06] | +1,54 [+0,65; +2,46] |
| Militär-VP | −11,52 [−12,25; −10,81] | +3,82 [+2,85; +4,78] |

Die Intervalle beschreiben Variation der Testspiele, nicht Unsicherheit über
Trainingsseeds. Beide Spezialisierungen wurden bislang nur mit einem Trainingsseed
untersucht. Es fehlt außerdem ein gleich lange weitertrainierter Kontrollspieler
mit Standardreward und denselben eingefrorenen Gegnern. Die Beobachtung einer
Spezialisierung ist klar; der isolierte kausale Anteil des Bonus gegenüber
zusätzlichem Training und verändertem Gegnerplan ist damit nicht bestimmt.

## Wissenschaft: Vorbereitung und hoher Ertrag

Der Spezialist erreicht mindestens 21 Wissenschafts-VP in jedem der 480 Spiele;
in 467 Spielen (97,3 %) mindestens 30 und in 282 Spielen mindestens 50.
Median 52, Maximum 85 Wissenschafts-VP. Das Ausgangsmodell erreicht nur in
6 Spielen mindestens 20 und niemals 30.

Die Aktionsfolgen zeigen Vorbereitung: Im ersten Zeitalter baut er im Mittel
2,49 graue Manufakturkarten und 1,96 grüne Karten; beim Ausgangsmodell sind es
0,07 und 0,27. Im zweiten/dritten Zeitalter folgen 3,29/4,18 grüne Karten.
Die Press wird in 476 der 480 Spiele gebaut. Das ist nicht bloß ein hoher
Terminalbonus, sondern zeigt eine wiederholbare Ressourcen- und Baureihenfolge
unter den getesteten Bedingungen.

Das Modell investiert kaum in Militär oder Wunder: 473 Spiele enden ohne eigene
Schilde, 454 mit dem maximalen Nettoverlust −6 Militär-VP. Das ist hier mit
starker Gesamtleistung vereinbar, da sich die Niederlagenpunkte begrenzen,
während Wissenschaft viele Punkte liefert. Ob diese Ausrichtung gegen Gegner
mit Wissenschaftskonkurrenz bzw. gezieltem Kartenentzug trägt, ist noch offen.

Die Verbesserung beschränkt sich nicht auf eine einzelne Position oder ein
Gegnerpaar. Sieganteile über die drei Sitze: 79,4 %, 71,3 %, 70,6 %; über die
vier Gegnerpaare: 80,4 %, 72,9 %, 71,3 %, 70,4 %. Die Untergruppen teilen
Seed-Blöcke und sind keine unabhängigen weiteren Experimente.

Auch gegen den neueren direkten Trainingsgegner zeigt die letzte
Trainingsevaluation 56,9 % Sieganteil, +1,61 VP-Abstand und 44,28 Wissenschafts-VP.
Diese Messung hat nur 36 Spiele und einen anderen Gegnerplan als der Holdout;
die absoluten Werte sind nicht direkt mit 73,75 % vergleichbar. Die erste
Trainingsevaluation bei 65.536 Schritten lag noch bei 27,3 % Sieganteil und
−4,47 VP-Abstand. Der spätere Verlauf zeigt Lernen mit Schwankungen.

## Militär: Spezialisierung, aber zu hoher Preis

Die Militärpunkte steigen von 5,58 auf 9,40, mit positiven Militärpunkten in
86,0 % der Spiele und dem Maximum +18 in 49 Spielen. Der Aufwand führt jedoch
nicht zu stärkerem Gesamtspiel: Der mittlere VP-Abstand sinkt um 2,42 Punkte,
der Sieganteil um 10,52 Prozentpunkte. Der Rückgang zeigt sich über alle Sitze
und alle vier Gegnerpaare.

Der Zugewinn von 3,82 Militär-VP wird unter anderem von 6,68 verlorenen blauen VP
überkompensiert. Der Spezialist wirft zudem durchschnittlich 4,13 statt 2,47 Karten
ab; allein im ersten Zeitalter sind es 1,45 statt 0,41. Die Daten zeigen diesen
Zusammenhang, beweisen aber noch nicht die einzelne Ursache der Verschlechterung.
Denkbare Ansatzpunkte sind die Balance aus militärischem Vorteil,
Ressourcenplanung und alternativen Punktquellen.

Der Holdout-Trainingszielwert ohne Skalierung, `VP-Abstand + Militär-VP`, steigt
von 2,96 auf 4,36, obwohl das eigentliche Spiel schlechter wird. Das illustriert
den beabsichtigten Zielkonflikt des Bonus. Daraus folgt weder, dass stärkerer
Militärbonus automatisch hilft, noch dass mehr Training allein das Problem löst.
Die acht Trainingsevaluationen liefern bislang keinen klaren Durchbruch: am Ende
22,2 % Sieganteil und −7,22 VP-Abstand gegen den direkten Trainingsgegner.
Die kleinen 36-Spiele-Messungen rechtfertigen keine sichere Rangfolge der
Zwischenstände; ein früherer Zwischenstand müsste auf einem gemeinsamen
separaten Plan geprüft werden.

## Zusätzlich gefundene Simulatorgrenze

Die Wissenschaftsjoker werden im vorhandenen C++-Simulator gierig verteilt und
nicht immer punktoptimal zugewiesen. Die gespeicherten Wissenschaftsscores
entsprechen dieser implementierten Regel vollständig. Eine zusätzliche exhaustive
Jokerverteilung ergibt beim Wissenschaftler in 52/480 Spielen 1–3 weitere
mögliche Wissenschaftspunkte (im Mittel +0,28 VP über alle Spiele), beim
Militärmodell in 11/480 Spielen (im Mittel +0,06). Bereits das Ausgangsmodell ist
in 6/480 Spielen betroffen. Beispielsweise enthält Wissenschaftsspiel-ID 4 einen
Engine-Science-Score von 52 statt des optimal erreichbaren 55.

Das ist eine geerbte Bewertungsgrenze, keine Inflation durch den neuen Reward.
Sie wurde hier nicht geändert; alle berichteten Spielausgänge beziehen sich auf
dieselbe aktuelle Engine. Ein Fix sollte separat mit Regressionstests und erneuter
Evaluation untersucht werden. Aus den theoretischen Zusatzpunkten wird hier
kein nachträglicher Sieganteil berechnet, da auch Gegnerwerte und Entscheidungen
betroffen sein können.

## Empfohlener nächster Schritt

1. Den Wissenschaftler als **Kandidaten** behalten und gegen normale Modelle aus
   anderen Trainingslinien, Wissenschaftskonkurrenz und gemischte Gegnerpaare
   prüfen. Erst danach den Umfang seines Einsatzes im Generalistenpool festlegen.
2. Das finale Militärmodell noch nicht als verifizierten starken Spezialisten
   einstufen. Einen gleich lange weitertrainierten Standardreward-Kontrolllauf
   und eine schwächere Zusatzgewichtung, etwa 0,5, unter demselben Setup vergleichen.
   0,5 ist eine zu prüfende Einstellung, keine bereits belegte Verbesserung.
3. Erfolgreiche Einstellungen über weitere Trainingsseeds wiederholen. Erst danach
   den Generalisten gegen die gewählte diverse Mischung trainieren.
4. Zusätzliche Deep-Exploration-Mechanismen bleiben vorerst nachrangig: Die
   Wissenschaftslinie zeigt bereits, dass der bestehende PPO-Pfad mit geändertem
   Lernziel die hier gewünschte längerfristige Vorbereitung lernen kann. Das ist
   kein allgemeiner Nachweis, dass Explorationsprobleme gelöst sind.

## Daten und Wiederholung

- [Wissenschafts-Dashboard](../../../python/runs/specialists-v2-frozen-science-seed31-holdout/dashboard.html)
- [Militär-Dashboard](../../../python/runs/specialists-v2-frozen-military-seed31-holdout/dashboard.html)
- [Audit und Einzelkennzahlen](../../../python/runs/specialists-v2-analysis-20261026/analysis.json)
- [Ausführbare Rohdatenprüfung](../../../python/runs/specialists-v2-analysis-20261026/analyze.py)

Aus `python/` wiederholbar mit
`.venv/bin/python runs/specialists-v2-analysis-20261026/analyze.py`.
Die Prüfung spielt keine neuen Partien und ändert keine Trainings-/Holdout-Dateien.
Alle bestehenden Modelle wurden unverändert gelassen. Eine kompakte dauerhafte
Messaufzeichnung liegt in `SPECIALIST_RESULTS_2026-10-26.json` neben diesem Bericht.
