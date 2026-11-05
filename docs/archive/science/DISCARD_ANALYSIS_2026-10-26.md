# Abwurfverhalten trainierter Modelle nach Zeitalter

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Auswertung vom 26. Oktober 2026. Analysiert wurden die sieben länger trainierten
Endmodelle mit vorhandenen vollständigen Holdout-Aktionsaufzeichnungen sowie
sechs Zwischenstände. Das normale Ausgangsmodell wird zusätzlich auf dem
Spezialisten-Testplan ausgewertet. Insgesamt 14 Modell-/Testplan-Kombinationen
mit je 480 vollständigen Spielen: 6.720 Datensätze. Zwei identische
Ausgangsmodell-Auswertungen der Spezialisten werden nur einmal gezählt.

**Hauptbefund:** Bei den normalen Endmodellen fallen rund 60–68 % der Abwürfe
im dritten Zeitalter an. Der Wissenschaftsspezialist verwertet dagegen im
letzten Zeitalter fast alle Entscheidungen für andere Aktionen. Beim Militärmodell
fallen zusätzlich schon im ersten Zeitalter viele Abwürfe auf.

## Gezählt wird eine tatsächlich ausgeführte Abwurfentscheidung

Eine Karte für drei Münzen abwerfen (`action=discard`, `execution=applied`)
ist hier ein Abwurf. Wunderbau zählt nicht als Abwurf; ungespielte letzte Karten,
übersprungene siebte Karten und bloße Protokollschritte zählen nicht. Ersetzungen
durch die Engine (`fallback_discard`) würden separat ausgewiesen; in diesen
Datensätzen gab es keine. Die Rohdatenzahlen stimmen bei jedem Modell exakt mit
den bisherigen Gesamt- und Zeitalter-Aggregaten überein.

Die folgenden Werte sind **durchschnittlich abgeworfene Karten je Partie** im
jeweiligen Zeitalter. Die Gesamtspalte ist die Summe der ungerundeten Werte.
Pro Zeitalter stehen gewöhnlich sechs tatsächlich ausgeführte Aktionen zur
Verfügung; manche Wunder erlauben eine siebte. Prozentquoten weiter unten
verwenden deshalb die echte Zahl ausgeführter Aktionen, nicht 21 Protokollticks.

## Die vier 524.288er-Varianten auf demselben Plan

Jeweils 480 Spiele, 40 Engine-Seed-Blöcke, Test-Seed 59001. Identische Gegner,
Sitze und Seeds für diese vier Varianten. Die Namen Entropie/Lambda bezeichnen
PPO-Parameter, keine zusätzlichen Spezialisten-Rewards.

| Endmodell | Zeitalter I | Zeitalter II | Zeitalter III | Insgesamt | Anteil aller Abwürfe in III |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard: Entropie 0,01 / Lambda 0,95 | 0,38 | 0,55 | 1,56 | 2,49 | 62,7 % |
| Höhere Entropie: 0,03 / Lambda 0,95 | 0,42 | 0,51 | 1,53 | 2,46 | 62,2 % |
| Lambda 1,0 / Entropie 0,01 | 0,24 | 0,43 | 1,38 | 2,05 | 67,5 % |
| Kombination: Entropie 0,03 / Lambda 1,0 | 0,61 | 0,37 | 1,49 | 2,46 | 60,4 % |

Bei diesen Modellen werden in Zeitalter III ungefähr 23–26 % aller ausgeführten
Aktionen zum Abwerfen verwendet; 70–74 % der Partien enthalten mindestens einen
Abwurf in diesem Zeitalter. Weniger Abwerfen bedeutet allein noch keine höhere
Spielstärke; beispielsweise war Lambda 1,0 im früheren gemeinsamen
Spielstärkevergleich nicht überzeugender als die Standardkonfiguration.

## Spezialisten und ihr Ausgangsmodell auf demselben Plan

Jeweils 480 Spiele, 40 Seed-Blöcke, Test-Seed 79001. Alle drei Modelle verwenden
denselben Testplan mit den normalen Checkpoints 131.072 und 262.144. Der
Test-Seed unterscheidet sich von der obigen Vierergruppe; die Ausgangspolicy
ist dieselbe wie das dortige Standard-Endmodell.

| Modell | Zeitalter I | Zeitalter II | Zeitalter III | Insgesamt | Anteil aller Abwürfe in III |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vortrainiertes Standard-Ausgangsmodell | 0,41 | 0,58 | 1,48 | 2,47 | 60,0 % |
| Wissenschaftsspezialist | 0,48 | 0,52 | **0,27** | **1,27** | 21,5 % |
| Militärspezialist | **1,45** | 0,84 | **1,84** | **4,13** | 44,6 % |

Beim Wissenschaftler sind es im letzten Zeitalter nur 4,5 % der Aktionen;
24,4 % der Partien enthalten dort überhaupt einen Abwurf. Sein Abwurfrückgang
gegenüber dem Ausgangsmodell entsteht fast vollständig im dritten Zeitalter:
−1,21 Karten je Partie. Früh wirft er nicht weniger ab. Dieses Muster passt zu
der zuvor beobachteten Vorbereitung für viele späte Wissenschaftsbauten.

Das Militärmodell wirft ungefähr 24,2 % seiner Aktionen in Zeitalter I und
30,6 % in Zeitalter III ab. Schon im ersten Zeitalter enthalten 90,6 % seiner
Partien mindestens einen Abwurf. Gegenüber dem Ausgangsmodell beträgt der
Zuwachs insgesamt +1,66 Abwürfe, davon +1,04 allein im ersten Zeitalter.
Nur 5/480 Militärpartien kommen ganz ohne Abwurf aus, beim Wissenschaftler
111/480 und beim Ausgangsmodell 69/480.

Die explorativen, über ganze Seed-Blöcke gebootstrappten 95%-Intervalle für
Abwürfe je Partie bestätigen die großen Unterschiede auf diesem Plan:
Militär I 1,34–1,56 gegenüber Ausgangsmodell I 0,33–0,49;
Wissenschaft III 0,20–0,36 gegenüber Ausgangsmodell III 1,32–1,65.
Diese absoluten Intervalle sind kein gepaarter Effektgrößentest und erfassen
keine Variation über Trainingsseeds.

## Der frühere moderate Lauf mit 262.144 Übergängen

Dieser Lauf wurde gegen andere Referenzen geprüft. Sein Test-Seed ist zwar
ebenfalls 59001, der Gegnerplan ist aber nicht derselbe wie beim Vierervergleich.
Er darf nicht als direkter Vergleich zum 262.144er-Zwischenstand der späteren
Standardlinie interpretiert werden.

| Endmodell | Zeitalter I | Zeitalter II | Zeitalter III | Insgesamt |
| --- | ---: | ---: | ---: | ---: |
| Moderater v3-Lauf, 262.144 | 0,32 | 0,49 | 1,26 | 2,06 |

Auch hier liegen 61,0 % aller Abwürfe im letzten Zeitalter, und dort enthalten
68,5 % der Partien mindestens einen Abwurf.

## Waren die Abwürfe wegen fehlender Baumöglichkeiten nötig?

Die Aktionsaufzeichnung enthält die vor der Entscheidung legal baubaren Karten
(`legal_build_cards`). Darin sind normale und freie Kartenbauten enthalten.
Sie enthält nicht sämtliche Wunderbaumöglichkeiten. Deshalb unterscheiden wir
zwischen einer vorhandenen Kartenbaualternative und dem Fehlen einer solchen;
Letzteres ist **kein** Nachweis, dass keine andere Aktion möglich war.

| Modell | Abwürfe I mit legaler Kartenbaualternative | Abwürfe III mit legaler Kartenbaualternative | Münzen vor einem Abwurf III, Ø |
| --- | ---: | ---: | ---: |
| Standard, Test-Seed 59001 | 85,0 % | 4,3 % | 12,8 |
| Höhere Entropie | 84,1 % | 8,7 % | 14,1 |
| Lambda 1,0 | 79,6 % | 16,1 % | 13,6 |
| Kombination | 91,1 % | 17,4 % | 14,3 |
| Moderater v3-Lauf | 82,9 % | 15,6 % | 14,8 |
| Standard, Spezialisten-Testplan | 93,9 % | 4,1 % | 13,4 |
| Wissenschaftsspezialist | 100,0 % | 77,9 % | 7,8 |
| Militärspezialist | **95,1 %** | 14,6 % | **16,7** |

**Früh sind Abwürfe oft eine Policy-Entscheidung trotz legaler Karten.** Beim
Militärmodell war in 66,7 % seiner Zeitalter-I-Abwürfe sogar die konkret
abgeworfene Karte selbst baubar. Von 697 frühen Abwürfen entfallen 292 auf graue
Manufakturen und 238 auf grüne Wissenschaftskarten. Das liefert einen konkreten
Ansatz für weitere Prüfung seiner Ressourcen- und Alternativplanung.
Es beweist nicht, dass jede dieser Entscheidungen falsch war: Münzengewinn,
späterer Nutzen und Kartenentzug können einen Abwurf rechtfertigen.

**Späte Abwürfe der normalen Modelle entstehen überwiegend in Zuständen ohne
legalen Kartenbau.** Beim Standardmodell auf Seed 59001 betrifft das 717 von
749 Abwürfen im letzten Zeitalter. Trotzdem hatte es dabei im Mittel 12,8 Münzen;
in 462 Fällen schon mindestens zehn. Reine Münzknappheit erklärt das Muster
also nicht. Nicht ausreichende passende Produktion/Handelsmöglichkeiten oder
bereits gebaute Karten sind mögliche Ursachen; welche davon im Einzelfall gilt,
lässt sich aus diesen kompakten Ereignissen nicht vollständig rekonstruieren.

Beim Wissenschaftler ist der Mechanismus anders: Er hat bei seinen seltenen
späten Abwürfen oft noch Baualternativen. Fast alle Wissenschaftskarten werden
gebaut statt abgeworfen; in Zeitalter III gibt es nur einen Abwurf einer grünen
Karte unter 480 Partien. Das Militärmodell wirft dort dagegen 361 grüne und
292 violette Karten ab. Die unterschiedliche Strategie ist damit auch in der
Farbauswahl sichtbar.

## In welchen Runden eines Zeitalters?

Die Abwürfe konzentrieren sich meist auf die letzten beiden normalen Runden.
Beispiel Standard, Test-Seed 59001: 491 der 749 späten Abwürfe (65,6 %) liegen
in Runde 5 oder 6. Beim Militärmodell sind es in Zeitalter I 546 von 697
(78,3 %). Das passt dazu, dass die Auswahl in der weitergereichten Hand zum
Zeitalterende kleiner wird; eine kausale Erklärung jeder Entscheidung ist es nicht.
Sonderfähigkeiten können zusätzlich Runde 7 erzeugen; übersprungene Aktionen
wurden nicht als Abwürfe mitgezählt.

## Entwicklung innerhalb der Trainingslinien

Alle folgenden Zwischen-/Endstände sind innerhalb der jeweiligen Zeile auf
demselben festen Testplan ausgewertet. Angegeben sind **I / II / III** je Partie.

| Linie | Früherer Stand | Endstand |
| --- | --- | --- |
| Standard, 262.144 → 524.288 | 0,38 / 0,63 / 1,68 | 0,38 / 0,55 / 1,56 |
| Entropie, 262.144 → 524.288 | 0,60 / 0,87 / 1,85 | 0,42 / 0,51 / 1,53 |
| Lambda 1,0, 262.144 → 524.288 | 0,24 / 0,28 / 1,48 | 0,24 / 0,43 / 1,38 |
| Kombination, 262.144 → 524.288 | 0,38 / 0,50 / 1,41 | 0,61 / 0,37 / 1,49 |
| Moderater Lauf, 65.536 → 262.144 | 0,60 / 0,48 / 1,58 | 0,32 / 0,49 / 1,26 |

Mehr Training senkt Abwürfe also nicht durchgängig. Beispielsweise wirft die
Kombinationsvariante am Ende häufiger früh ab; Lambda 1,0 verschiebt Abwürfe
teilweise vom letzten ins zweite Zeitalter.

## Einordnung und Daten

Für das Militärmodell würde ich als Nächstes die **frühen Abwürfe** gezielt mit
Ressourcenaufbau, möglichen Bauten und späteren Handproblemen untersuchen.
Für die normalen Modelle ist eher die Frage interessant, warum im letzten
Zeitalter trotz Münzen so wenige Karten baubar sind. Ein pauschaler Abwurfmalus
würde diese Situationen nicht automatisch lösen und könnte sinnvolle
Münzbeschaffung oder strategischen Kartenentzug unterdrücken.

Keine neuen Spiele oder Trainings wurden durchgeführt. Die Zahlen beschreiben
gefrorene Modelle im Holdout, nicht sämtliche Entscheidungen während ihres
Lernverlaufs. Nur vorhandene lange Läufe mit vollständiger Detailauswertung
wurden berücksichtigt. Kurze Performance-/Setup-Läufe, untrainierte Modelle und
erzwungene Grün-Prioritätsvarianten sind keine zusätzlichen trainierten Policies
in dieser Hauptauswertung. Wiederholte Seeds/Sitze sind korreliert; ein einzelner
Trainingsseed pro Linie begrenzt die Verallgemeinerung.

- Lokale vollständige Ergebnisse: `python/runs/discard-analysis-20261026/analysis.json`.
- Tabellenexport: `python/runs/discard-analysis-20261026/by_age.csv`.
- Kompakte dauerhafte Aufzeichnung: `DISCARD_ANALYSIS_2026-10-26.json` neben diesem Bericht.
- Wiederverwendbarer Offline-Analyzer: `python/scripts/ppo_training/analyze_discards.py`.

Der Analyzer liest die `summary.json`, `analysis.json` und zugehörigen JSONL-Dateien
der angegebenen Auswertungsverzeichnisse und prüft seine Zählungen gegen die
vorhandenen Aggregate. Aufruf aus `python/`:

```sh
.venv/bin/python scripts/ppo_training/analyze_discards.py \
  runs/science-common-holdout-pTfpDb/baseline \
  runs/science-common-holdout-pTfpDb/entropy \
  runs/science-common-holdout-pTfpDb/credit \
  runs/science-common-holdout-pTfpDb/combined \
  runs/poc-moderate-features-v3-seed31-restart1-detailed \
  runs/specialists-v2-frozen-science-seed31-holdout \
  runs/specialists-v2-frozen-military-seed31-holdout \
  --out-dir runs/discard-analysis-repeat
```
