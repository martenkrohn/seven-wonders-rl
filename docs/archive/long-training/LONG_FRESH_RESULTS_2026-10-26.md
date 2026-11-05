# Neue Seeds und Direktvergleich: Ergebnis

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Alle 4.032 Spiele sind vollständig und ohne Fehler/Abbrüche abgeschlossen:
1.152 Referenzspiele je Modell und 192 direkte Vergleichsspiele je Modell.
Alle Endzustände, Zeitalterfolgen und Aktionsprotokolle sind vorhanden;
keine inkonsistenten Punktsummen oder unbekannten gebauten Karten.
Der Seed-Audit meldet keine Überschneidung mit 145.849 zuvor gespeicherten Seeds.

## Referenztest: gleichbleibende Gegner, 32 neue Seed-Blöcke

| Modellstand | Sieganteil | Punkte Ø | Abstand zum stärksten Gegner Ø |
| --- | ---: | ---: | ---: |
| Start | 36,10% | 43,55 | −3,23 |
| 25% | 44,57% | 43,60 | −1,79 |
| Ende | **52,04%** | **48,91** | **+0,56** |

Gepaarte Verbesserungen, 95%-Bootstrapintervalle ganzer Engine-Seed-Blöcke:

- Ende gegen Start: +3,80 Abstandspunkte [+2,74; +4,78],
  +15,94 Prozentpunkte Sieganteil [+11,04; +20,78].
- Ende gegen 25%: +2,35 Abstandspunkte [+1,09; +3,62],
  +7,47 Prozentpunkte Sieganteil [+1,72; +13,19].
- 25% gegen Start: +1,45 Abstandspunkte [+0,62; +2,24].

Die vorherige kleine Kontrolle konnte 25% und Ende nicht unterscheiden.
Die größere frische Evaluation stützt nun das Endmodell als bessere Auswahl
unter diesen Bedingungen. Sie macht aus dem alten Nullbefund keinen Widerspruch:
anderer Spielplan und mehr unabhängige Seed-Blöcke verringern die Unsicherheit.

Ohne Wissenschaftsspezialist verbessert sich Ende gegen Start um +4,40
Abstandspunkte [+3,13; +5,57], Ende gegen 25% um +2,80 [+1,29; +4,20].
Nur gegen v1/v2-Referenzen beträgt der Zuwachs gegenüber Start +5,30
[+3,88; +6,67]. Das Ergebnis ist nicht allein durch Ausnutzen des
Wissenschaftsspezialisten erklärbar.

Mit Wissenschaftsspezialist ist weder Fortschritt noch Rückschritt klar:
Ende gegen Start +0,78 [−0,92; +2,44], Ende gegen 25% +0,09 [−1,53; +1,69].
Die im kleinen Kontrollpanel vermutete Verschlechterung bestätigt sich nicht.
Die beiden Wissenschaftler-Paarungen bleiben absolut schwierig: mittlerer
Abstand des Endmodells −2,99 bzw. −1,83. Dort gibt es weiter Verbesserungsbedarf.

## Direktvergleich: Start, 25% und Ende gemeinsam am Tisch

| Fokales Modell | Sieganteil | Punkte Ø | Abstand Ø |
| --- | ---: | ---: | ---: |
| Start | 20,83% | 43,48 | −7,88 |
| 25% | 26,82% | 45,49 | −4,84 |
| Ende | **53,56%** | **48,05** | **+0,03** |

Je Modell 32 Seed-Blöcke, beide Gegnerreihenfolgen, alle Sitze. Die Gegner
des jeweiligen fokalen Modells sind immer die beiden anderen Modellstände.
Damit gewinnt das Endmodell auch in direkter Konkurrenz am häufigsten.
Die drei Zeilen stammen aus getrennten fokalen Evaluationen mit jeweils
passenden RNG-/Kopfströmen. Es ist keine einzelne Turniertabelle, deren
Sieganteile zwingend exakt 100% ergeben. Es gibt auch keine gepaarten
Modellvergleiche mit identischen Gegnern in diesem Teiltest.

## Die Ressourcenverbesserung bestätigt sich

| Verhalten im Referenztest | Start | 25% | Ende |
| --- | ---: | ---: | ---: |
| Graue Bauten in I je Partie | 0,076 | 0,120 | **1,632** |
| Graue Bauten in II je Partie | 0,076 | 0,043 | **0,330** |
| III: Entscheidungen mit möglichem Kartenbau | 81,26% | 82,14% | **93,15%** |
| III: Abwürfe je Partie, insgesamt | 1,018 | 0,877 | **0,378** |
| III: Abwürfe ohne legalen Kartenbau je Partie | 0,969 | 0,846 | **0,302** |

Der späte zusätzliche Trainingsabschnitt liefert hier mehr als nur eine
andere Punkteverteilung: bessere Kartenbauverfügbarkeit und wesentlich
weniger spätes Abwerfen. Ohne legalen Kartenbau bedeutet weiterhin nicht,
dass auch kein Wunderbau möglich gewesen wäre.

Wissenschaftspunkte steigen von 8,64 auf 16,05, Gilden von 3,25 auf 6,57;
Militär liegt mit 7,65 deutlich über dem Start (2,86), etwas unter 25% (8,16).
Die Kombination von Ressourcen, Wissenschaft, Militär und Gilden ist erfolgreich.
Das beweist noch keine breite situationsabhängige Strategievielfalt pro Kopf.

## Entscheidung

Das Endmodell ist der beste derzeit geprüfte Ausgangspunkt für weitere Arbeit.
Den 25%-Stand als alternative Referenz behalten. Ein sofortiger weiterer
Millionenlauf ist nicht automatisch begründet: zuerst Verhalten nach
Wissenschaftskonkurrenz, Wunderseite und Kopf prüfen, idealerweise vorhandene
Spieldaten nutzen. Als nächste neue Spielprüfung wären bislang unbenutzte
Gegner sowie zwei Exemplare des Endmodells als Gegner sinnvoll, um starke
Wissenschaftskonkurrenz zu untersuchen. Das ist ein Vorschlag, noch nicht gestartet.

Diese Evaluation nutzt neue Seeds, aber vertraute Gegner, eine trainierte
Initialisierung und einen Trainingsseed. Informationsumfang, Gegnerpool und
Laufdauer wurden gemeinsam geändert; ihre einzelnen kausalen Beiträge bleiben
ungetrennt. Keine Aussage, dass sämtliche Strategien oder Wunder beherrscht werden.

Artefakte: `python/runs/long-fresh-seed130001-blocks32/summary.json`,
`exploration.json`, `analysis.json`, `dashboard.html` und die entsprechenden
Direktvergleichsdateien unter `direct/`. Keine neuen Spiele bei dieser Auswertung.
