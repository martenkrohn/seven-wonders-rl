# Gemeinsame Evaluation der vier Wissenschaftsvarianten

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Ausgewertet am 25. Oktober 2026. Lokales Ergebnisverzeichnis:
`python/runs/science-common-holdout-pTfpDb/`.

**Ergebnis:** Standard und erhöhte Entropie zeigen gegenüber dem Halbzeitmodell
deutlich mehr Wissenschaft. Ihre Endmodelle unterscheiden sich auf dem
gemeinsamen Testplan nicht eindeutig in der Spielstärke. Lambda 1,0 liefert in
diesem Experiment deutlich weniger Wissenschaft und keinen überzeugenden Grund
für einen Wechsel der bisherigen Standardeinstellung. Eine zuverlässig starke
Wissenschaftsstrategie ist auch bei den beiden besseren Varianten nicht belegt.

## Gemeinsamer Testplan und Prüfung

Jede Variante wurde mit Anfangsmodell, Halbzeitmodell (262.144 Transitionen) und
Endmodell (524.288) geprüft. Je Modell: 480 stochastisch gespielte Partien,
40 Engine-Seed-Blöcke mit je zwölf Partien. Jeder Block deckt alle drei Sitze
und die vier geordneten Gegnerkombinationen ab. Test-Seed: 59001.

Alle Varianten verwenden exakt dieselben beiden eingefrorenen Gegner aus der
Standardlinie: Checkpoints 131.072 und 262.144. Die Plan-Schlüssel aus Engine-Seed,
Sitz und Gegneridentitäten stimmen bei allen zwölf Modellauswertungen überein.
Die Gegner-Hashes sind in `cross_variant_analysis.json` gespeichert.

Alle **5.760 Partien** sind abgeschlossen, ohne fehlgeschlagene oder abgebrochene
Spiele. Endzustände, Aktionsprotokolle und alle drei Zeitalter-Snapshots sind
vollständig. Die Kategorien summieren sich zu den Gesamtpunkten; Punktetotale
stimmen mit den finalen Spielerzuständen überein. Doppelte Plan-Schlüssel oder
doppelte Aktionszüge wurden nicht gefunden. Die Rohdaten stimmen mit den
gespeicherten Wissenschaftsmittelwerten und VP-Abständen überein.

Für die direkten Vergleiche wurden 2.000 Bootstrap-Stichproben **ganzer
Engine-Seed-Blöcke** gezogen. Die zwölf Partien innerhalb eines Blocks gelten
nicht als unabhängige Wiederholungen. Intervalle sind explorative, nicht für
Mehrfachvergleiche korrigierte 95%-Intervalle. Sie erfassen Testspielvariation,
keine Unsicherheit über verschiedene Trainingsinitialisierungen.

## Endmodelle auf demselben Plan

| Variante | Sieganteil | Eigene VP Ø | Abstand zum stärksten Gegner Ø | Wissenschaft VP Ø | Wissenschaft-Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard: Entropie .01, Lambda .95 | 37,92 % | 38,25 | −2,88 | 4,22 | 34 |
| Entropie .03, Lambda .95 | 40,66 % | 38,97 | −2,74 | 4,30 | 28 |
| Entropie .01, Lambda 1,0 | 31,53 % | 38,54 | −3,93 | 0,77 | 10 |
| Entropie .03, Lambda 1,0 | 36,74 % | 38,40 | −3,48 | 0,48 | 5 |

Geteilte Siege zählen anteilig. Ein negativer mittlerer VP-Abstand kann trotz
eines Sieganteils über einem Drittel auftreten, wenn größere Niederlagen kleinere
Siege aufwiegen. Das untrainierte Modell kommt auf diesem Plan auf 16,01 eigene
Punkte, −33,06 VP-Abstand und null Siege. Alle vier Endmodelle sind ihm hier
deutlich überlegen; daraus folgt keine allgemeine Spielstärke gegenüber Menschen
oder unabhängigen starken Gegnern.

## Kein eindeutiger Sieger zwischen Standard und Entropie

Direkter gepaarter Vergleich **Entropie minus Standard**:

- VP-Abstand: **+0,15**, 95%-Intervall **[−1,35; +1,60]**.
- Sieganteil: **+2,74 Prozentpunkte**, Intervall **[−3,02; +8,37]**.
- Wissenschaftspunkte: **+0,08**, Intervall **[−0,49; +0,70]**.

Alle drei Intervalle enthalten null. Höhere Entropie hat den höchsten beobachteten
Sieganteil, aber ein Vorteil ist auf diesem Plan nicht eindeutig nachgewiesen.
Der große Unterschied der späten Trainings-Wissenschaftsmittelwerte (4,98 gegen
3,36) schrumpft im gemeinsamen Holdout praktisch auf null. Das unterstreicht,
warum Trainingswerte gegen verschiedene Self-Play-Gegner kein fairer
Variantenvergleich waren.

Entropie liegt gegenüber der reinen Lambda-1-Variante bei +9,13 Prozentpunkten
Sieganteil, Intervall [+3,61; +15,17], und +1,19 VP-Abstand, Intervall
[+0,03; +2,40]. Das spricht in diesem Testplan gegen Lambda 1,0 als Verbesserung.
Das VP-Intervall schließt null allerdings nur knapp aus; bei mehreren
explorativen Vergleichen und nur einem Trainingsseed ist eine allgemeine
Überlegenheitsaussage nicht gerechtfertigt. Gegenüber der Kombination ist der
Spielstärkeunterschied nicht eindeutig. Die deutlichen Unterschiede der
Wissenschaftserträge sind von dieser Spielstärkerangfolge zu unterscheiden.

## Frühe Wissenschaft und hohe Erträge

| Variante | Grün in Zeitalter I gebaut / legal angeboten | Bauquote | Partien ≥10 Wissenschafts-VP | ≥20 | ≥30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 147 / 407 | 36,12 % | 79 | 13 | 1 |
| Entropie | 126 / 403 | 31,27 % | 78 | 13 | 0 |
| Lambda 1,0 | 24 / 445 | 5,39 % | 6 | 0 | 0 |
| Kombination | 4 / 436 | 0,92 % | 0 | 0 | 0 |

Anzahlen beziehen sich auf jeweils 480 Partien. Legale Angebote zählen
Entscheidungsgelegenheiten, keine einzigartig ausgeteilten Karten. Grün als
Wunderbaumaterial oder Ablage zählt nicht als grünes Gebäude.

Beim Standard erfüllen neun Partien zusätzlich frühes Grün, mindestens 20
Wissenschaftspunkte und keinen Rückstand; bei Entropie sind es sieben. Bei
mindestens zehn Punkten sind es 26 beziehungsweise 27. Beide Lambda-1-Varianten
erfüllen selbst diese niedrigere Kombination kein einziges Mal.

Die 10/20-Grenzen sind Diagnosewerte, keine Definition starker Spielweise. Von
den Standardpartien erreichen nur **2,71 %** mindestens 20 Wissenschaftspunkte
und **0,21 %** mindestens 30. Keine Endmodellpartie erreicht 40. Der Median liegt
beim Standard bei einem, bei höherer Entropie bei 1,5 Wissenschaftspunkten.
Frühes Grün wird inzwischen genutzt, aber hohe Wissenschaftserträge bleiben
Ausnahmen. Eine dauerhaft zuverlässige Wissenschafts-Hauptstrategie ist nicht
nachgewiesen. Wissenschaftspunkte können auch aus Wundern und Gilden stammen.

## Was brachte die zweite Trainingshälfte?

Gepaarte Endmodell-minus-Halbzeit-Vergleiche auf unverändertem Testplan:

| Variante | Wissenschaft Ø: Halbzeit → Ende | VP-Abstandsgewinn [95 %] | Sieganteilsgewinn in Prozentpunkten [95 %] |
| --- | ---: | ---: | ---: |
| Standard | 1,01 → 4,22 | +1,15 [+0,10; +2,21] | +1,04 [−3,65; +5,63] |
| Entropie | 0,26 → 4,30 | +1,25 [−0,09; +2,65] | +6,08 [+0,28; +12,54] |
| Lambda 1,0 | 0,39 → 0,77 | +0,39 [−0,71; +1,44] | −3,92 [−10,00; +1,77] |
| Kombination | 0,30 → 0,48 | +0,52 [−0,97; +2,07] | +1,84 [−3,61; +7,33] |

Der Wissenschaftszuwachs ist bei Standard und Entropie klar: +3,21 Punkte mit
Intervall [+2,53; +3,93] beziehungsweise +4,04 mit [+3,30; +4,91]. Bei der
Spielstärke sind die Verbesserungen kleiner und nicht über alle Kennzahlen
eindeutig. Der Standard verbessert den VP-Abstand nur mit knapp positivem
unterem Intervallrand; Entropie den Sieganteil ebenfalls mit knapp positivem
unterem Rand. Diese Details sind keine robuste Garantie weiteren Fortschritts.

## Schlussfolgerung und nächstes Experiment

Die Daten stützen eine präzisere Diagnose als „PPO entdeckt Wissenschaft nie“:
Es erlebt solche Folgen und lernt bei Lambda .95 mit längerem Training deutlich
mehr frühes Grün. Das reicht bisher selten für hohe Wissenschaftserträge.
Ein spezieller Deep-Exploration-Engpass ist damit weder bewiesen noch ausgeschlossen.
Die Behauptung, dass Lambda 1,0 das späte Lernsignal hier erfolgreich verbessert,
wird nicht gestützt.

**Empfehlung:** Lambda .95 beibehalten. Noch keinen Entropiesieger ausrufen und
noch keinen Wissenschaftsbonus einführen. Als nächsten kontrollierten Schritt
Standard und Entropie .03 mit Trainingsseeds 47 und 59 wiederholen. Es liegen
bisher vier Varianten, aber nur ein Trainingsseed je Variante vor; die 480
Testpartien ersetzen diese Wiederholungen nicht. Für eine allgemeine
Spielstärkeprüfung wären später zusätzlich unabhängig trainierte Gegner nötig.

Vier entsprechende Trainingsbefehle sind lokal vorbereitet:

```bash
cd /home/marten/dev/7Wonders/python
bash runs/science-baseline-entropy-replications.sh
```

Das sind vier längere Läufe mit jeweils 524.288 Transitionen. Sie wurden bei
dieser Analyse nicht gestartet. Erfolgreiche passende Läufe werden übersprungen,
abgebrochene archiviert und frisch gestartet. Danach wird die Trainingsdiagnose
zusammengefasst; ein gemeinsamer Holdout der neuen Seeds bleibt ein separater
Schritt. Nach Abschluss wieder mit den Ergebnisverzeichnissen melden.

## Artefakte und Dashboard

- Rohprotokolle und Einzelberichte: Unterverzeichnisse `baseline`, `entropy`,
  `credit`, `combined` von `python/runs/science-common-holdout-pTfpDb/`.
- Geprüfte direkte Variantenvergleiche inklusive aller sechs Endmodell-Paare:
  `cross_variant_analysis.json` im selben Verzeichnis.
- Reproduzierbare Offline-Prüfung: `analyze_common.py` im selben Verzeichnis,
  aus `python/` mit der Projektumgebung ausführen.
- Der Workspace-Task **Analyse-Dashboard im Browser öffnen** lädt alle vier
  fertigen Varianten sowie die Trainingsdiagnose in eine gemeinsame lokale Ansicht.
  Die Tabellen unter „Vergleiche“ erlauben den Variantenvergleich; die hier
  ergänzten gepaarten laufübergreifenden Intervalle stehen im Analyse-JSON und
  in diesem Bericht.

## Nachprüfung: Verzicht der Gegner auf grüne Gebäude

Auf die Frage nach möglichem Contesten wurden die finalen Gebäudefarben der
Gegner zusätzlich ausgewertet. Dies ist eine nachträgliche beschreibende Analyse.
Sie erfasst gebaute grüne Gebäude, nicht sämtliche als Wunderbaumaterial oder
Ablage entfernten Wissenschaftskarten und auch nicht die Absicht einer Aktion.

| Endmodell | Beide Gegner bauen null Grün | Beide Gegner bauen jeweils höchstens zwei Grün |
| --- | ---: | ---: |
| Standard | 290 / 480 | 460 / 480 |
| Entropie | 296 / 480 | 456 / 480 |
| Lambda | 300 / 480 | 456 / 480 |
| Kombination | 295 / 480 | 467 / 480 |

Im Standard baut der Lerner in den 290 Partien ohne ein einziges gegnerisches
grünes Gebäude selbst durchschnittlich nur 1,65 Grün, erzielt 4,53
Wissenschaftspunkte und erreicht zehnmal mindestens 20 Punkte. In 107 dieser
Partien hatte er im Verlauf mindestens sechs verschiedene grüne Karten als
legale Bauoption vorliegen. Das bedeutet nicht, dass diese Optionen alle auf
einem gemeinsamen alternativen Spielpfad finanzierbar gewesen wären; es zeigt
aber, dass die vorhandene Policy vielfach grüne Angebote nicht nutzt.

Unter den 460 Standardpartien mit höchstens zwei grünen Gebäuden je Gegner
gibt es 36 mit mindestens fünf eigenen grünen Gebäuden. Diese Teilgruppe erzielt
18,97 Wissenschaftspunkte und 54,17 % Sieganteil. Die Teilgruppe wurde anhand
des beobachteten Verlaufs ausgewählt; günstige Karten, Ressourcen und weitere
Spielbedingungen können die Werte erklären. Daraus folgt kein kausaler
Wissenschaftsvorteil oder Beweis optimaler Strategie.

Die Daten stützen die Beobachtung fehlenden breiten Wissenschaftsaufbaus.
Sie liefern keinen positiven Beleg, dass die geringe Wissenschaftsausbeute
überwiegend durch absichtliches gegnerisches Denial erklärt wird. Fehlende grüne
Gegnergebäude schließen aber Denial als Material/Ablage nicht aus. Auch rein
beiläufige Entfernung grüner Karten braucht kein gelerntes strategisches Denial.
Eine Policy könnte umgekehrt ein lokales Denial-Verhalten lernen, ohne vorher
selbst eine vollständige Wissenschaftsstrategie zu beherrschen.

Der passende nächste gezielte Diagnosetest wäre, eine konsistente Wissenschafts-
Referenzpolicy gegen genau diese Gegner auf dem gemeinsamen Plan auszuwerten.
Damit ließe sich prüfen, ob die vermutete profitable Strategie unter den realen
Simulatorbedingungen überhaupt zuverlässig funktioniert. Ein erfolgreicher
Referenztest bei gleichzeitigem Versagen der gelernten Policy würde ein Problem
beim Entdecken, Lernen oder Beibehalten solcher Folgen stärker stützen, aber
Deep Exploration allein weiterhin nicht eindeutig als Ursache isolieren.
Dieser zusätzliche Test wurde noch nicht implementiert oder ausgeführt.

Die Detailzahlen liegen unter
`python/runs/science-common-holdout-pTfpDb/science_contention_analysis.json`.

## Nachprüfung: Wissenschaft nach dem Einstieg konsequent weiterbauen

Gezählt werden nach dem ersten grünen Gebäude ausschließlich weitere tatsächlich
ausgeführte Entscheidungen mit legalem grünem Gebäudebau. Die Einstiegsaktion
selbst ist nicht im Nenner. Für folgende Tabelle muss der Einstieg in Zeitalter I
liegen; wiederholte Angebote sind weitere Entscheidungsgelegenheiten.

| Endmodell | Partien mit frühem Einstieg | Weitere grüne Bauten / legale Angebote | Weiterbauquote |
| --- | ---: | ---: | ---: |
| Standard | 147 | 346 / 1.003 | 34,50 % |
| Entropie | 126 | 308 / 891 | 34,57 % |
| Lambda | 24 | 16 / 197 | 8,12 % |
| Kombination | 4 | 3 / 21 | 14,29 % |

Alle diese 301 Partien haben mindestens drei weitere legale grüne Angebote.
**Keine einzige** erreicht nach dem Einstieg eine Weiterbauquote von mindestens
90 %. Die niedrigere Schwelle von 80 % erreichen beim Standard fünf und bei
Entropie vier der früh begonnenen Partien; bei den anderen Varianten keine.

Zusätzlich wurden die gesamten Trainingsprotokolle geprüft. Unter frühen
Einsteigern mit mindestens drei Folgeangeboten gibt es im Standard 10 von 3.277
Partien mit mindestens 90 % Weiterbauquote, bei Entropie 10 von 3.083, bei Lambda
null von 1.405, bei der Kombination zwei von 731. Bei mindestens fünf
Folgeangeboten bleiben nur zwei Standard- und drei Entropiepartien übrig.
Damit wäre „nie in irgendeinem Trainingsspiel“ falsch; ein häufig oder
zuverlässig auftretendes Durchhalten dieser Strategie ist aber nicht beobachtet.

Diese Messung operationalisiert das besprochene Commitment. Sie beweist weder,
dass eine starre 90–100%-Regel optimal ist, noch dass allein Deep Exploration
das Hindernis erklärt. Die detaillierten Zahlen liegen in
`science_persistence_analysis.json` und
`training_science_persistence_analysis.json` im Holdout-Verzeichnis.
