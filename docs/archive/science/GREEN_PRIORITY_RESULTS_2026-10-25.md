# Grün-Priorität gegen freie Policy: Ergebnis

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Ausgewertet am 25. Oktober 2026. Ergebnisverzeichnis:
`python/runs/green-priority-754c02f23dba4ecdab03a5f3d04bfa7a/`.

**Die feste Grün-Priorität erzeugt deutlich höhere Wissenschaftserträge, aber
verschlechtert den Sieganteil dieses Modells gegen die getesteten Gegner.**
Die eigenen Gesamtpunkte bleiben im Mittel praktisch gleich. Der stärkste
Gegner erzielt dagegen mehr Punkte. Damit ist die Hypothese „dieses Modell
spielt schon durch konsequentes Erzwingen legaler grüner Bauten besser“ auf
diesem Testplan nicht bestätigt, sondern das Gegenteil beobachtet.

## Versuch und Rohdatenprüfung

Das finale Entropie-.03/Lambda-.95-Modell von Trainingsseed 31, Schritt 524.288,
spielt frei und mit Grün-Priorität. Beide Modi verwenden denselben Checkpoint
mit SHA-256
`4261792c2774b6e7ed68cb5dff7ab1e545246edb357b15d1ed147aa0affa8ed0`.
Der aktuelle Checkpointhash stimmt mit beiden gespeicherten Identitäten überein.
Gegner sind die eingefrorenen Standard-Checkpoints 131.072 und 262.144.

Test-Seed 69001; je 480 Partien in 40 Seed-Blöcken mit zwölf Sitz-/Gegner-
Kombinationen. Alle 960 Partien sind vollständig abgeschlossen. Geprüft wurden:
identische eindeutige Plan-Schlüssel, identische rollenbezogene Zufallsseeds,
Modellidentitäten, drei Zeitalter-Snapshots pro Partie, Endpunktzahlen und
Kategoriesummen. Die Regel wurde anhand jeder ausgeführten Aktion unabhängig
geprüft: **2.164 grüne Gebäude bei 2.164 legalen grünen Angeboten**, null
Regelverletzungen. Die freie Policy baut bei 2.603 Angeboten 731 grüne Gebäude.
Die unterschiedlichen Angebotszahlen entstehen entlang unterschiedlicher
Spielverläufe; es werden nicht identische spätere Beobachtungen verglichen.

Die Restriktion gilt ab Spielbeginn. Die Wahl zwischen mehreren grünen Gebäuden
und alle Entscheidungen ohne legal baubares Grün bleiben dem trainierten Modell
überlassen. Es findet kein nachträgliches Lernen der Ressourcenplanung statt.

## Hauptergebnisse

| Kennzahl | Freie Policy | Grün-Priorität |
| --- | ---: | ---: |
| Sieganteil, geteilte Siege anteilig | 37,92 % | 26,46 % |
| Eigene Gesamtpunkte Ø | 39,32 | 39,29 |
| Abstand zum stärksten Gegner Ø | −3,43 | −6,19 |
| Wissenschaftspunkte Ø | 4,07 | 17,29 |
| Wissenschaftsmedian | 1 | 18 |
| Wissenschaftsmaximum | 26 | 55 |
| Partien ab 20 Wissenschaftspunkten | 9 / 480 | 209 / 480 |
| Partien ab 30 Wissenschaftspunkten | 0 / 480 | 100 / 480 |
| Partien ab 40 Wissenschaftspunkten | 0 / 480 | 38 / 480 |
| Partien ab 50 Wissenschaftspunkten | 0 / 480 | 3 / 480 |
| Grüne Gebäude pro Partie Ø | 1,52 | 4,51 |

Die Regel erzeugt also genau das zuvor seltene Verhalten: hohe Wissenschafts-
erträge werden häufig. 43,54 % der Partien erreichen mindestens 20 Punkte und
20,83 % mindestens 30. Das reicht in dieser Kombination aus Prioritätsregel,
unveränderter restlicher Policy und Gegnern nicht für einen höheren Sieganteil.

## Direkter gepaarter Vergleich

Grün-Priorität minus freie Policy; 2.000 Bootstrap-Stichproben ganzer
Engine-Seed-Blöcke, explorative 95%-Intervalle:

| Änderung | Mittelwert | 95%-Intervall |
| --- | ---: | ---: |
| Sieganteil / Prozentpunkte | −11,46 | [−18,33; −4,27] |
| VP-Abstand | −2,76 | [−4,30; −1,20] |
| Eigene Gesamtpunkte | −0,03 | [−1,24; +1,23] |
| Eigene Wissenschaftspunkte | +13,22 | [+11,16; +15,45] |
| Punkte des jeweils stärksten Gegners | +2,73 | [+2,03; +3,47] |

Der schlechtere Sieganteil und VP-Abstand sind auf diesem Plan klarer als
gewöhnliche Testspielschwankungen. Das ist keine Aussage über beliebige Modelle,
Trainingsseeds oder Gegnerpopulationen.

## Punkteverschiebung

| Eigene Punktequelle | Frei Ø | Grün-Priorität Ø | Änderung |
| --- | ---: | ---: | ---: |
| Wissenschaft | 4,07 | 17,29 | +13,22 |
| Blau | 14,03 | 9,29 | −4,73 |
| Militär | 5,56 | 0,91 | −4,65 |
| Handel ohne Münzen | 1,90 | 1,02 | −0,88 |
| Gilden | 2,94 | 1,57 | −1,37 |
| Direkte Wunderpunkte | 5,99 | 4,86 | −1,13 |
| Münzen | 4,85 | 4,35 | −0,49 |

Der Wissenschaftszuwachs ersetzt nahezu gleich viele Punkte aus anderen
Quellen. Gleichzeitig steigt der durchschnittliche Gesamtwert eines Gegners
von 37,30 auf 40,51. Dessen Militärmittel steigt von 4,54 auf 7,15 und Blau
von 15,29 auf 17,02. Der jeweils stärkste Gegner steigt von 42,75 auf 45,48.
Die Zerlegung −0,03 eigene Punkte minus +2,73 gegnerische Punkte ergibt den
schlechteren VP-Abstand von −2,76.

Dies sind beobachtete Kategorienverschiebungen. Sie isolieren nicht, ob etwa
geänderte Kartenweitergabe, automatische Militärwertungen oder zustandsabhängige
Aktionsentscheidungen der Gegner den jeweiligen Anteil verursachen. Sie sind
insbesondere kein Beleg für absichtliches gegnerisches Wissenschafts-Denial.
Die Wissenschaftspunkte eines Gegners sinken im Mittel sogar von 0,59 auf 0,33.
Beide Gegner bauen in 367 erzwungenen Partien kein grünes Gebäude; frei sind es
312. Karten als Material/Ablage können trotzdem verloren gehen.

## Was der Test noch nicht beantwortet

Eine optimal vorbereitete Wissenschaftsstrategie wurde nicht getestet. Die
Grün-Priorität nutzt nur bereits legale Bauaktionen; sie plant keine bisher
fehlenden Ressourcen oder Bauketten voraus. Im ersten Zeitalter steigen grüne
Gebäude nur von 0,21 auf 0,43 pro Partie; im dritten von 0,62 auf 2,38. Frühes
Grün kommt frei in 100 und erzwungen in 206 Partien vor. Ein großer Teil des
zusätzlichen Wissenschaftsaufbaus geschieht damit später. Ob eine bessere
wirtschaftliche Vorbereitung die Strategie insgesamt stärker machen würde,
ist aus diesem Versuch nicht ableitbar.

Das Ergebnis widerlegt weder den allgemeinen Nutzen von Wissenschaft noch
beweist es, dass die freie Policy bereits optimal zwischen Strategien wählt.
Es liefert aber **keinen positiven Nachweis einer durch die einfache Regel
erschließbaren Gewinnstrategie** und damit auch keinen positiven Nachweis für
Deep Exploration als alleinigen Engpass.

Falls Wissenschaft weiter gezielt untersucht wird, müsste ein nächster Versuch
auch die Vorbereitung lernen oder anders wählen dürfen. Ein Wissenschaftsbonus
wäre dafür ein eigener Trainingsversuch, dessen Erfolg anschließend am echten
Spielziel ohne Bonus und ohne Zwang geprüft werden müsste. Dieser Bericht
startet oder implementiert keinen solchen Folgeversuch.

## Artefakte

- Originale: `summary.json`, `analysis.json`, `comparison_1.jsonl`, `candidate.jsonl`.
- Unabhängige Prüfung und zusätzliche gepaarte Intervalle: `audit_comparison.py`
  und `verified_comparison.json` im Ergebnisverzeichnis. Das Skript aus `python/`
  mit der Projektumgebung ausführen; es spielt keine neuen Partien.
- Der Workspace-Task **Analyse-Dashboard im Browser öffnen** lädt diese
  Auswertung einschließlich Regelprüfung und direktem Vergleich.
