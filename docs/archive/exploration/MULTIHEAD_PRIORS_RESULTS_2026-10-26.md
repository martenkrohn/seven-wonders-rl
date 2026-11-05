# Auswertung des Multi-Head-Laufs mit Priors, 26. Oktober 2026

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

**Ergebnis:** Der Prior-Lauf ist auf dem gemeinsamen Testplan klar stärker als
der erste Multi-Head-Lauf. Der Fortschritt zeigt sich auch gegen Standardgegner
und beim Abwurfverhalten. Die Wissenschaftsnutzung nimmt zu. Verlässliche eigene
Manufakturproduktion und deutlich bessere Kartenbauverfügbarkeit in Zeitalter III
sind dagegen noch nicht erreicht.

## Versuch und technische Prüfung

262.144 zusätzliche Übergänge, 512 Updates, 12.480 abgeschlossene Trainingspartien;
Dauer 21m37s, 202 Übergänge/s einschließlich Trainingsevaluationen. Die Laufzeit ist
kein kontrollierter Geschwindigkeitsvergleich mit v1.

Geprüft wurden:

- Identische initiale Tensoren aller gelernten Netze und aller vier Prior-Netze
  gegenüber v1; gleiche Engine, gleiches Budget, gleicher Seed, gleiche PPO-
  Hyperparameter und Standard-Reward. Unterschied: Prior-Stärke 0 → 32.
- Identische Trainingspartie-Seeds, Lernersitze, Kopfzuweisungen und Gegner-Hashes
  in den Startprotokollen beider Läufe. Damit ist dies ein sinnvoller kontrollierter
  Vergleich innerhalb eines Trainingsseeds.
- Unveränderte Prior-Gewichte nach Training; korrekte Quell-/Kopien-Hashes aller
  festen Gegner. Kopf-Übergänge wieder 65.612 / 67.219 / 64.457 / 64.856.
- Keine abgeschnittenen Updates oder nichtendlichen skalaren Trainingsmetriken.
- Alle 2.304 Evaluationspartien vollständig, mit vollständigen Endständen,
  Zeitalterfolgen und Aktionsaufzeichnungen. Checkpoint-Hashes geprüft; keine
  Punktsummenfehler oder unbekannten gebauten Karten in den Berichten.

## Spielstärke

Je Policy 384 Spiele: acht Engine-Seed-Blöcke × 16 geordnete Gegnerpaare × drei
Lernersitze. Sieganteil teilt Gleichstände zwischen punktgleichen Siegern auf.
Punkteabstand = eigene Punkte minus Punkte des stärksten Gegners.

| Policy | Sieganteil | Eigene Punkte | Punkteabstand | Wissenschaft | Militär |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ursprüngliches Standardmodell | 31,6% | 42,68 | -5,27 | 2,96 | 8,89 |
| v1, Zufallskopf ohne Priors | 37,4% | 45,33 | -2,52 | 4,60 | 3,60 |
| v2 vor Lernen, mit Priors | 30,2% | 42,26 | -5,33 | 2,67 | 8,66 |
| **v2 nach Lernen, Zufallskopf** | **49,3%** | **47,38** | **-0,03** | **9,53** | **2,51** |
| v2 Kopf 0 | 47,8% | 47,04 | -0,60 | 9,11 | 2,36 |
| v2 Kopf 1 | 49,3% | 47,19 | -0,39 | 9,58 | 1,89 |
| v2 Kopf 2 | 45,2% | 46,87 | -0,74 | 8,95 | 2,42 |
| v2 Kopf 3 | 47,0% | 46,67 | -0,93 | 9,52 | 1,72 |

Das finale Zufallskopfmodell gewinnt gegenüber v1 **+2,49 Punkte Abstand**,
gepaartes 95%-Intervall **[+0,94; +3,94]**. Der Sieganteil steigt um **11,94
Prozentpunkte**, Intervall **[+6,81; +17,27]**. Gegen das ursprüngliche Standardmodell
sind es +5,24 Abstandspunkte [+3,69; +7,01] und +17,66 Prozentpunkte Sieganteil.

`initial` in der Nutzertabelle ist das ursprüngliche gelernte Modell **mit aktiven
Priors vor zusätzlichem Training**, nicht der Endstand von v1. Sein Abstand zum
ursprünglichen Modell ist -0,06 [-0,95; +0,83]. Die zufälligen Präferenzen allein
haben hier keinen belegten Spielstärkegewinn erzeugt. Der Gewinn entsteht nach
Training unter ihrem Einfluss.

Kopf 0/1/2 sind im Punkteabstand jeweils klarer besser als v1s Zufallspolicy;
bei Kopf 3 schließt das entsprechende Intervall knapp null ein. Das beweist keine
Überlegenheit eines bestimmten Kopfes über die anderen v2-Köpfe. Kopf 3 hat
beispielsweise das günstigste Abwurf-/Manufakturmuster, aber nicht den höchsten
Punkteabstand. Eine Auswahl des stärksten Kopfes auf demselben Testplan wäre
explorativ und müsste auf neuen Seeds bestätigt werden.

## Breiterer Fortschritt gegenüber den Gegnern

| Geordnete Gegnergruppen links / rechts | Sieganteil v1 → v2 | Abstand v1 → v2 |
| --- | ---: | ---: |
| Standard / Standard | **25,0% → 44,8%** | **-5,98 → -2,13** |
| Standard / Wissenschaft | 42,0% → 42,7% | -2,25 → -0,77 |
| Wissenschaft / Standard | 31,2% → 32,3% | -5,79 → -4,69 |
| Wissenschaft / Wissenschaft | 100,0% → 91,7% | 20,13 → 18,67 |
| Standard / Militär | 35,4% → 48,6% | -3,83 → -0,35 |
| Militär / Standard | 31,2% → 40,6% | -3,46 → -1,96 |
| Militär / Wissenschaft | 60,4% → 66,7% | 2,96 → 3,04 |
| Wissenschaft / Militär | 43,1% → 62,5% | -2,58 → 1,58 |
| Militär / Militär | 14,6% → 60,4% | -6,17 → 0,29 |

Standard/Standard enthält 96 Spiele, gemischte Standard-/Spezialistengruppen je
48 und reine Spezialistenpaare je 24. Die Zahl unabhängiger Seed-Blöcke bleibt acht.

Gegen Standard/Standard beträgt der gepaarte Abstandszuwachs gegenüber v1 **+3,85**,
Intervall [+1,56; +7,15]; Sieganteil +19,79 Prozentpunkte [+7,81; +33,85]. Dieser
Fortschritt fehlte im ersten Versuch.

Ohne Wissenschaft/Wissenschaft beträgt der Gesamtzuwachs gegenüber v1 sogar
+2,75 Abstandspunkte [+1,27; +4,35]. Auf die Trainings-Gegnerwahrscheinlichkeiten
umgewichtet sind es +2,81 [+1,22; +4,73]. Beides verwendet vorhandene Spiele,
keine zusätzlichen Simulationen. Der Gewinn beruht diesmal nicht auf besserem
Ausnutzen zweier Wissenschaftler; genau dieses Matchup wurde etwas schwächer.

## Wissenschaft und Investitionen

Gegenüber v1 steigt Wissenschaft von 4,60 auf **9,53 Punkte**, während Blau mit
21,34 → 21,01 nahezu konstant bleibt. Militär sinkt von 3,60 auf 2,51; auch
Münzpunkte und Handel geben etwas nach. Die durchschnittlichen Wissenschaftsbauten
steigen in I/II/III von 0,45 / 0,80 / 0,55 auf **0,65 / 1,30 / 1,25**.

Der Wissenschaftsmedian steigt auf neun, das 90%-Quantil liegt bei 25, das Maximum
bei 43 Punkten. 13,5% der Partien enden ohne Wissenschaftspunkte. Das ist eine
breitere Nutzung von Wissenschaft unter normalem Reward, kein Nachweis eines
voll entwickelten Wissenschaftsspezialisten.

Frühe graue Bauten (I+II) steigen von **37 auf 78** über dieselben 384 Partien:
0,096 → 0,203 pro Partie. Allerdings baut bereits das untrainierte Prior-Startmodell
69 graue Karten auf diesem Plan. Ein großer Anteil dieser Verhaltensverschiebung
war somit von den Priors schon vorgegeben; die reine Häufigkeitszunahme ist kein
Beweis für gelerntes Ressourcenverständnis.

Kopf 0 baut weiterhin keine graue Karte in Zeitalter I und nur sechs in I+II
zusammen, erzielt aber dennoch einen guten Sieganteil. Kopf 3 baut 99 graue Karten
in I+II. Eigene graue Produktion ist deshalb keine notwendige Bedingung für gute
Resultate in jeder Situation; Handel und kostenlose Bauketten können ebenfalls
Ressourcenzugang ermöglichen. Diese möglichen Mechanismen sind hier nicht einzeln
kausal nachgewiesen.

## Was die Abwürfe tatsächlich erklären

| Policy | Abwürfe I | Abwürfe II | Abwürfe III | III ohne legale Kartenbaualternative |
| --- | ---: | ---: | ---: | ---: |
| v1 Zufallskopf | 0,245 | 0,500 | 1,016 | 0,617 |
| **v2 Zufallskopf** | **0,112** | **0,177** | **0,552** | **0,513** |
| v2 Kopf 0 | 0,328 | 0,346 | 0,714 | 0,612 |
| v2 Kopf 1 | 0,083 | 0,198 | 0,536 | 0,487 |
| v2 Kopf 2 | 0,094 | 0,188 | 0,633 | 0,612 |
| v2 Kopf 3 | 0,003 | 0,211 | 0,456 | 0,430 |

Die gesamten Abwürfe sinken von 1,760 auf **0,841 pro Partie** (-52%); in III
allein von 390 auf 212 (-46%). Besonders deutlich: Abwürfe trotz legaler Karten
in III sinken von **153 auf 15**. Verfügbare Karten werden wesentlich häufiger
gebaut. Eine legale Alternative macht einen Abwurf nicht automatisch falsch;
das bessere Spielergebnis stützt hier aber den Nutzen der Verhaltensänderung.

Die eigentliche Ressourcen-/Baubarkeitsfrage braucht eine genauere Trennung:

| Zeitalter-III-Entscheidungen, insgesamt 2.304 | v1 | v2 |
| --- | ---: | ---: |
| Ohne legalen Kartenbau, unabhängig von gewählter Aktion | 275 | 268 |
| Davon Abwurf | 237 | 197 |
| Davon Wunderbau | 38 | 71 |
| Mit legalem Kartenbau, trotzdem Abwurf | 153 | 15 |

Die Verfügbarkeit irgendeines Kartenbaus steigt nur von **88,06% auf 88,37%**.
Von den 40 vermiedenen Abwürfen ohne Kartenbaualternative entfallen netto 33 auf
mehr Wunderbau in solchen Situationen und sieben auf weniger solcher Situationen.
Das ist eine deskriptive Zerlegung, keine Zuordnung individueller kontrafaktischer
Spielentscheidungen. Weniger späte Abwürfe belegen hier vor allem bessere
Aktionsnutzung, nicht bereits eine große Verbesserung der Ressourcenversorgung.

## Diversität und Zufallskopf-Schätzung

Die durchschnittliche JS-Divergenz auf denselben Rolloutzuständen beträgt im letzten
Trainingsviertel 0,0785 statt 0,0189 in v1. Das gespeicherte Greedy-Abweichungsmaß
steigt von 3,59% auf 9,22% (Vergleich gegen Kopf 0, einschließlich dessen null
Abweichung). Die Köpfe bleiben unterschiedlicher. Ein Anteil davon entsteht direkt
durch die eingefrorenen Priors; vier klar getrennte gelernte Strategien sind damit
nicht nachgewiesen. Alle Köpfe liegen bei Wissenschaft nun nahe neun Punkten.

Der Zufallskopf wird auf diesem Plan in nur 24 Seed-/Sitzkontexten gezogen und
anschließend für die 16 Gegnerpaare wiederholt. Wie in v1 ergeben sich 48 / 128 /
112 / 96 Spiele je Kopf. Daher ist seine Schätzung nicht der exakt gleichgewichtete
Durchschnitt der vier einzeln geprüften Köpfe.

Aus allen vier vollständigen Kopf-Auswertungen kann dieser Erwartungswert ohne
neue Spiele berechnet werden: v1 **37,33%**, v2 **47,32%** Sieganteil; Abstand
-2,55 → -0,66. Der gepaarte Vorteil bleibt bestehen: **+1,89 Abstandspunkte**
[+0,17; +3,60] und **+9,99 Prozentpunkte** Sieganteil [+4,17; +16,45]. Der Befund
hängt also nicht allein an der konkreten zufälligen Kopfzuordnung des Holdouts.

## Einordnung und nächste Entscheidung

Die Priors sind in diesem kontrollierten Versuch ein überzeugender Kandidat für
nützliche Exploration: gleicher Start, gleiche äußere Trainingsbedingungen,
besseres Endergebnis und geändertes Verhalten. Das ist mehr Evidenz als beim
vorherigen Vergleich mit dem ursprünglichen Modell ohne gleich langes Kontroll-
training. Dennoch bleibt es ein Trainingsseed mit einem festen Satz Prior-Netze,
acht mehrfach benutzten Evaluationsseed-Blöcken und bekannten Gegnerfamilien.
Bootstrap-Intervalle erfassen keine Streuung über erneutes Training und ersetzen
keine frische Bestätigung.

Das Ergebnis rechtfertigt, dieses Setup beizubehalten. Als günstiger nächster
Prüfschritt bietet sich eine gemeinsame Evaluation von v1/v2 auf neuen, zahlreicheren
Seeds an, beispielsweise 32–40 Blöcken. Danach ist ein längerer Lauf mit festen
Einstellungen und Zwischenmessungen sinnvoll, um die zuvor diskutierte Budgetfrage
zu prüfen. Die kurzen 48-Spiele-Trainingsevaluationen allein belegen kein Plateau.

Für ein echtes Fortsetzen müssen Modell, Optimizer und relevante Zustände sauber
wiederhergestellt werden. Die vorhandene `--init-checkpoint`-Option übernimmt nur
Gewichte und startet Optimizer/Zähler/Streams neu; das ist kein exaktes Resume.
Eine Budgetverlängerung oder ein neues Training wurde für diese Auswertung nicht
gestartet. Fehlende explizite Wunderidentitäten und kommende Stufenkosten/-effekte
bleiben ein separates Thema der Beobachtungsrepräsentation.

## Nachvollziehbarkeit

Offline-Audit: `python/runs/multihead-v2-priors-analysis-20261026/analyze.py`.
Auditergebnis: gleichnamige `analysis.json`; Kopie als
`docs/archive/exploration/MULTIHEAD_PRIORS_RESULTS_2026-10-26.json`.
Dashboard: `python/runs/multihead-v2-priors-seed31-holdout/dashboard.html`.
Keine neuen Spiele oder Trainingsläufe wurden für diese Auswertung ausgeführt.
