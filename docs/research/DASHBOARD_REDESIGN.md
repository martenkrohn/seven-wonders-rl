# Dashboard neu denken: Feedback und Diskussionsgrundlage

Stand: 29. Oktober 2026. Zusammenfassung des Brainstormings, abgeglichen mit
der aktuellen Dashboard-Implementierung und vorhandenen Auswertungsdaten.
Enthält auch die anschließenden Präzisierungen und einen konkreten
Gestaltungsvorschlag. Festgehaltene Nutzerentscheidungen und ergänzende
Vorschläge sind getrennt. Der anschließende Umsetzungsauftrag ist inzwischen
ausgeführt; die tatsächliche Oberfläche und Statistik beschreibt der
[Evaluationsleitfaden](../guides/EVALUATION.md#neue-oberfläche).

## 1. Das eigentliche Problem

Das Dashboard enthält bereits sehr viele, grundsätzlich wichtige Informationen.
Die Menge und Darstellung machen das Lesen aber so aufwendig, dass die
Auswertung im Alltag häufig nicht vollständig angeschaut und durchdrungen wird.
Damit bleibt ein Teil ihres Nutzens ungenutzt.

**Ziel:** Die Auswertung soll einfacher, schneller erfassbar und einladender
werden und dabei möglichst viele der tatsächlich interessanten Informationen
bewahren. Mehr Information auf einer Seite ist nicht automatisch mehr Nutzen.

Die Überlegung geht deshalb über das Entfernen einzelner Spalten hinaus.
**Nachträglich präzisiert:** Ob die Umsetzung technisch ein Neuaufbau oder eine
starke Überarbeitung wird, ist für dich zweitrangig. Das bestehende visuelle
Design einschließlich Dark Mode gefällt dir und kann als Grundlage bleiben.
Inhaltlich und strukturell sind trotzdem größere Veränderungen zu erwarten.

## 2. Seitenstruktur und inhaltliche Prioritäten

Die folgenden Nummern entsprechen der tatsächlichen Navigation. Die im Gespräch
angesprochene Punkteübersicht heißt dort **02 Score breakdown**.

| Aktueller Bereich | Dein Feedback / gewünschte Richtung | Abgleich mit dem Dashboard |
| --- | --- | --- |
| **01 Overview** | Kann entfallen. Der relevante Überblick ist für dich die Punkteherkunft. | Zeigt Kennzahlkacheln, ein Punkteprofil, Testkontext und einen Modellvergleich. Das Punkteprofil erscheint auch auf 02; Modellvergleiche gibt es ebenfalls auf 05. |
| **02 Score breakdown** | Inhaltlicher Ausgangspunkt und möglicher neuer Einstieg. Das farbige **Score profile** ist besonders gelungen. | Enthält eine Verteilungstabelle, das farbige Punkteprofil und den Abstand zum besten Gegner nach Punktekategorie. |
| **03 Ages** | Interessant und erhaltenswert, besonders **Economy and development**. Die Entwicklung der Produktionsgüter über die Zeitalter sollte schneller lesbar werden. | Zeigt Punkte sowie Wirtschaft und Spielaufbau jeweils am Ende von Zeitalter I, II und III. |
| **04 Cards and actions** | Schlanker gestalten; insbesondere absolute Bauhäufigkeiten entfernen. Bilder und Farbcodierung könnten die Orientierung erleichtern. | Enthält eine Tabelle einzelner Karten mit acht Spalten sowie eine zweite Tabelle mit gebauten Farben, Wunderbau und Kartenverkauf je Zeitalter. |
| **05 Comparisons** | Vergleiche werden durch das fortlaufende Training besonders relevant. Die künftige Darstellung muss neu durchdacht werden. | Enthält Modellvergleiche, gepaarte Fortschrittswerte, Untergruppen und eine Tabelle geladener Auswertungen. Eine konkrete Beibehaltung dieser Seitenform wurde im Gespräch nicht festgelegt. |
| **06 Data quality** | Kann komplett als Dashboard-Seite entfallen. | Zeigt Vollständigkeit, Fehler, Testbedingungen und technische Modellidentität. |
| **07 Science diagnostics** | Kann komplett als Dashboard-Seite entfallen. | Eine gesonderte Trainingsdiagnose mit Wissenschaftskennzahlen und Policy-Wahrscheinlichkeiten; ohne geladene Diagnosedaten bleibt ein Hinweis zum Importieren. |

Das Entfernen von Seiten ist hier zunächst als Wunsch für die Oberfläche
festgehalten. Daraus folgt kein Auftrag, gespeicherte Auswertungsdaten zu löschen.
Ob einzelne Informationen aus entfallenden Seiten anderswo gebraucht werden,
bleibt Teil der späteren Designentscheidung.

## 3. Kennzahlen und Tabellenspalten reduzieren

### Punkteherkunft und Verteilungen

Deine Einschätzung ist, dass folgende Angaben für die gewünschte Auswertung
keine Rolle spielen beziehungsweise keinen erkennbaren Nutzen haben:

- **Median:** aus der Darstellung entfernen.
- **Zero points / Ohne Punkte:** entfernen.
- **Mean when positive / Ø bei positiven Punkten:** Nutzen ist für dich unklar;
  diese Angabe wird ebenfalls als Streichkandidat festgehalten.

Die Tabelle auf 02 zeigt aktuell neben der Kategorie acht numerische Spalten:
Spiele, Mittelwert, Median, P10, P90, Maximum, Zero points und Mean when positive.
Das macht deinen Eindruck der Überladung konkret nachvollziehbar.

Zur Einordnung des unklaren Begriffs: **Mean when positive** ist der Mittelwert
nur über Spiele, in denen die betreffende Punktekategorie einen positiven Wert
hatte. Beispiel aus der festen historischen Referenz mit 336 Partien:
Wissenschaft erzielt im Mittel **21,44 Punkte über alle Spiele**, aber
**23,86 Punkte bei Betrachtung nur der Spiele mit positiven Wissenschaftspunkten**.
Das ist eine zusätzliche Diagnosefrage, keine notwendige zweite Angabe zum
gewöhnlichen Durchschnitt. Die Erklärung begründet keine Empfehlung, die Spalte
beizubehalten.

Für P10, P90, Maximum, Spielanzahl und Unsicherheitsintervalle wurde noch keine
abschließende Entscheidung getroffen. Sie sind durch das Brainstorming nicht
automatisch mit gestrichen.

### Einzelne Karten und Aktionen

Konkret genannt wurden **Build actions** und **Games with a build**. Dein
Grundgedanke: Absolute Häufigkeiten sind für die inhaltliche Beurteilung wenig
interessant und beanspruchen unnötig Platz.

Die aktuelle Tabelle unterscheidet allerdings mehrere Größen:

| Spalte | Tatsächliche Bedeutung | Verständnis deines Feedbacks |
| --- | --- | --- |
| Build actions | Absolute Anzahl der Bauaktionen für eine Karte. | Entfernen. |
| Games with a build (#) | Absolute Anzahl der Partien, in denen die Karte gebaut wurde. | Entfernen. |
| Games with a build (%) | Anteil der Partien, in denen die Karte gebaut wurde. | Beibehalten: Du findest diese Statistik interessant. Im Kontext der Rückfrage ist dies als Prozentanteil verstanden; die absolute Anzahl entfällt weiterhin. |
| Legal opportunities | Anzahl der Entscheidungen, bei denen die Karte legal baubar war; wiederholte Angebote zählen mehrfach. | Ebenfalls eine absolute Häufigkeit und damit ein weiterer Prüfpunkt, aber nicht ausdrücklich von dir gestrichen. |
| Selection rate | Bauaktionen geteilt durch legale Angebote. | Möglicherweise informativer für die tatsächliche Auswahlneigung; noch keine festgelegte Entscheidung. |

Ein konkretes Beispiel für die Redundanz: Bei **Altar** stehen im geprüften
Referenzbericht sowohl unter Build actions als auch unter Games with a build (#)
**33**. Die beiden weiteren Quoten beantworten unterschiedliche Fragen:
Altar wurde in **9,8 % der Partien** gebaut und bei **5,7 % seiner legalen
Angebote** gewählt.

**Ergänzende Einordnung:** Absolute Zahlen sind vor allem hilfreich, um die
Datengrundlage einer Quote einzuschätzen. Das verlangt nicht, sie ständig als
eigene Spalten zu zeigen. Ein gemeinsamer Hinweis zur Stichprobe oder Details
auf Nachfrage wären mögliche Alternativen. Die Zahl abgeschlossener Partien
und die durchschnittlichen Aktionen je Spiel sind außerdem andere Größen als
die beanstandeten absoluten Kartenhäufigkeiten.

## 4. Bilder, Symbole und Farben stärker einsetzen

### Dein Gestaltungswunsch

- Mehr mit Bildern, Symbolen und Piktogrammen arbeiten, wenn Inhalte dadurch
  schneller erkannt werden können.
- Produktionsgüter auf 03 mit ihren tatsächlichen Symbolen darstellen, um lange
  Textbezeichnungen kompakter zu ersetzen oder zu ergänzen.
- Auch Kartenbilder als mögliche Orientierungshilfe prüfen.
- Farbcodierung konsequenter verwenden. Das farbige Score profile ist das
  konkrete positive Vorbild.

### Was bereits vorhanden ist

Im lokalen Projekt liegen tatsächlich passende Grafiken:

- [Produktions- und weitere Spielsymbole](../../images/resources/), darunter
  Holz, Erz, Lehm, Stein, Stoff, Glas und Papyrus sowie Münzen und Schilde.
- [Kartenbilder](../../images/cards/), beispielsweise Lumber Yard und Workshop.
- [Wunderbilder](../../images/wonders/).

Die vorhandene Spieloberfläche [Game.py](../../Game.py) lädt diese Dateien
bereits. Holzsymbol und Lumber-Yard-Karte wurden exemplarisch visuell geprüft.
Das aktuelle Auswertungsdashboard verwendet diese Bilder bislang nicht;
es besitzt bereits feste Farben für die Punktekategorien im Score profile.

**Fachliche Präzisierung für 03:** Die Güterwerte zeigen die durchschnittliche
**feste Produktionskapazität am Ende eines Zeitalters**, keinen angesammelten
Gütervorrat. Wahlproduktion ist bisher nicht eingerechnet. Eine kompaktere
Darstellung sollte diese Bedeutung erhalten.

**Ergänzende Gestaltungsidee:** Eine kompakte Matrix mit Gütersymbolen und den
drei Zeitaltern könnte die Entwicklung direkt vergleichbar machen. Farben
sollten über alle Ansichten dieselbe Bedeutung haben; kurze Beschriftungen
oder Tooltips könnten Symbole erklären. Das ist ein Vorschlag, noch keine
vereinbarte Darstellung.

## 5. Fortlaufendes Training und Vergleiche über die Zeit

### Neue Anforderung aus deinem Trainingsablauf

Ein Modell wird ohne festes Endbudget weitertrainiert und regelmäßig an
Checkpoints beziehungsweise nach einem Trainingsintervall evaluiert.
Das Dashboard soll diesen Verlauf und die Unterschiede zwischen Modellständen
verständlich zeigen. Eine einzelne Momentaufnahme reicht dafür nicht aus.

### Bereits vorhandener Stand

Laut [Dauerlauf-Anleitung](../guides/CONTINUOUS_TRAINING.md) und Implementierung
ist das neue Setup vorbereitet; laut Dokumentationsstand wurde der eigentliche
Dauerlauf noch nicht gestartet:

- Evaluation zu Beginn und anschließend alle **262.144 Übergänge**, jeweils
  **96 Partien** auf einem festen Kontrollplan. Das ist derzeit ein
  schrittbasiertes Intervall, kein Intervall nach verstrichener Uhrzeit.
- Die erzeugte Dashboard-Datei enthält bereits eine Verlaufstabelle mit
  Trainingsschritt, Sieganteil und durchschnittlichem Punktevorsprung. Sie wird
  oberhalb der Seitenauswahl eingeblendet; sie ist noch keine grafische Lernkurve.
- Im Auswahlfeld bleiben der Start und die zwölf neuesten Detailberichte
  sowie die feste historische Referenz. Der vollständige Verlauf bleibt
  gespeichert, und die Verlaufstabelle wird bislang nicht entsprechend gekürzt.
- Für direkte Vergleiche wird das feste Referenzmodell zusätzlich auf dem
  neuen Kontrollplan evaluiert. Seine alte historische Evaluation hat einen
  anderen Prüfplan und ist nicht unmittelbar dieselbe Vergleichsgrundlage.

### Ergänzende Diskussionspunkte für das neue Design

Die folgenden Punkte sind aus deinem Wunsch abgeleitet, noch keine von dir
festgelegten Anforderungen an die konkrete Oberfläche:

1. **Fortschritt erkennen:** Wenige klar lesbare Verlaufskurven statt einer
   auf Dauer immer längeren Tabelle. Welche Kennzahlen verdienen diesen Platz?
2. **Veränderungen verstehen:** Neben der Spielstärke könnte die Veränderung
   des farbigen Punkteprofils zeigen, wie sich die Spielweise entwickelt.
3. **Modellstände gezielt vergleichen:** Aktueller Stand gegen einen gewählten
   früheren Stand oder die feste Referenz, mit direktem Zugang zu den Details.
4. **Lange Verläufe lesbar halten:** Zeitraum oder Schrittbereich wählen und
   ältere Checkpoints bei Bedarf erreichen können.
5. **Vergleichsbasis sichtbar machen:** Ergebnisse auf gleichem Prüfplan von
   historischen Ergebnissen mit anderen Bedingungen unterscheidbar halten.
6. **Zeitachse klären:** Trainingsschritte sind bereits vorhanden. Ob zusätzlich
   tatsächliche Trainingsdauer oder Kalenderzeit wichtig ist, bleibt offen.

## 6. Präzisierte Entscheidungen

- Bestehendes Erscheinungsbild und Dark Mode als Grundlage weiterverwenden;
  die technische Frage „Neuaufbau oder Umbau?“ muss nicht vorab entschieden werden.
- **Games with a build bleibt erhalten**, verstanden als Prozentanteil der Partien.
- Zu Streuung und Unsicherheit wünschst du zunächst eine verständliche
  Darstellung der Optionen und einen Vorschlag.
- Auch für die Fragen, die der Einstieg beantworten soll, möchtest du einen
  konkreten Vorschlag, statt sie selbst vorab festlegen zu müssen.
- **Weitere Präzisierung:** P10–P90 und eine Beurteilung der statistischen
  Signifikanz von Verbesserungen sind erwünscht. Als kompakte Darstellung
  schlägst du Signifikanzsterne wie in Regressionstabellen vor.

## 7. Vorschlag: Vier Bereiche mit einer klaren Einstiegsseite

Dieser Abschnitt ist ein Gestaltungsvorschlag, noch keine bestätigte Entscheidung.
Die vorhandene visuelle Sprache bleibt die Grundlage. Die Navigation wird auf
vier Bereiche reduziert:

| Bereich | Leitfrage | Direkt sichtbarer Inhalt |
| --- | --- | --- |
| **Punkte** – neuer Einstieg | Woher kommen die Punkte, und wo liegen Stärken und Schwächen? | Farbiges Punkteprofil, Durchschnitt pro Kategorie und Abstand zur gewählten Vergleichsbasis. |
| **Zeitalter** | Wie baut das Modell seine Wirtschaft und Wertung auf? | Gütersymbole in einer Matrix für Ende I, II und III; dazu Punkteentwicklung, Münzen, Schilde und Wunderabschnitte. |
| **Karten & Aktionen** | Welche Karten und Aktionen prägen seine Spielweise? | Kartenname mit Farbsymbol, Zeitalter, Games with a build (%) und Auswahlquote; darunter Aktionen je Zeitalter. |
| **Trainingsverlauf** | Wird das Modell stärker, und was verändert sich dabei? | Punktevorsprung über Trainingsschritte, wählbarer Vergleichsstand und vertiefende Vergleiche. |

### Was beim Öffnen sofort beantwortet werden soll

Ich schlage drei Fragen vor, in dieser Reihenfolge:

1. **Woher kommen die Punkte?** Das farbige Punkteprofil ist das größte und
   zentrale Element. Pro Kategorie stehen der durchschnittliche Punktebeitrag
   und, bei gewähltem Vergleich, eine kompakte Differenz daneben.
2. **Wie stark ist dieser Modellstand?** Eine schmale Zeile zeigt eigene Punkte
   im Mittel, Sieganteil und durchschnittlichen Punktevorsprung gegen den
   stärksten Gegner. Die Zahl ausgewerteter Partien steht einmal beim Modellstand.
3. **Hat das Training zuletzt geholfen?** Ein kompakter Vergleich zum vorherigen
   evaluierten Checkpoint zeigt die Veränderung des Punktevorsprungs samt
   Unsicherheitsintervall, sofern ein gültiger gepaarter Vergleich vorliegt.
   Ein Link führt zum vollständigen Trainingsverlauf.

Beim Öffnen wird der **neueste vollständig evaluierte Stand** des aktiven Laufs
gewählt. Solange es dort noch keinen Bericht gibt, wird die vorhandene feste
Referenz eindeutig als solche angezeigt. Trainingsschritt und Vergleichsbasis
stehen direkt bei der Auswahl. Für die Punkteherkunft kann wahlweise der
stärkste Gegner je Partie oder ein anderer Modellstand als Vergleich dienen;
die Beschriftung muss klar sagen, welche Basis gerade verwendet wird.

Damit bleibt die Punkteherkunft der eigentliche Einstieg. Die bisherige
Overview-Seite entfällt; ihre wenigen nützlichen Gesamtkennzahlen finden in
einer kompakten Zeile Platz. Der vollständige Trainingsverlauf und die
Gegnerliga belegen nicht mehr den oberen Bereich jeder Seite.

### Was erst auf Nachfrage erscheint

- **Verteilungen:** typischer Wertebereich und Maximum je Punktekategorie.
- **Datengrundlage:** legale Kartenangebote, exakte Fallzahlen, Testplan und
  Modellidentität. Bei Kartenquoten beispielsweise im Detail zur jeweiligen Zeile.
- **Untergruppen:** Unterschiede nach Wunder, Sitzposition oder Gegnerpaar,
  erreichbar im Bereich Trainingsverlauf / Vergleiche.
- **Kartenbilder:** kleine Vorschau oder Ansicht bei Auswahl einer Karte;
  Gütersymbole können dagegen direkt in der Zeitaltermatrix stehen.
- **Technische Details und Exporte:** über einen aufklappbaren Bereich statt
  über eine eigene Datenqualitätsseite. Ein tatsächlicher Fehler oder fehlende
  Daten werden weiterhin direkt an der betroffenen Darstellung kenntlich gemacht.

Median, Zero points und Mean when positive werden gemäß deinem Feedback aus
der Oberfläche entfernt, statt in zusätzliche Detailtabellen verschoben zu werden.

## 8. Vorschlag zu Streuung und Unsicherheit

Hier sind zwei verschiedene Fragen gemeint: **Wie unterschiedlich laufen
einzelne Partien?** und **Wie präzise ist der geschätzte Durchschnitt oder
Fortschritt?** Eine Angabe beantwortet nicht automatisch beide Fragen.

| Angabe | Was sie beantwortet | Vorgeschlagener Platz |
| --- | --- | --- |
| **Mittelwert** | Wie viele Punkte erzielt das Modell durchschnittlich? | Direkt im Punkteprofil. |
| **P10–P90** | In welchem Bereich liegen ungefähr die mittleren 80 % der einzelnen Spielergebnisse? | Aufklappbare Verteilung unter dem Punkteprofil, als eine Spalte „Typischer Bereich“. |
| **Maximum** | Welcher höchste Wert wurde in dieser Stichprobe erreicht? | Dort zusätzlich auf Nachfrage; beschreibt eine beobachtete Spitze, keine typische Leistung. |
| **95%-Unsicherheitsintervall der Änderung** | Wie präzise ist die geschätzte Verbesserung zwischen zwei Modellständen? | Direkt neben dem Fortschrittswert; im Verlauf beim ausgewählten Checkpoint. |
| **Spielanzahl** | Auf wie vielen ausgewerteten Partien beruhen die Angaben? | Einmal beim Modellstand; abweichende Teilmengen direkt beim betroffenen Wert. |

**Meine Empfehlung:** Auf der Einstiegsseite nur Mittelwerte, die gemeinsame
Spielanzahl und das Unsicherheitsintervall des Fortschritts zeigen. P10–P90
und Maximum bleiben zum Aufklappen verfügbar. So entsteht keine breite
Statistiktabelle als erste Ansicht.

Beispiel zur Lesart eines Fortschrittswerts:

- **+2,1 Punkte; 95%-Intervall +0,6 bis +3,6:** spricht auf dem verwendeten
  Prüfplan für eine Verbesserung gegenüber dem gewählten Vergleichsstand.
- **+2,1 Punkte; 95%-Intervall −0,8 bis +5,0:** beobachteter Anstieg, aber die
  Richtung ist mit diesen Daten noch nicht eindeutig abgesichert.

Diese Zahlen sind illustrative Beispiele. Der Bereich beschreibt die
Unsicherheit der geschätzten Änderung; er ist weder der Streubereich einzelner
Partien noch eine Aussage darüber, wie viel das Modell künftig besser wird.
Fehlt ein passender Vergleich auf demselben Prüfplan, wird keine vermeintliche
Fortschrittsaussage daraus erzeugt.

Für die Lernkurve empfehle ich den **durchschnittlichen Punktevorsprung** als
anfangs gewählte Kennzahl, weil er neben dem Gewinnen auch die Größe des
Abstands zeigt. Sieganteil und eigene Punkte sind alternativ auswählbar.
Das Intervall zum ausgewählten Fortschrittsvergleich und die feste Referenz
bleiben erreichbar; zusätzliche Kurven müssen nicht gleichzeitig sichtbar sein.

### Ergänzung: Signifikanzsterne als kompakte Anzeige

Dein Vorschlag lässt sich gut mit der schlanken Oberfläche verbinden:
**Änderung und Sterne direkt anzeigen**, genaue Statistik auf Nachfrage.
Eine mögliche, ausdrücklich zu beschriftende Konvention ist:

| Markierung | Zweiseitiger p-Wert |
| --- | --- |
| kein Stern | p ≥ 0,05 |
| \* | p < 0,05 |
| \*\* | p < 0,01 |
| \*\*\* | p < 0,001 |

Beispiel nur zur Gestaltung: **+2,1 Punkte \*\***. Beim Auswählen oder im
Tooltip erscheinen Vergleichsstand, p-Wert, 95%-Intervall und Zahl unabhängiger
Spiel-Seed-Blöcke. Fehlende oder ungültige Tests erhalten eine eigene Kennzeichnung
„nicht getestet“, damit sie nicht mit einem nicht signifikanten Ergebnis verwechselt werden.
Sterne gelten ebenso für eine signifikante Verschlechterung; das Vorzeichen
zeigt die Richtung. Die Zahl der Sterne beschreibt nicht die Größe des Nutzens.
Siehe hierzu die [Einordnung der American Statistical Association](https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf).

**Technischer Abgleich:** Die aktuelle Auswertung berechnet ein gepaartes
95%-Bootstrap-Intervall mit 2.000 Wiederholungen über ganze Spiel-Seed-Blöcke,
aber noch keinen p-Wert. Die drei Sternstufen lassen sich daraus nicht einfach
ablesen. Für die konventionellen Sterne muss ein geeigneter gepaarter Test
ergänzt werden, der dieselbe Abhängigkeit zwischen Sitzrotationen berücksichtigt.

Für das fortlaufende Dashboard sollen solche Vergleiche zunächst als
**explorativ** gelten: Wiederholtes Prüfen vieler Checkpoints auf demselben Panel
macht einen einzelnen Sternfund noch nicht zu einem abschließenden Nachweis
für eine Trainingsmethode. Ein solcher Nachweis benötigt eine gesondert geplante
Prüfung. Die konkrete Testmethode und der Umgang mit mehreren Vergleichen sind
bei der Umsetzung festzulegen.

## 9. Verständnis zur gemeinsamen Prüfung

Der Kern deines Feedbacks ist für mich: **Die Punkteherkunft soll den Einstieg
bilden; Spielaufbau, Kartenentscheidungen und Trainingsentwicklung sollen
erhalten bleiben, aber deutlich leichter lesbar werden.** Weniger Seiten,
weniger wenig hilfreiche Zahlen und mehr visuelle Orientierung sollen die
vorhandenen Erkenntnisse tatsächlich nutzbar machen.

Die früheren Rückfragen zu Neuaufbau und Games with a build sind damit
beantwortet. Zur weiteren Diskussion stehen jetzt die konkreten Vorschläge aus
Abschnitt 7 und 8: vier Bereiche, Punkteprofil als Einstieg, wenige
Gesamtkennzahlen und vertiefende Statistik auf Nachfrage.

## Grundlage des Abgleichs

Der Abgleich erfolgte anhand der lokalen Quellen und gespeicherten Daten,
nicht anhand einer geöffneten Browseransicht:

- [Ansichten, Tabellen und Farben](../../python/src/sevenwonders_rl/evaluation/assets/dashboard.js)
  und [englische Beschriftungen](../../python/src/sevenwonders_rl/evaluation/assets/i18n.js).
- [Aktuell zusammengestelltes Dashboard](../../python/runs/analysis-dashboard/dashboard.html)
  und [Referenzbericht mit 336 Partien](../../python/runs/long-mixed-wonders-v2-seed231/evaluations/step_2097152/analysis.json).
- [Erzeugung der Trainingsübersicht](../../python/src/sevenwonders_rl/evaluation/milestones.py),
  [Dashboard-Konfiguration](../../python/configs/dashboard.json) und
  [Evaluationsleitfaden](../guides/EVALUATION.md).

Für die ursprüngliche Zusammenfassung wurden keine Änderungen an Dashboard
oder Training vorgenommen. Der danach beauftragte Umbau setzt die vier Bereiche,
die lokalen Spielgrafiken, aufklappbare Verteilungen und die gepaarten
Signifikanztests um. Die Lern- und Gegnerauswahlregeln bleiben unverändert.
