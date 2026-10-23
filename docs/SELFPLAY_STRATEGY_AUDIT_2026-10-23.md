# Strategieauswertung und Audit der verfügbaren Spieldaten

Stand: 23.10.2026. Untersucht wurden der Simulator, die Python-Umgebung,
die aktuellen Self-Play-Evaluatoren und die gespeicherten Spiele aus
`python/runs/poc-moderate-seed31-v2` und dessen `-holdout`.
Dieser Audit ändert weder Simulator noch Training oder Belohnung.
Berechnete Kennzahlen stehen in [der JSON-Datei](SELFPLAY_STRATEGY_AUDIT_2026-10-23.json).

## 1. Regelgrundlage und Bedeutung der Kategorien

Referenz: [offizielle deutsche Anleitung der aktuellen Ausgabe, Seiten 2, 5 und 8](https://cdn.svc.asmodee.net/production-rprod/storage/downloads/games/7wonders/de/sev-de02-rules-1598889183fjTCW.pdf),
verlinkt auf der [Verlagsseite](https://www.rprod.com/en/games/7-wonders).
Der Simulator bildet weiterhin die bereits dokumentierte Variante mit alten
Wunderstufen ab. Die Kategorien lassen sich nutzen, ohne diese Migration vorzuziehen.

| Kategorie | Bedeutung | Besonderheit für die Analyse |
|---|---|---|
| Blau/Zivilbauten | Direkte Punkte auf gebauten blauen Karten | Gut als unmittelbarer Punktepfad interpretierbar |
| Rot/Militär | Konfliktmarker nach Vergleichen mit beiden Nachbarn am Zeitalterende | Punktzahl kann negativ sein; Schilde können auch aus Wundern kommen |
| Grün/Wissenschaft | Quadrate der Symbolhäufigkeiten plus 7 je vollständigem Dreierset | Nichtlinear; Wissenschaftsgilden und Wunder können mitwirken |
| Gelb/Handel | Punktebedingungen auf gelben Karten | Münzeffekte, Handel und Rabatte sind zusätzliche wirtschaftliche Effekte |
| Violett/Gilden | Bedingte Punkte abhängig von eigener Stadt und/oder Nachbarn | Wissenschaftsgilde wirkt in die Wissenschaftswertung hinein |
| Wunder | Direkte Punkte vollendeter Abschnitte | Andere Wundereffekte tauchen in anderen Kategorien auf |
| Vermögen | Ein Punkt je vollständigen drei Münzen am Spielende | Separat von gelben Karten auswerten |
| Braun/Grau | Produktion von Rohstoffen und Manufakturprodukten | Kein eigener direkter Wertungstopf; ermöglicht spätere Punkte und Gildenwerte |

Wertungskategorie ist somit nicht identisch mit der Farbe der zuletzt gebauten
Karte oder der kausalen Herkunft eines Punktes. Insbesondere sind Wissenschafts-
und Militärpunkte keine ausschließlichen Belege für grüne bzw. rote Karten.

## 2. Bereits berechenbare Strategie des aktuellen Laufs

Die folgende Tabelle verwendet jeweils 480 vollständige Spiele gegen die
eingefrorenen Modelle bei 65.536 und 131.072 Schritten. Ausgangs- und finales
Modell spielen auf demselben Testplan. Alle 960 Punkteaufschlüsselungen sind
vorhanden und summieren sich exakt zum jeweiligen Agenten-Endstand.

| Punkte je Spiel im Mittel | Untrainiert | Final | Veränderung |
|---|---:|---:|---:|
| Blau | 3,14 | 13,19 | +10,06 |
| Militär | −4,81 | 9,01 | +13,82 |
| Grün/Wissenschaft | 1,78 | 0,14 | −1,64 |
| Handel **einschließlich Münzwertung** | 9,13 | 8,97 | −0,16 |
| Gilden | 1,68 | 3,21 | +1,53 |
| Direkte Wunderpunkte | 6,68 | 6,97 | +0,30 |
| Gesamt | 17,59 | 41,50 | +23,91 |

Rundungsdifferenzen sind möglich. Die Veränderungen betreffen eigene Punkte,
nicht den Punktevorsprung; dessen Änderung beträgt +31,61 Punkte und enthält
auch die veränderten Gegnerendstände.

### Militär und Blau tragen die beobachtete Verbesserung

Der größte deskriptive Beitrag liegt bei Militär, gefolgt von Blau. Das finale
Modell erzielt bei alleinigen Siegen durchschnittlich 11,26 Militärpunkte, bei
Niederlagen 5,73. Auch Gilden und direkte Wunderpunkte sind bei Siegen höher.
Das sind Zusammenhänge innerhalb der Testspiele, kein Nachweis, dass eine
zusätzliche rote Karte in einem bestimmten Zug die beste Aktion gewesen wäre.

Gegen zwei 65.536er-Gegner erzielt das finale Modell 11,06 Militärpunkte, gegen
zwei 131.072er-Gegner nur 6,95. Gleichzeitig steigen die eigenen blauen Punkte
von 12,17 auf 14,62. Das passt zu höherer militärischer Konkurrenz; eine bewusste
strategische Anpassung ist daraus allein nicht bewiesen.

### Wissenschaft wird weitgehend ausgelassen

In 87,1 % der finalen Testspiele fallen gar keine Wissenschaftspunkte an.
Der Mittelwert beträgt 0,14. Auch gegen die unveränderten Anfangsreferenzen
sinkt Wissenschaft von 0,53 Punkten bei Schritt 65.536 auf 0,035 am Ende.
Das ist ein klarer Befund zur Spielweise dieses Laufs, kein Beweis, dass
Wissenschaft generell schlecht ist, ein Implementierungsfehler vorliegt oder
dass die Strategie gegen anders trainierte Gegner robust wäre.

Eine sinnvolle nächste Diagnose wäre, für tatsächlich gewählte und verfügbare
grüne Aktionen die Wissenschaftssymbole, Sets, Bezahlbarkeit und Alternativen
zu erfassen. Die vorhandene Simulatorformel verwendet Symbolquadrate und
Dreiersets; eine vollständige Regelprüfung einschließlich Jokern wurde hier
nicht durchgeführt. Die Belohnung sollte nicht allein wegen des schmalen
Strategieprofils sofort auf bestimmte Farben umgestellt werden.

### Weniger Verkaufen, nicht bloß höhere Punktwerte

Das Ausgangsmodell verkauft durchschnittlich 9,84 Karten je Spiel, das finale
Modell 1,60. Wunderbauaktionen steigen leicht von 2,33 auf 2,44. Die groben
Aktionszähler unterscheiden aber nicht, welche Kartenfarben gebaut wurden oder
ob ein Verkauf wegen Geldnot, fehlender Baumöglichkeiten oder aus strategischen
Gründen erfolgte. Diese Gründe sind aus den bestehenden Daten nicht ableitbar.

Die finalen eigenen Punkte reichen von 23 bis 64; der Median beträgt 41,5,
das 10./90. Perzentil 31,9/51. Ein einzelner Highscore ist daher erheblich
weniger aussagekräftig als die Verteilung auf einem festen Gegnerplan.

## 3. Dateninventar: vorhanden, gespeichert, fehlend

| Information | Simulator/Umgebung liefern sie | Aktuelle Self-Play-JSONL bewahren sie auf |
|---|---|---|
| Endpunkte aller drei Spieler | Ja | Ja, `totals` |
| Punkte nach sechs Simulatorkategorien | Für alle Spieler und jeden Status | Nur Agent am Spielende, `points_breakdown` |
| Münzbestand | Für alle Spieler und jeden Status | Nein |
| Münzpunkte getrennt von gelben Karten | Ableitbar aus Münzen und `commercial` | Nein, Münzen fehlen |
| Karten auf dem Tisch, Farben und Anzahlen | Namen und Anzahlen im Status; Farbnachschlagen in Kartendatenbank | Keine vollständigen Tischkartenlisten |
| Schilde, Wissenschaftssymbole und Produktion | Im Status; ergänzende Ableitung aus Karten/Wundern möglich | Nein |
| Wunder-ID, Name und Ausbaustufe | Für alle Spieler im Status | Nein |
| Zeitalter und globale Zugnummer | Im Status | Keine zeitliche Zustandsfolge |
| Tatsächlich angewendete Aktionen | `last_actions` plus Engine-Bestätigung | Nur grobe Aktionssummen des Agenten |
| Engine-Seed, Spielerposition, Modellidentität, Gegner | Ja | Ja |
| Gewählte und verfügbare Karten je Zug | Im Status/Aktionsmaske | Nein |

Codebelege:

- `main.cpp`, `WriteGameStatus`: Spielerzustände, Ressourcen, Punkte,
  gespielte Karten, Wunder und Zeitalter.
- `src/player.cpp`, `CalculateCommercialScore`: addiert
  `floor(coins / 3)` zu gelben Kartenpunkten. `commercial` deshalb **nicht**
  einfach als reine Gelb-Punkte beschriften.
- `src/player.cpp`, `CalculateScientificScore`: Wissenschaftssymbole,
  Wissenschaftsgilde und babylonische Wunderfähigkeit fließen gemeinsam ein.
- `python/src/sevenwonders_rl/evaluation/runner.py`, `run_game`:
  speichert bei Abschluss `points_breakdown` nur für den Agenten.
- `python/scripts/ppo_training/train_unified.py`, `reset_env`:
  Trainings-Episoden protokollieren Identitäten, Abschluss und Gesamtpunkte;
  noch keine Kategoriehistorie.
- Die Austauschverzeichnisse werden nach Auswertungen entfernt. Bestehende
  Endpunkt-JSONL erlauben daher keine rückwirkend exakte Münzaufteilung oder
  Rekonstruktion von Zeitalterzuständen. Ein gezieltes deterministisches Replay
  wäre ein neues Experiment und muss seine Endergebnisse gegen die Originale prüfen.

Vorhandene ältere Skripte enthalten Teile der Analyse, aber keinen einheitlichen
Weg für den aktuellen neuronalen Self-Play-Prozess. Beispielsweise erwartet
`analyze_jsonl.py` andere Feldnamen; `analyze_eval.py` verwendet rohe Gegnerwerte
als Gruppenschlüssel, obwohl diese bei neuronalen Gegnern Dictionaries sind.
`eval_with_traces.py` behandelt `by_id` teilweise als Dictionary, obwohl es eine
Funktion ist, und enthält Annahmen zur Agentenposition. Diese Werkzeuge sollten
nicht ungeprüft als verlässliche Pipeline übernommen werden.

## 4. Priorisierter Aufbau einer wiederverwendbaren Auswertung

### Priorität 1: Gemeinsames Datenformat und automatischer Endbericht

Eine kleine zentrale Python-Komponente normalisiert ausschließlich Daten aus
dem Engine-Status. Sie wird von regulärer Evaluation, separatem Holdout und
Trainings-Episodenende gemeinsam genutzt. Pro Spieler speichern:

- Position, Modell-/Gegneridentität und SHA-256; Engine-Seed, Protokollversion,
  Simulator-/Kartendaten-Stand sowie Bewertungsziel.
- Unveränderte Engine-Werte als Rohdaten; daneben normalisierte Wertung mit
  `blue`, `yellow`, `treasury`, `military`, `science`, `guild`, `wonder`, `total`.
  Dabei `treasury = coins // 3` und `yellow = commercial - treasury`, eindeutig
  an die aktuell geprüfte Simulatorsemantik gebunden.
- Münzen, Wunder-ID/Seite/Ausbaustufe, Schilde, Kartenzahlen je Farbe und
  Kartenliste. Wissenschaftssymbole und Joker getrennt kennzeichnen, statt
  eine vermeintlich vollständige Symbolzählung aus einzelnen Statusfeldern anzunehmen.
- Fehlende Daten ausdrücklich als nicht verfügbar, nie als Null behandeln.
  Für künftige Daten eine Schema-Version vergeben. Alte Modelle oder alte
  experimentelle Dateiformate müssen dafür nicht nachträglich migriert werden.

Automatischer Bericht nach jeder Auswertung:

1. Technischer Abschluss und Datenvollständigkeit zuerst.
2. Sieganteil, eigene Punkte und Vorsprung einschließlich gruppierter Unsicherheit.
3. Kategorie-Mittelwerte, Streuungen, Median, Perzentile und Nullhäufigkeiten.
4. Agent gegen **denselben** besten Gegner pro Partie, nach dessen Gesamtpunkten
   bestimmt; nicht je Kategorie einen anderen besten Gegner auswählen.
5. Gruppen nach Gegnerkombination, Position, Wunder und Ergebnis; Gruppengröße
   sichtbar halten. Siege/Niederlagen sind beschreibende Gruppen, keine kausalen Tests.
6. JSON für weitere Verarbeitung, CSV für Tabellenprogramme, lesbare Konsole
   und ein dauerhaft speicherbarer Bericht. Diagramme nach Messfrage auswählen.

Das bestehende `summary.json` bleibt eine kompakte Ergebnisübersicht. Die
Rohprotokolle erlauben neue Auswertungen ohne erneute Spiele. Der offline
berechenbare Report soll auch unabhängig von einem neuen Training laufen können.

### Priorität 2: Drei saubere Zeitalter-Snapshots für alle Spieler

Den Zustand am Ende von I und II **nach Militärwertung** sowie den endgültigen
Abschluss von III **nach Militärwertung und finalen Gildeneffekten** speichern.
Aus dem aktuellen Statusfluss lassen sich Übergänge erkennen: Ende I erscheint
mit Beginn von II, Ende II mit Beginn von III, das Spielende mit `phase=finished`.
`completed_age` separat speichern, da das Statusfeld `era` bereits das nächste
Zeitalter bezeichnet und am Ende sogar 4 sein kann.

Die genaue Übergangserkennung an echten Spielen testen, statt lediglich nach
einer festen Anzahl Policy-Aktionen zu zählen: Simulator-Ticks und angewendete
Aktionen sind verschieden; Wunder können Sonderaktionen ermöglichen.

Je Snapshot: Kategorien, Münzen, Produktion, Schilde, Wunderstufe und Tischkarten.
Das ermöglicht Vergleiche wie wirtschaftlicher Aufbau in I, militärischer
Rückstand in II und Punkteumsetzung in III. Änderungen der kumulierten
Militärpunkte geben die jeweilige Zeitalterwertung an.

Zwischenwerte ausdrücklich als **Wertung des aktuellen Zustands** bezeichnen:
Münzen werden später ausgegeben, Wissenschaft wächst nichtlinear, und Gilden
können von späteren Nachbarkarten abhängen. Ein Delta ist damit keine eindeutige
kausale Punktezuordnung zu der letzten Aktion. Eine frühe eigene Produktion mit
null unmittelbaren Punkten kann trotzdem die Grundlage für einen späteren Sieg sein.

### Priorität 3: Optionales Zugprotokoll für ausgewählte Evaluationen

Pro tatsächlich angewendeter Aktion: Spiel-ID, Spieler, Zug, Zeitalter,
Aktionstyp, Karte/ID/Farbe, gewählte Kostenoption, kostenloser Kettenbau,
bestätigtes Engine-Ergebnis und verknüpfte Vorher-/Nachher-Zustände.
Gebäude bauen, Wunderbau und Kartenverkauf müssen getrennt bleiben; die Farbe
der unter ein Wunder gelegten Karte darf nicht als gebautes Gebäude zählen.
Automatisch abgelegte letzte Karten sind ebenfalls keine Verkaufsaktionen.

Für Analysen der Auswahl zusätzlich verfügbare legale Alternativen speichern:
Anteil gewählter grüner Karten bezogen auf tatsächlich verfügbare grüne
Baumöglichkeiten, nicht nur die reine Grün-Anzahl. Schätzwerte für Handel klar
von tatsächlich beobachteten Münzbewegungen trennen. Ein Münzdelta allein
belegt keine eigene Handelsausgabe, da Nachbarn gleichzeitig Geld zahlen können.

Zunächst optional in separaten Evaluationen; kein vollständiges Handkarten-
und Aktionsmaskenarchiv für jeden Trainingszug. Das begrenzt Speicher- und
Laufzeitkosten. Die Auswertung muss Beobachtungsdaten von den tatsächlich an
die Policy übergebenen Informationen trennen; Logging soll keine zusätzlichen
Gegner-Handinformationen in die Modellbeobachtung einspeisen.

### Priorität 4: Feste Vergleichspläne und Diagnose der Strategiebreite

Untrainierte Referenzen als Grundfähigkeitstest behalten; trainierte eingefrorene
Referenzen zur Fortschrittsmessung ergänzen, ohne sie innerhalb einer Messreihe
auszutauschen. Neue Trainings-Seeds und Gegner aus anderen Trainingslinien
liefern zusätzliche Tests gegen die derzeitige Militär/Blau-Spezialisierung.
Kategorien nur bei identischem Gegner-/Seed-/Positionsplan direkt vergleichen.

Ein sinkender Grün-Anteil soll einen Diagnosehinweis auslösen, keine automatische
Abwertung oder Änderung der Belohnung. Gute Spielstärke verlangt nicht in jeder
Partie ausgeglichene Farbanteile. Ein Bericht kann erklären, welche Spielweisen
auftreten; Kausalität oder optimale Entscheidungen erfordern gezieltere Experimente.

## 5. Absicherung und begrenzter Aufwand

Für die ersten drei Stufen sind nach aktuellem Dateninventar keine C++-Änderungen
erforderlich. Kein neuer Simulator-Fork und keine Wunderkorrektur sind dafür nötig.
Sollten später echte fehlende Engine-Ereignisse ergänzt werden, gilt weiterhin
die detaillierte Dokumentationspflicht in `SIMULATOR_CHANGES.md`.

Wesentliche Prüfungen einer späteren Implementierung:

- Alle normalisierten Kategorien summieren sich bei allen drei Spielern zum
  unveränderten Engine-Gesamtstand, einschließlich negativer Militärpunkte.
- Münzaufteilung stimmt bei 0, 2, 3 und weiteren Münzbeträgen; fehlende Münzen
  werden nicht fälschlich als Null oder reine Gelbwertung interpretiert.
- Zeitalter-Snapshots einmalig, korrekt nach Schlachten und vor der nächsten
  Spieleraktion; finaler Snapshot nach allen Endeffekten.
- Protokollierung verändert weder RNG-Verbrauch noch Aktionen, Endstände,
  Gegnerauswahl oder Modellgewichte bei identischem Seed.
- Berichte verarbeiten unvollständige Spiele sichtbar und zählen diese nicht
  als Nullpunktepartien. Bootstrap resampelt vollständige Engine-Seed-Blöcke.
- Vorher-/Nachher-Messung von Laufzeit und Dateigröße mit demselben kurzen
  Spielplan; detaillierte Traces bei Bedarf zuschaltbar.

Empfehlung: Jetzt eine begrenzte, solide Auswertung mit gemeinsamem Schema,
Endpunkten aller Spieler und Zeitalter-Snapshots bauen. Darauf können alle
weiteren Trainingsläufe und Holdouts aufsetzen. Aufwendige Live-Dashboards,
automatische Strategiebewertung und große Trace-Archive erst ergänzen, wenn
die daraus zu beantwortende Frage konkret ist.
