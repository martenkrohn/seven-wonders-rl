# Evaluation: kompakte und detaillierte Ansicht

Die Checkpoint-Auswertung im Trainer und die separat gestartete Auswertung
verwenden dieselbe Punkteaufteilung. Diese Anleitung beschreibt die Bedienung;
die [Forschungsgrundlage](../research/CONTENT_BANK.md#evaluation) beschreibt,
welche Aussagen die Messungen tragen. Stand: 29. Oktober 2026.

## Bereinigte Dashboard-Auswahl

### Neue Oberfläche

Das Dashboard öffnet mit **Punkteherkunft** und bietet vier Bereiche:

- **Punkteherkunft:** farbiges Punkteprofil, eigene Punkte, Sieganteil und
  Punktevorsprung. Vergleich mit dem stärksten Gegner je Partie oder einem
  auf demselben Prüfplan ausgewerteten Modellstand. P10–P90 und Maximum sowie
  Datengrundlage und Export sind aufklappbar.
- **Zeitalter:** Produktionsgüter mit den vorhandenen Spielsymbolen als
  kompakte Matrix für Ende I, II und III. Die Werte sind durchschnittliche
  feste Produktionskapazitäten; Wahlproduktion ist nicht enthalten.
- **Karten & Aktionen:** vorhandene Kartenbilder, Zeitalter, Anteil der Partien
  mit Bau und Auswahlquote. Ein Kartenbild öffnet die größere Darstellung und
  die Anzahl legaler Angebote. Absolute Bauzahlen sind keine eigenen Spalten mehr.
- **Trainingsverlauf:** Lernkurve mit wählbarer Kennzahl und Schrittbereich,
  gepaarte Fortschrittsvergleiche und aufklappbare Modell-, Untergruppen- und
  Ligadetails. Die vollständige Kurve bleibt auch außerhalb des Auswahlfensters
  der zwölf neuesten Detailberichte erreichbar.

Die bisherigen Seiten Overview, Data quality und Science diagnostics entfallen
in der Navigation. Median, Zero points und Mean when positive werden nicht mehr
angezeigt. Gespeicherte Berichte behalten ihre ursprünglichen Statistikfelder.
Dark Mode und Sprachumschaltung bleiben erhalten. Lokal verfügbare Spielbilder
werden direkt in die HTML-Datei eingebettet; die Datei benötigt keinen Server.
Öffentliche Demo-Exporte enthalten weiterhin keine lokalen GUI-Grafiken.

### Fortschritt und Signifikanz

Neue Checkpoint-Berichte enthalten zusätzlich einen gepaarten Vergleich mit
dem vorherigen evaluierten Checkpoint. Dieser ist die Standardauswahl;
die feste Referenz beziehungsweise der Startstand bleibt auswählbar.
Bei älteren Berichten ohne diesen Vergleich wird ein vorhandener gepaarter
Referenz-/Startvergleich verwendet.

Direkt neben der Änderung des Punktevorsprungs stehen Signifikanzsterne:
`* p < 0,05`, `** p < 0,01`, `*** p < 0,001`. Die Vergleichsdetails zeigen den
p-Wert, das vorhandene 95%-Bootstrap-Intervall und die Zahl unabhängiger
Spiel-Seed-Blöcke. Ohne gespeicherten Test steht „Nicht getestet“; aus einem
95%-Intervall werden keine weiteren Signifikanzstufen abgeleitet.

Der zweiseitige Test vertauscht die Modellzuordnung innerhalb ganzer
Spiel-Seed-Blöcke (Vorzeichenwechsel der mittleren gepaarten Differenzen).
Er setzt unter der Nullhypothese symmetrische Blockdifferenzen beziehungsweise
austauschbare Modellzuordnungen voraus. Sitz- und Gegnerrotationen innerhalb
eines Blocks zählen nicht als unabhängige Stichproben. Bis 16 Blöcke werden
alle Vorzeichenkombinationen geprüft, darüber 65.535 reproduzierbare Ziehungen
mit einer Plus-eins-Korrektur. Es wird keine zusätzliche Statistikbibliothek
benötigt. Die Methode folgt dem Prinzip des
[gepaarten Permutationstests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).

Die Sterne sind explorativ und nicht für die wiederholten Checkpoint-Vergleiche
korrigiert. Sie zeigen weder die Größe des Nutzens noch einen abschließenden
Nachweis für eine Trainingsmethode. Der p-Wert stammt aus dem Vorzeichen-Test,
das Intervall aus dem Bootstrap; bei kleinen Stichproben können ihre Aussagen
voneinander abweichen. P10–P90 beschreibt dagegen die Streuung einzelner Partien.

### Auswahl der Berichte

Das normale Dashboard enthält ausschließlich **neue Trainingsauswertungen**
und **eine feste historische Referenz**: die letzte vollständige Evaluation
des Endmodells `long-mixed-wonders-v2-seed231` bei Schritt 2.097.152
(336 Partien). Frühere Zwischenstände, Spezialistenversuche und
Explorationsdiagnosen werden nicht mehr automatisch zusammengestellt.
Ihre Rohdaten und Modelle bleiben als Versuchsnachweise gespeichert.

Die Auswahl wird ausdrücklich in
[dashboard.json](../../python/configs/dashboard.json) festgelegt; Berichts-
und Modellprüfsummen schützen die Identität der Referenz. Der Referenzeintrag
enthält nur dieses Endmodell, keine damaligen Startmodelle oder anderen
Vergleichskandidaten.

```bash
.venv/bin/python scripts/open_dashboard.py
```

Vor der ersten neuen Evaluation öffnet dieser Befehl nur die feste Referenz.
Sobald der neue Dauerlauf sein Dashboard veröffentlicht, wird dieses geöffnet.
Auch der VS-Code-Task „Trainings-Dashboard öffnen (fortlaufender Lauf)“ zeigt
vor dem Trainingsstart die feste Referenz.

**Direkte Vergleiche:** Die alte Evaluation bleibt unverändert mit ihrem
ursprünglichen Prüfplan verfügbar. Weil das neue Training einen anderen
Prüfplan verwendet, wird das feste Referenzmodell beim ersten Evaluieren
einmal zusätzlich auf genau diesem neuen Plan geprüft. Die 96 zusätzlichen
Partien werden unter `reference_evaluation/` im neuen Lauf gespeichert und
bei einer Wiederaufnahme weiterverwendet. Sie beeinflussen das Training nicht.

Jeder neue Detailbericht enthält das aktuelle Modell und dieses feste
Referenzmodell mit gepaartem Vergleich auf denselben Gegnern, Sitzen und
Spiel-Seeds. Der alte Bericht mit anderem Plan wird nicht in die Tabelle
laufübergreifender Vergleiche gemischt. Die kurze Trainingstabelle enthält
die Referenz auf dem neuen Plan; die Gegnerliga ist bei Bedarf aufklappbar.
Die bisherige Spalte „ohne Wissenschaftsspezialist“ ist dort entfernt.

Damit gilt: Die historische Auswahl zeigt den letzten damaligen Befund;
die direkten Fortschrittswerte verwenden die neue, vergleichbare Auswertung
desselben festgehaltenen Modells. „Referenzmodell“ ist eine Arbeitsentscheidung,
kein Nachweis, dass es jede andere vorhandene Policy schlägt.

## Milestone-Evaluation und Dashboard während des Trainings

Die langen Läufe verwenden `--milestone-evaluations` mit einem festen
Referenzpanel. Bei 0/25/50/75/100% entstehen Detailberichte und ein aktualisiertes
gemeinsames Dashboard im Laufverzeichnis. Während der Evaluation pausiert die
Datensammlung; die Evaluation hat getrennte Zufallsströme.

Aus `python/` lässt sich ein vorhandenes Dashboard beispielsweise so öffnen:

```bash
.venv/bin/python scripts/open_dashboard.py --run-dir runs/long-mixed-wonders-v2-seed231
```

Das öffnet den vorhandenen Bericht und startet kein Training. Nach einem neuen
Kontrollpunkt die Browseransicht neu laden. Das Dashboard ist eine gespeicherte
Momentaufnahme; es ist keine unabhängige abschließende Prüfung, wenn wir seine
Ergebnisse für Auswahlentscheidungen verwenden.

Der konkrete letzte Prüfplan und seine Grenzen stehen im
[Ergebnisbericht](../archive/long-training/LONG_TRAINING_V2_RESULTS_2026-10-29.md).

## Ohne Milestone-Modus: wechselnde Vorgänger

`scripts/ppo_training/train_unified.py` zeigt bei jeder Evaluation:

- Testschritt, Spielanzahl und Gegnerart (in Self-Play laufend aktualisierte trainierte Vorgänger).
- Erfolgreiche Spiele, Abbrüche, Fehler, Sieganteil, eigene Punkte und Vorsprung.
- Blau, Militär, Wissenschaft, Gelb, Gilden, Wunder, Münzen und Gesamtpunkte:
  jeweils Mittelwert, P10–P90 und Maximum.
- Wissenschaft: Anteil ohne/mit Punkten und Durchschnitt bei vorhandenen Punkten.
- Gepaarte Änderung zur vorherigen Evaluation nur bei identischem Testplan,
  Dauer und Speicherung des zuletzt erfolgreich geprüften Stands. P10–P90 ist die Streuung einzelner Spiele,
  kein Unsicherheitsintervall des Mittelwerts.

Die Vorgänger sind der neueste streng ältere Snapshot und ein weiterer nahe der
Hälfte des bisherigen Trainings. Anfangsmodell, aktueller Stand und zukünftige
Stände sind ausgeschlossen. Beispiel bei Schritt 32.768 mit Snapshots alle
4.096 Schritte: 28.672 und 16.384. Gibt es nur einen trainierten Vorgänger,
wird nur dieser genutzt; ohne Vorgänger wird die Messung verschoben. Es gibt
keine Evaluation vor dem ersten Training. Die Referenzen werden bei jeder
Evaluation neu bestimmt und mit SHA-256, Schritten und Plan-ID gespeichert.

Wechselnde Gegner erlauben keinen unmittelbaren Vergleich aufeinanderfolgender
Siegquoten. Die Konsole weist das aus und unterdrückt solche Differenzen.
`evaluated.pt` speichert den letzten vollständig geprüften Stand; eine globale
`best.pt`-Auswahl entfällt im Self-Play. Der feste Botmodus bleibt unverändert.

Hier werden keine umfangreichen Karten-/Zeitaltertabellen ausgegeben.
Die automatische letzte Evaluation des Trainings verwendet ebenfalls die
kompakte Ansicht. Alle aktuellen Evaluationen speichern Endzustände aller drei
Spieler. Trainings-Episoden protokollieren diese ebenfalls beim Abschluss.
Aufwendigere Zugprotokolle entstehen nur im separaten Evaluator.

## Separate detaillierte Auswertung

Aus `python/`, mit einem neuen Ausgabeordner:

```sh
.venv/bin/python scripts/eval_selfplay.py \
  --run-dir runs/poc-moderate-seed31-v2 \
  --reference \
    runs/poc-moderate-seed31-v2/checkpoint_65536.pt \
    runs/poc-moderate-seed31-v2/checkpoint_131072.pt \
  --compare \
    runs/poc-moderate-seed31-v2/checkpoint_65536.pt \
    runs/poc-moderate-seed31-v2/checkpoint_131072.pt \
  --blocks 40 --seed 39001 \
  --out-dir runs/poc-moderate-seed31-v2-detailed
```

`--reference` bestimmt die eingefrorenen Gegner. `--compare` bestimmt zusätzliche
Modelle, deren Spielstärke auf genau demselben Gegner-/Seed-/Positionsplan geprüft
wird. Ausgangsmodell und finales Modell werden immer geprüft; das finale Modell
ist standardmäßig `latest.pt`, alternativ `--checkpoint PFAD`.
Mit zwei Gegnerreferenzen umfasst jeder Seed-Block zwölf Spiele pro Modell;
das obige Beispiel erzeugt 1.920 Spiele. Ohne `--compare` sind es 960 Spiele.
`--blocks 1` eignet sich für eine technische Vorschau, liefert aber kein
aussagekräftiges Unsicherheitsintervall.

Die Konsole enthält diese Abschnitte:

1. Datenqualität einschließlich Vollständigkeit der Zeitalter-/Aktionsdaten.
2. Spielstärke aller getesteten Modelle, absolute und gepaarte Unsicherheit.
3. Punkteprofile und Verteilung, Wissenschaftsbeteiligung und Kategorieabstand
   zum selben besten Gegner pro Partie (bei Punktegleichstand niedrigere Position).
4. Kumulierte Wertung nach jedem Zeitalter, Wertungsdeltas, Münzen, Schilde,
   Wunderfortschritt und feste Ressourcenproduktion.
5. Tatsächlich gebaute Kartenfarben und Wunder-/Verkaufsaktionen je Zeitalter.
6. Einzelne Karten mit Bauhäufigkeit und Auswahlquote bei legaler Verfügbarkeit,
   einschließlich angebotener, nie gebauter Karten; Auswahl nach Farbe.
7. Ergebnisse nach Gegnerpaar, Position, Wunder und Spielausgang. Vollständige
   Kategorieprofile dieser Untergruppen stehen in JSON/CSV.
8. Datenbasierte Beobachtungen und Grenzen der Aussagekraft.

Während jeder Modellprüfung gibt es nach jeweils 60 Spielen und am Phasenende
eine Fortschrittszeile mit Laufzeit und geschätzter Restzeit.

### Gespeicherte Dateien

| Datei | Inhalt |
|---|---|
| `summary.json` | Kompakte Ergebnisse, Modellidentitäten und ursprünglicher Vergleich zum Ausgangsmodell |
| `analysis.json` | Alle berechneten Statistiken, Untergruppen, Karten und Zeitalter |
| `report.txt`, `report.md` | Lesbarer Bericht mit denselben Tabellen wie die Detailkonsole |
| `dashboard.html` | Lokales interaktives Dashboard, direkt im Browser öffnen |
| `categories.csv` | Punkteverteilungen je Modell, Zeitalter und Untergruppe |
| `cards.csv` | Karten, Farbe, Zeitalter, Bau- und legale Auswahlhäufigkeiten |
| `initial.jsonl`, `comparison_*.jsonl`, `candidate.jsonl` | Einzelspiele je Modell |
| `games.jsonl` | Zusammengeführte Einzelspiele mit Modellzuordnung |
| `manifest.json` | Konfiguration, Software-/Engine-Identität und Kartendaten-Hash |

### Zustands- und Aktionsdaten

Spielprotokolle verwenden Schema-Version 3. Endzustände enthalten pro Spieler
unveränderte Engine-Punkte und normalisierte Kategorien, Ressourcen, Wunder,
Tischkarten und Farbanzeigen. Münzpunkte werden als `coins // 3` von der
Engine-Kategorie `commercial` abgezogen, um reine gelbe Kartenpunkte zu erhalten.

Im Detailmodus kommen Ausgangszustand und drei Zeitalter-Snapshots hinzu.
Der Übergang von I nach II bzw. II nach III wird am Engine-Status erkannt;
der Snapshot trägt deshalb ein separates `completed_age`. Der Abschluss von
III wird nach der letzten Militärwertung und allen Gildenkopien erfasst.
Es wird nicht nach einer angenommenen Anzahl Policy-Aktionen gezählt.

Aktionsprotokolle betreffen das ausgewertete Modell. Sie enthalten bestätigte
Ausführung, Karte, Farbe, Zug, Zeitalter, legal baubare Karten und Punkt-/Münz-
werte davor/danach. Übersprungene siebte Karten werden nicht als Aktionen gezählt.
Wunderbau und Verkauf zählen nicht als gebautes Gebäude der geopferten Farbe.
Die legalen Baukarten berücksichtigen normalen Bau und den freien Handkartenbau.
Wiederholte Verfügbarkeit zählt als weitere Entscheidungsmöglichkeit.

Die Endzustände aller Spieler ermöglichen Gegnervergleiche; die vollständigen
Einzelentscheidungen der Gegner werden nicht rekonstruiert. Münzdeltas sind
keine exakten Handelsausgaben, da auch Zahlungen der Nachbarn einfließen.
Wahlproduktion wird nicht irreführend als feste Produktion mehrfach addiert.
Die Beobachtungen der Policies erhalten keinerlei zusätzliche Daten.

## Interaktives Dashboard

Jede neue Detailauswertung und `--report-only` erzeugt automatisch eine
`dashboard.html` neben den Berichten. Sie enthält die Auswertungsdaten und alle
benötigten Skripte und Stile in einer Datei. Kein Webserver, keine zusätzlichen
Pakete und keine Netzwerkverbindung sind erforderlich. Die Merkmalskorrektur
für neue Modelle verhindert nicht das Anzeigen vorhandener Auswertungsdaten.

Vorhandene `analysis.json`-Dateien lassen sich auch ohne erneute Reportberechnung
in ein Dashboard übertragen. Aus dem Projekt-Unterordner `python/`:

```sh
cd /home/marten/dev/7Wonders/python
.venv/bin/python scripts/build_dashboard.py \
  --evaluation-dir runs/poc-moderate-seed47-detailed \
  --compare-dir runs/poc-moderate-seed31-v2-holdout
```

Die erzeugte HTML-Datei direkt im Browser öffnen (zum Beispiel über Strg+O).
Standardmäßig öffnet sie auf Englisch. Über „Language“ lässt sich zwischen
English und Deutsch wechseln; eine manuelle Auswahl wird lokal gespeichert.
Beschriftungen, Modellnamen, Zahlenformate und CSV-Ausgaben passen sich an.
Die Auswahl von Auswertung/Modell, Kartenfilter und Tabellensortierung bleiben
beim Sprachwechsel erhalten. Die Trainingskonsole bleibt unverändert.
Oben rechts lässt sich der Dark Mode umschalten; zunächst gilt die
Systemeinstellung, eine manuelle Auswahl wird lokal gespeichert.
Die öffentliche Beispieldemo und ihr Aktualisierungsablauf stehen in
[DASHBOARD_DEMO.md](DASHBOARD_DEMO.md).

Das Dashboard bietet:

- Auswahl von Auswertung und Modellstand mit denselben verständlichen Namen
  und Schrittzahlen wie in der Konsole.
- Überblick mit Sieganteil, Punkten, Vorsprung, Unsicherheitsintervallen und
  einer Grafik zur Punkteherkunft.
- Sortierbare Punkteverteilungen, Wissenschaftsbeteiligung, Zeitalterzustände,
  Ressourcen, Kartenhäufigkeiten und tatsächlich ausgeführte Aktionen.
- Kartensuche, kombinierte Filter nach Farbe und Zeitalter, Seitenauswahl und
  einstellbare Zeilenzahl. Ein Klick auf einen Spaltenkopf ändert die Sortierung;
  ein weiterer Klick kehrt sie um. Zahlen werden numerisch sortiert, fehlende
  Angaben bleiben am Ende.
- CSV-Export der aktuellen Tabelle mit **allen gefilterten und sortierten
  Zeilen**, nicht nur der sichtbaren Seite. Prozentwerte werden in der angezeigten
  Darstellung exportiert; die bestehenden Analyse-CSVs behalten ihre Rohdaten.
- Direkte Modellvergleiche sowie Untergruppen nach Gegnern, Position, Wunder und
  Ausgang, mit wählbarer Punktekategorie.
- Vergleich geladener Auswertungen. Gleiche Dateinamen in anderen Läufen sind
  dabei nicht automatisch dieselben Gegner; unterschiedliche Testpläne sind
  ausdrücklich keine kontrollierten Fortschrittsvergleiche.
- Datenqualität und Download der vollständigen Auswertungsstatistiken.

Über „Auswertung hinzufügen“ können weitere `analysis.json`-Dateien lokal
ausgewählt werden. Diese zusätzlich geladenen Daten bleiben für die aktuelle
Browsersitzung verfügbar. Für dauerhaft gemeinsam eingebettete Auswertungen
den obigen Generator mit `--compare-dir` verwenden. Der Generator verändert
weder Original-Spielprotokolle noch Modellgewichte. Es werden keine neuen
Spiele ausgeführt.

Fehlende Daten werden als „nicht erfasst“ gekennzeichnet. Insbesondere werden
alte kombinierte Handels-/Münzpunkte nicht als reine gelbe Kartenpunkte dargestellt.

## Bestehende Ergebnisse neu darstellen

Ohne neue Spiele und ohne Modelle zu laden:

```sh
.venv/bin/python scripts/eval_selfplay.py \
  --report-only \
  --out-dir runs/poc-moderate-seed31-v2-holdout
```

Hierbei werden nur abgeleitete Berichte neu geschrieben; Original-Spielprotokolle
und `summary.json` bleiben bestehen. Fehlende alte Münz-, Zeitalter- und
Kartendaten erscheinen als nicht erfasst. Insbesondere werden kombinierte
Handels-/Münzpunkte nicht als reine gelbe Kartenpunkte ausgegeben.
Für diese fehlenden Details braucht es eine neue separate Evaluation;
ein erneutes Training ist dafür nicht erforderlich.

## Interpretation

Unsicherheitsintervalle verwenden 2.000 Bootstrap-Resamples ganzer Engine-Seed-
Blöcke. Die mehreren Positionen und Gegnerkombinationen eines Seeds sind keine
unabhängigen Wiederholungen. Untergruppen sind deskriptiv und zeigen ihre
Spielanzahl. Wissenschafts-/Militärpunkte können aus Wunderfähigkeiten entstehen;
Wertungskategorie und Kartenfarbe sind deshalb nicht identisch.

Zwischenstände sind Wertungen des aktuellen Zustands. Bereits gezählte Münzen
können ausgegeben werden, Wissenschaft wächst nichtlinear und Gilden hängen
von weiteren Karten ab. Deltas sind keine kausalen Werte einzelner Entscheidungen.
Das anfängliche Münzvermögen wird bei Wertungsdeltas berücksichtigt.
Die vorhandene Belohnung und der bisherige Umgang mit Punktegleichständen
bleiben unverändert. Die Regelmigration der Wunder bleibt eine eigene Aufgabe.
