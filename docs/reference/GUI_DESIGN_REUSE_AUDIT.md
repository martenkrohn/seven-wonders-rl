# Übernahme des ursprünglichen GUI-Designs in die neue Edition

Stand: 6. November 2026. Untersucht: Consumer `86d4866`, Engine `e59a3eb`,
Upstream-Basis `a0bc333`, Originalbilder und lokal vorhandene BGA-Bildreferenz.
Dieses Dokument hält die Machbarkeitsprüfung vor der Umsetzung fest.
Der anschließend beauftragte Umbau ist lokal in `engine/gui/decision_view.py`,
`tableau.py` und `bga_assets.py` umgesetzt. Aktuelle Bedienung und Prüfungen:
[GUI-Anleitung](../../engine/docs/GUI.md). Die Umsetzung verwendet eine gemeinsame
vollständige Handreihe, kompakte Effektausschnitte und ausdrückliche Bestätigung;
Standardgröße 1320 × 940, Mindestgröße 1120 × 800 Pixel. Die geometrischen
Alternativen im weiteren Text dokumentieren die ursprüngliche Prüfung.

## Ergebnis und gewünschte Darstellungsgrundlage

Der visuelle Aufbau des Originals ist weitgehend übernehmbar. Die neue Edition
erzwingt andere Daten- und Aktionsanbindungen sowie Änderungen an der räumlichen
Anordnung der Handkarten. Sie erzwingt weder Gebäudetabellen noch textuelle
Ressourcen-/Punkteübersichten oder den Verlust grafischer Wunderstufen.

Die im Gespräch konkretisierte Grundlage lautet:

- Hand und Kartenauswahl: vollständige neue Hochkantkarten.
- Gebaute Gebäude: neue Kartenbilder mit abgeschnittener Illustration, sodass
  kompakte horizontale Effektausschnitte entstehen.
- Drei öffentliche Spielerbereiche mit Wunderbildern, grafischen Stufen,
  Ressourcen-/Punktesymbolen und grafischen Gebäuden bleiben erhalten.
- Neue Sonderentscheidungen werden in dieses Erscheinungsbild integriert.

Die lokale BGA-Referenz benutzt selbst kompakte Effektausschnitte für gebaute
Gebäude und vollständige Karten als Detaildarstellung. Beleg:
`references/bga-sevenwonders/sevenwonders.css`, Regeln `.new_edition .board_item`
und `.board_item_type_*`; `sevenwonders.js`, Funktion `placeItemOnBoard`.
Die CSS-Ansicht verwendet je nach Effekttyp 40 oder 96 Pixel Breite und für
die neue Edition 38 Pixel Höhe. Das ist ein Machbarkeitsbeleg; deren Zahlen
und Spritearithmetik sind kein ungeprüft zu kopierender Zuschnittvertrag.

## Entstehung des Bruchs

Die lokale ursprüngliche `Game.py` ist bytegleich mit der Upstream-Basis.
Commit `f39c1d7` im Engine-Repository stellte das Canvas-Layout zunächst in
`gui/legacy_view.py` wieder her und trennte die Prozessanbindung ab.
Commit `0603e86` ersetzte die aktive Ansicht durch `gui/decision_view.py` und
die Sitzung durch `DecisionSession`; `cd07eec` ergänzte neue Bilddaten.
Die visuellen Unterschiede sind daher wesentlich auf einen neuen Renderer
zurückzuführen. Beide Renderer verwenden weiterhin Tkinter.

## Prüfung nach Oberflächenelement

| Element | Übernahme | Zwingend zu adressieren |
| --- | --- | --- |
| Drei Spielerbereiche | Grundaufbau beibehalten | Relative Nachbarschaft bei Handel und Handweitergabe korrekt kennzeichnen |
| Wunderbild oben | Neue Bilder an derselben grundsätzlichen Stelle | Neues Verhältnis 450 × 223 statt 450 × 198; geprüfte Wunder-ID-/Sprite-Zuordnung |
| Gebaute Wunderstufen als untergeschobene Karten | Visuelle Idee beibehalten | Positionen pro Wunderseite; zwei, drei oder vier Stufen statt einer Drei-Stufen-Formel |
| Ressourcen mit Symbolen und Zahlen | Beibehalten | Feste Produktion von Wahlproduktion unterscheiden; keine vermeintlichen Ressourcenvorräte anzeigen |
| Punktekategorien mit Symbolen | Beibehalten | Werte aus Engine übernehmen; laufender Stand und Schlusswertung unterscheiden |
| Gebaute Karten als Bilder | Beibehalten, neue Effektausschnitte | Farbe, vollständiger Effekt und relevante Kettensymbole erhalten; Namen ergänzen; dynamischer Platzbedarf |
| Hand als auswählbare Kartenbilder | Vollständige neue Karten | Neue Anordnung, passende Größe und Detailvergrößerung für kleine Symbole |
| Auswahlrahmen, ausgegraute Verfügbarkeit | Beibehalten | Verfügbarkeit muss zur gewählten Aktion und zum konkreten angebotenen Modus passen |
| Bauen/Wunder/Verkaufen als kompakte Aktionsleiste | Beibehalten | Neue Freibaumodi unterscheidbar machen; Zahlungen vor Bestätigung zeigen |
| Gemeinsame Zugbestätigung | Möglich | Lokale Auswahl an konkrete Entscheidungen/Revisionen binden; offene Sitze und bereits bestätigte Eingaben beachten |
| Kopfzeile mit Ära/Zug/Weitergaberichtung | Beibehalten | Sonderphase und siebte Karte anzeigen; Richtung aus aktueller Engine-Semantik ableiten |
| Schlussanzeige | Im selben Fenster beibehalten | Erst bei `finished` endgültig, ausstehende Ablagewahlen vorher erledigen |

## Handkarten: die tatsächliche räumliche Einschränkung

Das ursprüngliche Fenster hat 1300 × 705 Pixel, etwa 433 Pixel je Spieler.
Alte Handkarten werden aus 625 × 266 Pixeln ungefähr auf 156 × 66 verkleinert.
Sieben Karten benötigen vertikal einschließlich sechs Abständen à 5 Pixel
etwa 492 Pixel. Die ursprüngliche Darstellung war bereits eine kompakte
Informationskarte und kein vollständiges illustriertes Hochkantbild.

Die neuen Karten haben 255 × 393 Pixel. Beispiele ohne Namenszeilen,
Aktionsleiste und äußere Ränder:

| Kartenbreite | Kartenhöhe ungefähr | Sieben untereinander, 5 px Abstand | Sieben nebeneinander, 8 px Abstand |
| --- | --- | --- | --- |
| 90 | 139 | 1003 | 678 |
| 100 | 154 | 1108 | 748 |
| 120 | 185 | 1325 | 888 |
| 140 | 216 | 1542 | 1028 |

Die alte vertikale Handspalte kann daher bei vergleichbarer Lesbarkeit nicht
unverändert bleiben. Das gilt auch dann, wenn gebaute Gebäude sehr kompakt sind.

Für einen Menschen mit zwei Bots ist eine vollständige Handreihe unter den
drei öffentlichen Spielerbereichen gut unterzubringen: sieben Karten à
120–140 Pixel Breite brauchen etwa 888–1028 Pixel. Kartenhöhe 185–216 Pixel,
zuzüglich Name und kompakter Bedienung. Dies ist eine geometrische Abschätzung,
noch keine visuelle Abnahme auf realen Displays mit unterschiedlichen DPI.

Eine strengere Beibehaltung der Hand im jeweiligen Spielerbereich ist ebenfalls
möglich: bei 90 Pixel Breite passen vier Karten mit 8-Pixel-Abständen in 384
Pixel, also zwei Reihen 4+3 in ungefähr 286 Pixel Höhe, vor Beschriftungen.
Dann brauchen Hand und gebaute Stadt denselben Bereich unter dem Wunder und
müssen räumlich neu aufgeteilt werden. Die kleineren Symbole erfordern eine
gut erreichbare Vergrößerung. Es entsteht ein stärkerer Platz-/Lesbarkeitskonflikt.

Empfohlener erster Entwurf: drei öffentliche Spielerbereiche oben und eine
vollständige aktive Hand unten. Bei mehreren Menschen wechselt die untere
Hand; bei Bots muss deren verdeckte Hand nicht dauerhaft Platz belegen.
Ein offener Analysemodus wäre eine eigene Darstellungsentscheidung.
Das Protokoll selbst erzwingt weder eine gemeinsame Handansicht noch offene
oder verdeckte Hände: der lokale Koordinator erhält alle Hände.

## Gebaute Gebäude: Ausschnitte der neuen Karten

Der Nutzervorschlag ist technisch passend. Im oberen Bereich der neuen Karten
liegen farbiger Hintergrund, Produktion, Wissenschaft, Militär, Punkte,
Handelsrabatte beziehungsweise komplexere Gildeneffekte. Darunter liegt ein
großer Illustrationsteil. Die Einkaufskosten entlang des linken Randes sind
für bereits gebaute Gebäude keine dauerhafte Stadtwirkung.

Für die Umsetzung sind folgende Einzelheiten erforderlich:

1. Einen bewusst gewählten Effektbereich ausschneiden. Nicht das Hochkantbild
   auf Querformat stauchen oder die Illustration als einziges Merkmal behalten.
2. Den Ausschnitt für sämtliche 80 Karten-/Ärazeilen prüfen: besonders breite
   Gildeneffekte, Richtungszeichen und Kettensymbole dürfen nicht abgeschnitten
   werden. Ein einheitlicher oberer Streifen über die gesamte Breite ist ein
   robuster Ausgangspunkt; zusätzliche enge Effektausschnitte brauchen Prüfung.
3. Namen lesbar ergänzen. Die neuen Spritekarten haben eine leere Namensfläche;
   die tatsächlichen Namen kommen aus den Definitionen, nicht aus dem Bild.
4. Bei Hover oder Klick die vollständige Karte mit Name, Effekt und Bauketten
   zeigen. Auf kleinen Ausschnitten sind Details naturgemäß schlechter lesbar.
5. Nach Kartenfarbe und beispielsweise Wissenschaftssymbol sortieren; die
   vorhandenen strukturierten Definitionen liefern Farbe und Effekt.
6. Das Layout aus der tatsächlichen Gebäudeliste berechnen. Keine feste
   18-Karten-Grenze übernehmen; Sonderfähigkeiten verändern Anzahl und Verlauf.

Effektausschnitte sind Bilddarstellung, keine eigene Berechnung der Spielregeln.
Bei fehlenden oder nicht passenden Bildern muss eine kompakte beschriftete
Ersatzdarstellung mit aktuellen Werten verfügbar bleiben.

## Wunderstufen und Symbole

Die alte Stufenformel verteilt Marker mit festen Abständen und funktioniert
laut ursprünglichem Kommentar nur für drei Stufen korrekt. Die neue Edition
enthält Gizah Nacht mit vier sowie Babylon Nacht und Rhodos Nacht mit jeweils
zwei Stufen. Zusätzlich unterscheiden sich die Positionen der Stufenflächen:
Zwei Stufen einfach gleichmäßig zu verteilen trifft nicht jede Tafel.

Benötigt werden skalierbare Positionen je Wunderseite, die anhand der 14 Bilder
visuell geprüft sind. Stufenzahl und Baufortschritt liefert die Engine;
Kosten und Fähigkeiten stehen in `data/generated/wonders-v1.json`.
Die gebaute Karte kann weiterhin grafisch untergeschoben oder die Stufe
markiert werden. Die Engine veröffentlicht nicht die Identität beziehungsweise
das Bauzeitalter jeder vergrabenen Karte im Status. Für einen generischen
Kartenrücken reicht der vorhandene Zustand aus. Eine exakte historische
Rückseitenfarbe beim Wiederherstellen eines Zustands würde zusätzliche
Aufzeichnung oder eine Protokollerweiterung benötigen.

Die vorhandenen Ressourcen- und Punktesymbole sind technisch wiederverwendbar.
Eine neue Edition verlangt keinen neuen Iconstil. Ob alte oder passende neue
Symbole schöner zum Bildmaterial passen, ist eine Gestaltungsentscheidung.

Die Rohstoffzahlen im Status enthalten feste Produktion, keine vollständige
Aufzählung aller möglichen Ressourcen aus Wahlquellen. Beispielsweise ist
„Holz oder Lehm“ eine Quelle, keine gleichzeitige Einheit beider Rohstoffe.
Wahlproduktion und nicht handelbare Wunder-/gelbe Produktion müssen aus den
Gebäuden und Stufeneffekten erkennbar bleiben. Für Aktionslegalität und Zahlung
bleiben ausschließlich die angebotenen Engine-Optionen maßgeblich.
Die Kategorie `commercial` enthält auch Münzpunkte; dieselben Münzpunkte
dürfen in einer zusätzlichen Münzkategorie nicht erneut addiert werden.

## Zusätzliche Bedienung durch die neue Grundlage

Die bekannte Folge „Karte anklicken – Aktion wählen – bestätigen“ bleibt
möglich. Die GUI ordnet diese Auswahl einem angebotenen `option_id` zu;
physische Karteninstanz, Entscheidung und Angebotsrevision bleiben intern.
Die Engine veröffentlicht Optionen für mehrere Spieler und nimmt auch ein
gemeinsames Bundle an. Eine separate Aktionsschaltfläche unter jeder Karte
ist somit keine Protokollvorgabe.

- **Olympia:** Die alte Funktion „einmal je Ära kostenlos“ ist überholt.
  Erste Farbe beziehungsweise erste/sechste Karte einer Ära sind getrennte
  angebotene Modi. Eine Karte kann regulär und über einen Freibaumodus baubar
  sein. Ein kompakter Modusknopf genügt, wenn die Auswahl eindeutig bleibt.
- **Bauketten:** Kostenlosigkeit und Grund sichtbar machen. Verbindungen aus
  neuen Definitionen lesen, keine alten Namens-/ID-Regeln weiterverwenden.
- **Zahlungen:** Gesamtkosten und Empfänger aus dem konkreten Angebot zeigen,
  beispielsweise kleine Münzsymbole mit Nachbarpfeilen. Die aktuelle Engine
  bietet einen kanonischen minimalen Zahlungsplan; eine freie Wahl zwischen
  gleich teuren Handelsplänen wäre zusätzliche Engine-Arbeit.
- **Halikarnassos:** Eigenständige Auswahl aus angebotenen Ablagekarten nach
  der regulären Auflösung. Vollständige Karten derselben Handansicht können
  dafür wiederverwendet werden. Die Ablage kann mehr als sieben Karten
  enthalten; Umbruch oder Scrollen ist erforderlich. Duplikate bleiben
  unselektierbar; freiwilliges Überspringen einer legalen Wahl ist im
  aktuellen Projektvertrag ausgeschlossen. Die Engine überspringt leere
  beziehungsweise vollständig unzulässige Angebote selbst.
- **Babylon:** Eine siebte Karte ist eine echte zusätzliche Entscheidung für
  berechtigte Spieler, keine normale Runde für alle. Das Recht kann durch
  den Wunderbau in Zug sechs noch in derselben Ära entstehen.
- **Zeitpunkt:** Eine neue Entscheidung kann bei unverändertem Zugindex
  erscheinen. Auswahlzustand nach Entscheidungskennung/Angebot verwalten,
  nicht nur beim Wechsel der Zugnummer zurücksetzen.
- **Mehrere Menschen:** Verdeckte Handübergabe kann bleiben; sie verlangt
  keine Änderung der öffentlichen Stadtansichten. Gemeinsame Bestätigung
  ist möglich, wenn lokale Wahlen bis dahin gesammelt werden. Bereits
  übermittelte und angenommene Entscheidungen sind gesperrt.
- **Ende und Abbruch:** Ausstehende Pflichtentscheidungen abwarten;
  abgebrochenen Zustand nicht als normale Schlusswertung anzeigen.

## Architektur und Abnahme einer späteren Umsetzung

`DecisionClient`, `DecisionSession`, die geprüften Spritezuordnungen und die
generierten Daten sind brauchbare Grundlagen. Die Darstellung kann erneut
mit Canvas und kompakten Aktionswidgets aufgebaut werden. Die alte
`legacy_view.py` dient als visuelle Vorlage; ihre globalen Variablen, alten
Statusfelder und Aktionsableitungen sollten nicht unverändert übernommen werden.
Für diese Änderung sind keine neuen Spielregeln oder RL-Features erforderlich.

Eine passende Abnahme muss zusätzlich zu den vorhandenen Funktionsprüfungen
die visuelle Treue ausdrücklich prüfen:

- Alle 80 Effektausschnitte und 14 Wunderseiten vollständig und richtig.
- Zwei-, Drei- und Vier-Stufen-Markierungen an den richtigen Bildpositionen.
- Sieben vollständige Handkarten, wachsende Stadt und große Ablageauswahl.
- Richtige Darstellung bei ursprünglicher Fenstergröße beziehungsweise klar
  begründetem neuem Mindestmaß und bei größerer Schrift-/Displayskalierung.
- Karte plus Aktion, mehrere Freibaumodi, Zahlungsanzeige und Bestätigung.
- Ablagepflicht, siebte Karte, Handübergabe, Ende, Abbruch und fehlende Bilder.
- Öffentliche Gebäude bleiben grafisch; Ressourcen und Punkte bleiben
  symbolbasiert; keine Rückkehr zu den heutigen Tabellen als Hauptdarstellung.

Die geometrischen Vorschläge sind noch kein getesteter Layout-Prototyp.
Die Bildprüfung erfolgte an vorhandenen Dateien und Clientquellen; daraus
folgt kein Vergleich mit laufendem BGA-Servercode.

## Maßgebliche lokale Quellen

- `Game.py`: ursprüngliches Layout und Originalgröße.
- `engine/gui/legacy_view.py`: wiederhergestelltes Layout und bekannte Grenzen.
- `engine/gui/decision_view.py`: heutiger Renderer und konkret ersetzte Elemente.
- `engine/gui/bga_assets.py`: geprüfte Bildzuordnung, Zuschnitte und Ersatztext.
- `engine/gui/decision_session.py`, `engine/sevenwonders_engine/client.py`:
  Trennung zwischen Darstellung, Sitzungen und Transport; Bundle-Unterstützung.
- `engine/src/decision.cpp`: Status, Optionen, Zahlungspläne, Weitergabe,
  Ablageentscheidungen und simultane Auflösung.
- `engine/src/player.cpp`: feste Produktion und aktuelle Punktekategorien.
- `engine/data/generated/cards-v1.json`, `wonders-v1.json`: vollständige
  strukturierte Daten der neuen Edition.
- `engine/docs/PROTOCOL.md`, `KNOWN_ISSUES.md`, `GUI_AUDIT.md`: Verträge und Grenzen.
- `references/bga-sevenwonders/img/`: tatsächlich vorhandenes neues Bildmaterial.
- `references/bga-sevenwonders/sevenwonders.css`, `sevenwonders.js`:
  bestehende kompakte Darstellung gebauter Karten und tafelabhängige Positionen.
