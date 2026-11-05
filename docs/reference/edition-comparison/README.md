# Karten und Wunder im direkten Vergleich

Stand: **5. November 2026**. Verglichen wird ausschließlich das Grundspiel:
alte gedruckte Ausgabe, heutige C++-Datendefinitionen und zweite Edition 2020.

Die [vollständige Tabelle](COMPARISON.md) enthält 80 Kartenzeilen und 14
Wunderseiten. Das entspricht 75 alten und 77 neuen Kartentypen; die drei
grauen Typen erscheinen jeweils in Zeitalter I und II. Nach dem BGA-Abgleich
ändern sich bei **17 Kartenzeilen und 11 Wunderseiten** Daten gegenüber der
alten gedruckten Ausgabe. Alle 80 neuen Kartenzeilen sind mit den BGA-
Definitionen abgeglichen. Alle 14 neuen Wunderseiten stimmen mit den
Tafelfotos und BGA überein.

Der lokale grafische Bericht liegt nach dem Erzeugen unter
`python/runs/edition-comparison/index.html`. Im Browser öffnen; eine
Serververbindung ist nicht nötig. Die alten Bilder und die neuen BGA-Sprite-
Sheets sind eingebettet. Es bleiben keine neuen Bildplätze leer. Der Bericht bietet:

- Suche nach Namen, Zeitalter und Farbe sowie Filter auf Editionsänderungen
  und Abweichungen der Engine von der alten Ausgabe.
- Alte und neue Bilder nebeneinander; beide per Klick vergrößern.
- Kosten, Effekte, beide Richtungen der Bauketten und Kopien pro Spielerzahl.
- Startressource und jede einzelne Wunderstufe auf allen 14 Seiten.
- CSV-/JSON-Export und Drucken des gesamten Berichts, auch als PDF.
- Lokale Ordnerauswahl zum probeweisen Ergänzen neuer Bilder ohne Upload.

## Neu erzeugen

Aus `python/`:

```sh
python3 scripts/build_edition_comparison.py
```

Keine zusätzlichen Python-Pakete nötig. Benötigt werden die vorhandenen,
absichtlich nicht versionierten Dateien `references/cards.csv`,
`00_marten/final_cards.csv`, die alten Bilder unter `images/` sowie die
[BGA-Referenz](../../../references/bga-sevenwonders/README.md) unter
`references/bga-sevenwonders/` (Definitionen, Manifest und Sprite-Sheets).
Außerdem liest der Generator die aktuellen Engine- und Paketdaten. Er verändert
keine dieser Eingaben. Er schreibt den Bericht nach `python/runs/` und
aktualisiert die Gegenüberstellung und das Bildmanifest in diesem Ordner.

Die Input-Prüfsummen stehen in [comparison.json](comparison.json).
[catalog_2020.json](catalog_2020.json) enthält die an Tafelfotos erfassten
neuen Wunderseiten und den offiziellen neuen Kettengraphen. Die unabhängigen
BGA-Definitionen werden mit diesen Daten und der lokalen CSV verglichen.
Der Generator prüft insbesondere
die vollständige Inventarlänge, eindeutige Zuordnungen, vorhandene Kettenziele
und die Deckgrößen beider gedruckten Ausgaben für 3–7 Spieler. Außerdem prüft
er alle Asset-Hashes gegen das lokale BGA-Manifest und alle Wunderstufen
gegen die bereits geprüfte Tafeltranskription. Bilder werden über Sprite-
Koordinaten eingeblendet, nicht konvertiert. Die tatsächlichen MIME-Typen
werden aus den Dateibytes erkannt; die BGA-Dateiendungen sind teils irreführend.

## Neue Bilder später einsetzen

[images_manifest.json](images_manifest.json) enthält für jede Position den
alten Bildpfad, BGA-ID und Bildrechteck sowie einen reservierten Pfad für
eigene neue Bilder, die das BGA-Bild optional ersetzen. Beispielstruktur:

```text
neue-bilder/
  cards/
    altar_age1.png
    glassworks_age1.png
    glassworks_age2.png
    well_age1.png
  wonders/
    gizah_a.png
    babylon_b.png
```

Für Wunder bezeichnet `_a` die Tagseite und `_b` die Nachtseite. Diese
Dateibezeichnungen bewahren die Vergleichszuordnung; sie ändern keine Engine-ID.

```sh
python3 scripts/build_edition_comparison.py --new-images /pfad/neue-bilder
```

Eigene neue Bilder werden in die HTML-Datei eingebettet; wo sie fehlen,
bleibt das BGA-Bild sichtbar. Das reine Ergänzen eines eigenen Bildes ändert
keinen Datenwert und keinen Prüfstatus. Die Bilddateien bleiben lokal und
werden nicht in das versionierte Projekt oder eine veröffentlichte Website kopiert.

## Quellenbewertung und gefundene Abweichungen

Die lokale Vorarbeit war bereits vorhanden: `00_marten/final_cards.csv`
enthält die neuen Kartenwerte, aber keine nutzbaren Bauketten. Ihre IDs sind
anders als die Engine-IDs; zudem sind graue Karten über Zeitalter hinweg
zusammengeführt und sechs zusätzliche Gilden enthalten. Der Bericht ordnet
über Namen/Zeitalter zu, übernimmt keine IDs, trennt Grau und schließt die
sechs Nicht-Grundspielgilden aus. Schreibvarianten wie `Univeristy` und
`Bazaar` werden normalisiert.

Die alte lokale Referenz-CSV ist ebenfalls keine fehlerfreie Wahrheit:
Chamber of Commerce nennt darin zwei Stein statt der zwei Lehm im lokalen
Bild, bei Arsenal fehlt 4+, und Shipowners Guild wird fälschlich als
Nachbarwertung beschrieben. Diese Quellenfehler sind im Vergleich ausdrücklich
korrigiert. Die Bilddateien für Glassworks sind anhand ihrer aufgedruckten
Spielerzahlen dem richtigen Zeitalter zugeordnet; ihre Suffixe folgen nicht
der Konvention von Loom und Press.

Vier Kartenzeilen weichen in der Engine von den alten gedruckten Daten ab:

| Karte | Alte gedruckte Daten | Engine aktuell |
| --- | --- | --- |
| Aqueduct | 3 Stein | 3 Lehm |
| Town Hall | 2 Stein + Erz + Glas | 2 Stein + Erz + Papyrus |
| Arsenal | Kopien 3+, 4+, 7+ | Kopien 3+, 7+ |
| University | Kopien 3+, 4+ | Kopien 3+, 5+ |

Diese vier Befunde sind keine Änderungen der neuen Edition. Weitere
Ausführungsfehler, etwa Wissenschaftsjoker, Halikarnassos oder gleichzeitige
Erträge, sind im [Editionsaudit](../EDITION_AUDIT.md) beschrieben. Bei Lighthouse
ist zusätzlich im Bericht vermerkt, dass die heutige Engine die eigene Karte
beim Sofort-Münzertrag noch nicht mitzählt. Die Gegenüberstellung der
Effektdefinitionen ist keine vollständige Prüfung ihrer Ausführung.

Primärquellen sind die offizielle Karten-/Kettenübersicht, Effektübersicht,
FAQ und der öffentliche BGA-Tutorial-Snapshot der zweiten Edition. Die
neuen Wunder wurden außerdem direkt an veröffentlichten Fotos der
Originalkomponenten gelesen. Einzelne Umlauflisten widersprechen den Bildern:
Gizeh Tag Stufe 2 ist **2 Lehm + Stoff**, Babylon Tag Stufe 2 **2 Erz + Stoff**.

Die [BGA-Referenz](../../../references/bga-sevenwonders/README.md) schließt nun
die frühere Quellenlücke bei den Kartenkosten. Dabei wurden vier Werte der
lokalen CSV durch die übereinstimmenden BGA-Definitionen und v2-Bildausschnitte
ersetzt. Alte Werte und Entscheidung sind in
[source_discrepancies.json](source_discrepancies.json) erhalten:

| Karte | Frühere lokale CSV | BGA-Definition und v2-Bild |
| --- | --- | --- |
| Town Hall | 3 Stein + Glas | 2 Stein + Erz + Glas |
| Arena | 2 Lehm + Erz | 2 Stein + Erz |
| Circus | 3 Lehm + Erz | 3 Stein + Erz |
| Craftsmens Guild | 2 Stein + Erz | 2 Stein + 2 Erz |

Der Kostenparser des Vergleichs wurde zusätzlich berichtigt: Er versteht
nicht nur `2x Wood`, sondern auch `+2 Wood`. Dadurch werden Doppelproduzenten
auf beiden Seiten mit Menge 2 gezeigt. Das war ein Fehler des Vergleichs,
keine Änderung der Spielregeln.

BGA behält im Snapshot einige alte oder abweichende Namen bei: `Pawnshop`
steht dort für die Well-Position der zweiten Edition, `Apothocary` wird als
Apothecary und `Craftsmen Guild` als Craftsmens Guild zugeordnet. Die Quellnamen,
BGA-IDs und Sprite-Rechtecke bleiben erhalten. Die Zuordnung verwendet keine
Gleichsetzung mit Simulator-IDs. Bei Wunderseiten ist die ausdrücklich
geprüfte BGA-Reihenfolge 1–14 mit den Engine-Seiten 0–13 verknüpft.

Die statischen Definitionen und Bilder ersetzen keine Prüfung des gesamten
BGA-Serververhaltens. Der Simulator wurde auch durch diesen Abgleich noch
nicht migriert.
