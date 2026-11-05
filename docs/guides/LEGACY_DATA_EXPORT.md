# Alte Engine-Daten reproduzierbar exportieren

Stand: 5. November 2026, Paket 2. Die Exporter bilden weiterhin
`legacy-dmag-with-downstream-fixes` ab. Sie aktivieren keine BGA-Zielwerte.
Engine-Pin und `engine.lock.json` bleiben auf
`f39c1d7ead271df437269b7f555a2c7a0d4b9ad1` (`v0.1.1-legacy`). Die lokalen
Paket-1-/Abnahmedokumente in der Engine liegen auf einem neueren Commit;
ihre C++-Definitionen sind mit dem Pin identisch.

## Prüfen und Änderungen ansehen

Aus `python/`, ohne Paketdaten zu verändern:

```sh
python3 scripts/extract_card_db_from_cpp.py --check
python3 scripts/extract_wonder_db_from_cpp.py --check
python3 scripts/audit_edition_data.py --check
```

Die Exportprüfungen benötigen nur Python 3.10+ und die Engine-Quelldateien,
keinen Build und keine RL-Bibliotheken. Der zusätzliche Editionsaudit braucht
für seinen historischen Vergleich außerdem den ursprünglichen Upstream-Commit
im lokalen Engine-Git. CI verwendet die beiden Exportprüfungen auch mit flachen
Submodule-Klonen. `--check` erzeugt keine Dateien und liefert bei abweichendem,
fehlendem oder ungültigem JSON Rückgabewert 1.

Vor einem bewussten Neu-Export zunächst ein separates Verzeichnis verwenden:

```sh
python3 scripts/extract_card_db_from_cpp.py --output-dir /tmp/7wonders-export
python3 scripts/extract_wonder_db_from_cpp.py --output-dir /tmp/7wonders-export
```

Beide Skripte unterstützen `--engine-root /absoluter/pfad/zur/engine`.
Standardpfade werden vom Skriptstandort aus aufgelöst; das aktuelle
Arbeitsverzeichnis muss kein Projektverzeichnis sein. Nach geprüftem Diff
erzeugt der Aufruf ohne `--output-dir` die Paketdateien am bisherigen Ort.
Die Skripte lehnen einen anderen aktiven Regelnamen beziehungsweise eine
implementierte zweite Edition ab. Der BGA-Datengenerator folgt später separat.

## Quellen und bestehendes Schema

| Feld | Maßgebliche Quelle und Abbildung |
| --- | --- |
| Karten-IDs und Typen | `engine/include/card.h`, Enum-Inhalt ohne syntaktisches `enum`/`{` |
| Kartennamen, Zeitalter, Kosten und Ketten | `Card(...)`-Deklarationen in `engine/main.cpp`; acht Kostenfelder in der Reihenfolge aus `engine/src/card.cpp` |
| Kopien | Alle fünf Mengen aus jeder Deklaration; `Card::GetAmount(players)` adressiert `players-3`, also Schlüssel `"3"` bis `"7"` |
| Feste Produktion, Schilde, unmittelbare feste Münzen, Flags | `Player::BuildStructure()` in `engine/src/player.cpp`; tatsächliche `+=`-/`++`-Mengen |
| Produktionsauswahl und bedingte Wertung | Bestehende Legacy-Zuordnung in den Parserfunktionen zu `AvailableCard()`, `CalculateCommercialScore()` und `CalculateGuildScore()`; keine neue allgemeine C++-Interpretation |
| Wissenschaft und feste blaue Punkte | Vorhandene Switch-Auswertung in `CalculateScientificScore()` und `CalculateCivilianScore()` |
| Wunder-IDs und Effektnamen | `engine/include/wonder.h` und `engine/include/resources.h` |
| Wunderstart und Stufen | Konstruktor und `AddStage()` in `engine/src/wonder.cpp`: erste Kosten im Konstruktor, Folgeänderungen im vorherigen erfolgreichen Stufenbau; tatsächlich geforderte Ressourcen aus `ProduceResource()` |
| Mechanikdaten | Bestehender expliziter Legacy-Vertrag in `build_mechanics_json()`; in Paket 2 unverändert |

Das Kartenschema bleibt ein Objekt mit ID-Schlüsseln 1–75. Loom, Glassworks
und Press behalten ihre IDs 15–17 und jeweils `eras=[1,2]` sowie getrennte
`amount_per_player_by_era`-Einträge. Der Bestand umfasst 78 Zeitalterzeilen.
Da ihre Kosten und Ketten im Legacy-Stand über beide Zeitalter gleich sind,
ist kein neues Schema nötig. Unterschiedliche Kosten oder Ketten derselben ID
über mehrere Zeitalter sowie doppelte ID-/Zeitalterzeilen werden jetzt
ausdrücklich abgelehnt, statt Daten still zu überschreiben. Eine spätere
Edition mit unterschiedlichen Alterskosten braucht einen bewussten Vertrag.

`effects.produces_fixed` bleibt für Menge 1 ein Ressourcenstring; Mengen größer
1 werden als Ressourcenobjekt ausgegeben, etwa `{"wood": 2}`. Der vorhandene
Produktionsconsumer unterstützt bereits beide Formen. Featurecode,
Aktionsraum und Beobachtungsdimensionen werden durch dieses Paket nicht umgebaut.

## Geprüfter Daten-Diff

Die ursprünglichen Skripte wurden zuerst in temporäre Dateien exportiert;
Regressionen zeigten 18 Fehlerfälle vor der Korrektur. Der danach geprüfte
JSON-Diff gegen den gepflegten Bestand ist vollständig begrenzt auf:

- Bei allen 75 Kartentypen werden die falschen Spalten 2–5 durch die fünf
  korrekten Spalten 3–7 ersetzt. Jede der 78 Zeitalterzeilen wird gegen ihre
  eigene C++-Deklaration geprüft; keine Menge wird aus BGA-Daten übernommen.
- Sawmill 11, Quarry 12, Brickyard 13 und Foundry 14 erhalten ihre im C++-Code
  vorhandene Produktionsmenge 2. Die bisherige Stringdarstellung bedeutete 1.
- Forum 35 und Caravansery 36 verlieren `manuf_extra`/`raw_extra`, die C++ seit
  der alten Auswahlproduktionskorrektur nicht mehr setzt. Ihre nicht handelbare
  Auswahlproduktion bleibt erhalten.

Alle Kosten, Ketten, Namen, IDs, Farben, Zeitalter, Punkte und sonstigen Effekte
bleiben gleich. Die Wunderdatei wird nach der Enum-Korrektur wieder mit den
richtigen IDs 0–13 erzeugt und ist gegenüber dem gepflegten Bestand bytegleich;
auch `mechanics.json` ist bytegleich. Der bestehende Exportfehler hatte diese
beiden gepflegten Dateien nicht schon in falsche Wunder-IDs umgeschrieben.

Zwei unabhängige Exporte aus einem Archiv des tatsächlich gepinnten
Engine-Commits ergeben bytegleiche Ergebnisse und passen zu den Paketdaten.
Die [Prüfnachweise](../reference/legacy-export-validation.json) enthalten
Vorher-/Nachher-Befunde, Quell- und Ergebnisprüfsummen sowie Tests.

## Verbleibende Grenzen

Die alten Engine-Deckfehler bleiben absichtlich abgebildet: Mit ausgewählten
Gilden ergeben sich für die Zeitalter I/II/III bei 3 Spielern 21/21/21, bei 4
Spielern 28/28/26, bei 5 Spielern 35/35/34, bei 6 Spielern 42/42/41 und bei 7
Spielern 49/49/48 Karten. Die bekannten Arsenal-/University-Abweichungen sind
keine Exportfehler und werden hier nicht als neue Edition korrigiert.
Die Engine unterstützt weiterhin nur drei Spieler.

Die Datenkorrekturen ändern den bereits bestehenden `public_schema_hash()`;
die vorhandenen Checkpoint-/Resume-Prüfungen bleiben aktiv. Ältere Modelle
oder Laufkurven werden hier weder neu bewertet noch automatisch migriert.
Neue Spielregeln, Wissenschaftswertung, GUI und Training bleiben Folgeaufgaben.
