# Korrigierte Spielmerkmale – 23.10.2026

Merkmalssemantik: **Version 3**. Globale Eingabe: **102** Werte, Aktionsvektor:
**99** Werte. Die Modellarchitektur bleibt Version 2; die Eingabedimensionen
werden beim frischen Training aus dem Beobachtungsraum ermittelt.

## Korrekturen

| Merkmal / Datenweg | Vorher | Jetzt |
|---|---|---|
| `is_chain_free_now` | Nicht vorhandenes Feld `chain_prerequisite`, deshalb immer 0 | Prüft gebaute Voraussetzungen aus `chain_from` und `chain_from_any`, einschließlich numerischer/textueller IDs und Ausschluss bereits gebauter Zielkarten |
| `enables_chain_to_count` | Numerische ID an einen nach textuellen IDs indizierten Lookup übergeben | Übergibt `id_name`; direkte mögliche Nachfolger werden korrekt gezählt |
| Baukosten bei Ketten und freiem Handkartenbau | Druckkosten/Handelsschätzung konnten eine kostenlose Aktion als kostenpflichtig beschreiben | Ressourcen- und Münzkosten der Aktion sind 0; erwarteter Münzbestand unverändert |
| Wissenschaft in globaler Eingabe | Nicht gepflegte Engine-Ressourcenfelder `gear`, `compass`, `tablet` | Aus gebauten Karten abgeleitete feste Symbole; identische gemeinsame Auswertung für Actor und Critic |
| Wissenschaftsjoker | In den Symbolzahlen nicht gesondert dargestellt | Eigene und gegnerische Joker separat; Wissenschaftsgilde und bestehende Babylon-Effekte berücksichtigt; keine vorzeitige Wahl eines festen Symbols |
| Vorhandene Handelsrabatte | Aus nicht veröffentlichtem `player.flags` gelesen | Aus gebauten Karten und bestehenden Olympia-B-Effekten abgeleitet |
| Neue Rabattwirkung | Aus nicht vorhandenem `card.flags` gelesen | Liest `effects.discount_flags` |
| Richtung der Rabatte | Linker Beobachtungsnachbar als West interpretiert | Der Beobachtungsnachbar `(seat-1)%3` entspricht Engine-East; `(seat+1)%3` Engine-West |

Die gemeinsame Zustandsauswertung liegt in
`python/src/sevenwonders_rl/features/player_state.py`. Globale Beobachtung,
Aktionsmerkmale und Kostenabschätzung nutzen dieselben Symbole und Rabatte.
Die globale Tabelle behält ihre bisherige Form, ihre Wissenschaftsspalten sind
jetzt korrekt. Hinzu kommen `science_jokers[3]` und `table_discounts[3,3]`.
Aktionsmerkmale ergänzen drei Joker-Zustände, einen Joker-Karteneffekt und drei
vorhandene Rabattflags.

Zwei beim gleichen Datenabgleich gefundene Fehler mit direktem Bezug zur
Produktionsbewertung sind ebenfalls korrigiert:

- Wahlproduktion einer angebotenen Karte verwendet jetzt `produces_choice`
  und `choice_tradable` statt eines nicht vorhandenen verschachtelten Felds.
  Die Merkmale zeigen die Wahlproduktion von beispielsweise Clay Pit und Forum.
- Wunder-ID 0 ist gültig. Die Startproduktion von Gizah wird deshalb auch bei
  der Ableitung des handelbaren Nachbarangebots berücksichtigt.

Freier Handkartenbau erlaubt keine zweite Kopie eines bereits gebauten Gebäudes.
Der kostenlose Bau wird weiterhin über einen eigenen Aktionstyp gewählt;
das Kettenmerkmal beschreibt ausschließlich eine tatsächlich erfüllte Baukette.

## Abgrenzung

Keine neue Belohnung, keine Wissenschaftsboni, keine vorgeschriebene Farbstrategie.
PPO, Explorationskoeffizient und die Auswahl der Trainingsgegner bleiben bestehen.
Die Korrekturen machen reale Spielmöglichkeiten sichtbar; die Anzahl möglicher
Kettennachfolger ist keine Garantie, diese später auf die Hand zu bekommen.

Es wurden **keine C++-Simulatorquellen oder Wunderregeln geändert**. Die vorhandene
Wunderdatenbank wird für die bereits vom Simulator ausgeführten Wissenschafts-
und Rabatteffekte genutzt. Die Migration auf die neuere Wunderedition bleibt
aufgeschoben. Die Handelsabschätzung bleibt eine Schätzung und behauptet keine
allgemeine Identität mit der Engine bei beliebigen komplexen Kaufkombinationen.

## Verifikation

`python/tests/test_feature_semantics.py` prüft:

- Alle in der Kartendatenbank enthaltenen Kettenvoraussetzungen, alternative
  Voraussetzungen und die Abgrenzung von bloßer Bezahlbarkeit.
- Tatsächlich kostenlose Ketten-/Handaktionen und direkte Kettennachfolger.
- Wissenschaft aus gebauten Karten trotz absichtlich falscher Ressourcenfelder,
  getrennte Joker sowie identische Actor-/Critic-Information für alle Positionen.
- Vorhandene und neue Rabatte, Wahlproduktion und Ablehnung alter Merkmalsversionen.
- Kontrollierte Fälle mit der echten Engine für kostenlose Bauketten, Forum mit
  alternativem Vorgänger, beide Richtungen von Rohstoffrabatten, Manufakturrabatt
  sowie Wissenschaftssymbole und Wissenschaftsgilde.

Für diese Engine-Prüfung wird ein temporäres C++-Testprogramm außerhalb des
Repositories kompiliert und gegen die unveränderten Simulatorquellen gelinkt.
Seine Vorher-/Nachher-Statusdaten prüfen eigene Münzkosten und Zahlungen an den
richtigen Nachbarn. Es ändert keine Datei des Simulatorinventars.

## Verwendung

Neue Trainingsläufe benötigen einen frischen Ausgabe-/Ligaordner. Checkpoints
mit alter Merkmalssemantik werden regulär abgewiesen. Eine Weiterverwendung oder
Migration alter Gewichte ist nicht Teil dieser Änderung. Bereits gespeicherte
Auswertungsdaten bleiben über `--report-only` darstellbar, da dabei keine
Modellgewichte geladen werden.

Die bisherigen Trainingsresultate stammen aus Version 2 und belegen daher noch
keine Verbesserung durch diese Korrekturen. Das muss ein neuer Trainingsvergleich
mit Version 3 zeigen.
