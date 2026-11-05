# Audit: Karten und Wunder gegenüber der Ausgabe 2020

**Nachtrag, 5. November 2026:** Die [vollständige Gegenüberstellung](edition-comparison/README.md)
ergänzt diesen Audit um sämtliche Karten und alle 14 neuen Wunderseiten.
Dabei wurde auch die frühere lokale Vorarbeit `00_marten/final_cards.csv`
gefunden: Neue Kartenwerte sind dort bereits erfasst, aber noch nicht in die
aktive Engine übernommen. Die Aussage zur fehlenden Migration unten bezieht
sich auf die aktiven Spieldaten. Alle neuen Wunderkosten sind inzwischen an
Tafelfotos geprüft. Der anschließend bereitgestellte lokale
`references/bga-sevenwonders/`-Snapshot ergänzt neue Bilder und strukturierte
Definitionen: Alle 80 Kartenzeilen sind damit abgeglichen, vier Kostenwerte
der alten lokalen Vorarbeit wurden korrigiert. Die früher offene Quellenlücke
bei den neuen Kartenkosten ist damit geschlossen. Weitere Abweichungen von
den alten gedruckten Karten sind im neuen Vergleich gesondert markiert.

Stand: 29. Oktober 2026. Geprüft wurde der aktuelle, teilweise noch nicht
committete Arbeitsstand. Zielausgabe ist hier das **Grundspiel der zweiten
Edition von 2020**, ohne Erweiterungen, Promos oder Zweispielervariante.
Die eigentliche Migration ist noch nicht durchgeführt.

## Ergebnis und Größenordnung

Die Karten wurden in dieser Arbeitsfassung **noch nicht auf 2020 umgestellt**.
Alle 78 Kartendeklarationen in `main.cpp` stimmen mit dem festgehaltenen
Upstream-Commit überein. Das gilt für Kosten, Bauketten und Häufigkeitsvektoren.
Die 75 unterschiedlichen Karten-IDs in der Python-Datenbank haben dieselben
Kosten und einfachen Kettenvoraussetzungen wie diese C++-Deklarationen.
Die drei zusätzlichen Deklarationen entstehen durch die grauen Karten in zwei
Zeitaltern. 78 ist also keine Anzahl physischer Karten.

Auch die Wunder bilden weiterhin die alte Ausgabe ab. Gegenüber Upstream
ändert `src/wonder.cpp` außer Herkunftskommentaren ausschließlich zwei
`case 4` zu `case 3`, damit die vierte Gizeh-B-Stufe erreichbar ist.
Diese Reparatur war keine Editionsmigration.

Die Aufgabe ist eine **mittlere, mehrere Schichten betreffende Migration**.
Kosten und feste Punkte sind einfach. Die wesentliche Arbeit liegt in drei
neuen Olympia-Freibauregeln, Babylons geänderter Stufenstruktur,
Halikarnassos' bislang fehlender Agentenentscheidung sowie konsistenter
Auflösung und Wertung. Ein vollständiger Neubau des Simulators ist dafür
nicht erforderlich. Ein Austausch zweier JSON-Dateien reicht jedoch nicht.

| Bereich | Umfang | Bewertung |
| --- | --- | --- |
| Kartenkosten, Häufigkeiten, Bauketten | sämtliche Kartendefinitionen prüfen; konkrete Änderungen unten | kleiner bis mittlerer Datenumbau |
| Kartenpunkte und neue Effekte | drei bestätigte Änderungen blauer Punktwerte, Ludus, Decorators Guild | mittlerer Logik-/Datenumbau |
| Wunderkosten und Startressourcen | 14 Seiten prüfen; zahlreiche Kostenänderungen | überwiegend einfach |
| Olympia Tag/Nacht | drei andere Freibauregeln | mittlerer Logikumbau |
| Babylon Nacht | 3 auf 2 Stufen, frühere Zusatzkarte und Joker | mittlerer Umbau mit Folgestellen |
| Halikarnassos | Auswahl aus Ablage im Agenten und sichere Ausführung | größter eigenständiger Teil |
| Datenexport und Beobachtungen | nachgewiesene Exportfehler, neue Merkmale und Vertragsversion | obligatorischer Begleitumbau |
| Abnahme | Regelfälle, sämtliche Stufen, vollständige Spiele | wesentlicher Teil des Aufwands |

## Nachweise und Grenzen

Vergleichsbasis ist der lokal vorhandene Commit
[`a0bc333a7a7f21f83156de8a2d522e7baa913f0a`](https://github.com/dmag-ufsm/7Wonders/tree/a0bc333a7a7f21f83156de8a2d522e7baa913f0a),
nicht der heutige GitHub-Branch. Historische oder externe Arbeitskopien lassen
sich damit nicht ausschließen; im aktuellen Simulator ist eine frühere
Kartenmigration nicht nachweisbar. Die Änderungen an `Player` betreffen unter
anderem Auswahlproduktion, Handel und Laufzeit; die Punktwert-Switches blieben
im Vergleich zur Basis erhalten.

Der lokale Vergleich ist wiederholbar:

```sh
cd python
python3 scripts/audit_edition_data.py
.venv/bin/pytest -q tests/test_cards_db.py tests/test_wonders_db.py
```

[EDITION_AUDIT_EVIDENCE.json](EDITION_AUDIT_EVIDENCE.json) enthält die
Quellprüfsummen, Deklarationsvergleiche und Regenerationsergebnisse dieses
Audits. Das Skript fängt die Schreibfunktionen der Exporter ab und ersetzt
keine Paketdaten. Die sieben vorhandenen Datenbanktests bestehen; sie sind
überwiegend Strukturprüfungen und belegen keine Editionskonformität.

Externe Quellen, am Auditdatum geprüft:

- [Offizielle Produktseite und Downloads](https://www.rprod.com/en/games/7-wonders).
- [Offizielles Regelheft](https://cdn.svc.asmodee.net/production-rprod/storage/downloads/games/7wonders/en/sev-en02-rules-1716388903zBEZx.pdf).
- [Offizielle Karten- und Kettenübersicht](https://cdn.svc.asmodee.net/production-rprod/storage/downloads/games/7wonders/en/sev-en02-chain-list-1598888877Na4yW.pdf).
- [Offizielle Effektübersicht](https://cdn.svc.asmodee.net/production-rprod/storage/downloads/games/7wonders/en/sev-en02-effects-description-1716388911mLqDj.pdf).
- [Offizielle FAQ](https://www.rprod.com/en/7-wonders/faq).
- [Fotografierte Originalkomponenten: Olympia Nacht und Babylon Nacht](https://vonguru.fr/2020/10/31/7-wonders-fallait-il-reediter-cette-merveille-du-jeu-de-societe/).

Die offiziellen PDFs wurden lokal gelesen, die Karten-/Kettenübersicht und
Effektsymbole zusätzlich als Bilder geprüft. Die beiden Nachtseiten wurden an
Detailfotos der tatsächlichen Komponenten gelesen. Die vollständigen
Kosten sämtlicher Karten stehen **nicht** auf der offiziellen Kettenübersicht.
Auch das Regelheft enthält keine vollständige Matrix aller 14 Wunderseiten.
Dieser Audit ist daher eine belastbare Bestands- und Aufwandsprüfung, aber
**noch kein lückenlos freigegebener 2020er Ersatzdatensatz**. Nicht direkt
abgelesene Einzelkosten bleiben unten ausdrücklich offen.

## Karten: Was muss sich ändern?

Die offizielle Kartenübersicht bestätigt diese Unterschiede zum Projekt:

| Gegenstand | Aktuelles Projekt | Zielausgabe 2020 |
| --- | --- | --- |
| Altar | 2 Punkte | 3 Punkte |
| Theater | 2 Punkte | 3 Punkte |
| Temple | 3 Punkte | 4 Punkte |
| Pawnshop | vorhanden | Well an dieser Stelle im Kartenbestand; kein bloßer Texttausch ohne Kosten-/Kettenprüfung |
| Ludus | fehlt | gelb, 3 Münzen je eigener roter Karte beim Bau und 1 Punkt je eigener roter Karte am Ende |
| Castrum | fehlt | rot, 3 Schilde; ab 4 bzw. 7 Spielern |
| Strategists Guild | Punkte je Niederlagemarker der Nachbarn | fehlt in der neuen Grundspiel-Liste |
| Decorators Guild | fehlt | 7 Punkte bei vollständig gebautem eigenem Wunder |

Die übrigen auf der Übersicht erkennbaren festen blauen Punktwerte stimmen
mit dem Projekt überein. Das ist keine Aussage über deren Kosten.
Arena bleibt die Karte mit Münzen und Punkten je eigener Wunderstufe;
sie darf bei der Ergänzung von Ludus nicht versehentlich ersetzt werden.
Auch Schiffseigner- und Baumeistergilde bleiben grundsätzlich vorhanden.

**Bauketten:** Das neue Diagramm zeigt insbesondere `Well → Statue`,
`Altar → Pantheon` und `Theater → Gardens`. Aktuell stehen dort
`Theater → Statue`, `Temple → Pantheon` und `Statue → Gardens`.
Die Kette `Altar → Temple` entfällt. Die übrigen Verbindungen sind bei der
Migration vollständig gegen das Diagramm zu prüfen; aus diesen Beispielen
folgt keine Vollständigkeit der Änderungsliste. Zeitalter überspringende
Ketten sind durch einen Vorgänger-ID-Vergleich grundsätzlich darstellbar.

**Kartenverteilung:** Beispielsweise ist Tree Farm offiziell erst bei 6
Spielern enthalten, passend zum ersten C++-Vektor an dieser Stelle, aber
nicht zu den Spielerzahlbeschriftungen im Python-Export. Ludus und Castrum
sind bei drei Spielern nicht enthalten, Decorators Guild und die geänderten
blauen Karten wirken dagegen auch im derzeitigen Dreispielerbetrieb.
Für jede Spielerzahl 3–7 müssen die Deckgrößen und Kopien geprüft werden;
Erweiterungen auf mehr als drei aktive Engine-Spieler sind ein eigener Scope.

**Kosten:** Dass sie noch upstream-identisch sind, ist vollständig belegt.
Welche davon von den gedruckten 2020er Karten abweichen, ist noch nicht
vollständig transkribiert. Ein unabhängiger Sollkatalog mit Quelle pro Karte
ist hier das erste Datenarbeitspaket. Auch geerbte Fehler gegenüber der alten
Ausgabe dürfen nicht als Editionsänderung eingeordnet werden: Aqueduct kostet
im aktuellen C++ etwa drei Lehm. Die offizielle neue Kettenübersicht gibt
keine Baukosten an und kann diesen Wert allein nicht verifizieren.

## Wunder: Bestand und Änderungen

Im Folgenden entspricht A der bisherigen Tag-Zuordnung und B der
Nacht-Zuordnung. Die IDs sollen bei der Migration stabil bleiben. Das ist
eine Projektentscheidung, keine Behauptung identischer Fähigkeiten.

### Nachgewiesene strukturelle Änderungen

**Olympia Tag:** Der bisher frei wählbare einmalige Gratisbau pro Zeitalter
wird durch Gratisbau der ersten eigenen Karte jeder noch nicht vorhandenen
Farbe ersetzt. Bereits vor Freischaltung gebaute Farben zählen mit.
Die Starting Resource wechselt von Holz zu Lehm. Die heutige Prüfung in
`CanBuildHandFree()` kennt nur Wunder-ID, Stufe und ein Zeitalter-Flag und
reicht dafür nicht aus. Für Gratisbau muss die Zulässigkeit je Karte gelten.

**Olympia Nacht:** Die fotografierte neue Tafel hat weiterhin drei Stufen:

| Stufe | Kosten 2020 | Effekt 2020 |
| --- | --- | --- |
| 1 | 2 Erz | 2 Punkte und erste Zeitalterkarte gratis |
| 2 | 3 Lehm | 3 Punkte und letzte reguläre Zeitalterkarte gratis |
| 3 | Glas + Papyrus + Stoff | 5 Punkte |

Startressource ist Lehm. Der frühere Rohstoffrabatt und die Gildenkopie
entfallen auf dieser Seite. Die zeitlichen Freibaubedingungen sind durch die
FAQ bestätigt: erster Zug eines Zeitalters beziehungsweise sechster regulärer
Zug des Grundspiels. Ein Wunderbau oder Verkauf im betreffenden Zug hebt die
Freibaumöglichkeit nicht für später auf; sie verfällt für dieses Zeitalter.
Der Bau der ersten Wunderstufe ist selbst kein erster kostenloser Kartenbau.

**Babylon Nacht:** Das Foto zeigt nur noch zwei Stufen. Stufe 1 kostet zwei
Stein und erlaubt die sonst abgeworfene letzte Karte; Stufe 2 kostet drei
Lehm plus Glas und liefert den Wissenschaftsjoker. Die alte erste Stufe mit
drei Punkten entfällt. Die Startressource wechselt von Lehm zu Holz, ebenso
auf der Tagseite. Der Joker ist auf Nacht künftig nach zwei statt drei
gebauten Stufen aktiv. Diese Bedingung ist in C++ derzeit separat codiert.

### Vollständiges Inventar der 14 aktuellen Seiten

Kosten sind pro Stufe mit Semikolon getrennt. Diese Tabelle beschreibt
**exakt den heutigen Projektbestand**, nicht einen fertigen Sollkatalog.

| ID / Seite | Start | Heutige Stufenkosten | Heutige Erträge |
| --- | --- | --- | --- |
| 0 Gizeh A | Stein | 2 Stein; 3 Holz; 4 Stein | 3 / 5 / 7 Punkte |
| 1 Babylon A | Lehm | 2 Lehm; 3 Holz; 4 Lehm | 3 Punkte / Joker / 7 Punkte |
| 2 Olympia A | Holz | 2 Holz; 2 Stein; 2 Erz | 3 Punkte / einmal pro Ära gratis / 7 Punkte |
| 3 Rhodos A | Erz | 2 Holz; 3 Lehm; 4 Erz | 3 Punkte / 2 Schilde / 7 Punkte |
| 4 Ephesos A | Papyrus | 2 Stein; 2 Holz; 2 Papyrus | 3 Punkte / 9 Münzen / 7 Punkte |
| 5 Alexandria A | Glas | 2 Stein; 2 Erz; 2 Glas | 3 Punkte / Rohstoffwahl / 7 Punkte |
| 6 Halikarnassos A | Stoff | 2 Lehm; 3 Erz; 2 Stoff | 3 Punkte / Ablagebau / 7 Punkte |
| 7 Gizeh B | Stein | 2 Holz; 3 Stein; 3 Lehm; 4 Stein + Papyrus | 3 / 5 / 5 / 7 Punkte |
| 8 Babylon B | Lehm | Lehm + Stoff; 2 Holz + Glas; 3 Lehm + Papyrus | 3 Punkte / siebte Karte / Joker |
| 9 Olympia B | Holz | 2 Holz; 2 Stein; 2 Erz + Stoff | Rohstoffrabatt / 5 Punkte / Gildenkopie |
| 10 Rhodos B | Erz | 3 Stein; 4 Erz | 3 Punkte + 3 Münzen + Schild / 4 Punkte + 4 Münzen + Schild |
| 11 Ephesos B | Papyrus | 2 Stein; 2 Holz; Glas + Stoff + Papyrus | 2 / 3 / 5 Punkte, jeweils 4 Münzen |
| 12 Alexandria B | Glas | 2 Lehm; 2 Holz; 3 Stein | Rohstoffwahl / Manufakturwahl / 7 Punkte |
| 13 Halikarnassos B | Stoff | 2 Erz; 3 Lehm; Glas + Stoff + Papyrus | 2 / 1 / 0 Punkte, jeweils Ablagebau |

Für Gizeh, Ephesos, Alexandria und Halikarnassos sind Kostenänderungen zu
erwarten; eine zeitgenössische
[Vergleichsliste](https://opinionatedgamers.com/2020/08/08/dale-yu-first-impressions-of-7-wonders-2020-edition/)
nennt sie ausdrücklich. Sie ist **keine freigegebene Implementierungsvorlage**:
Sie enthält selbst Unsicherheiten und eine auffällige Angabe einer vierten
Alexandria-B-Stufe. Für diese Seiten sowie Babylon Tag sind die exakten
neuen Kosten noch direkt an gut lesbaren Tafeln zu bestätigen. Rhodos ist
ein Kandidat für unveränderte Daten, ebenfalls erst nach Tafeldirektvergleich
freizugeben. Die weiterverwendeten Effektfamilien sind Punkte, Münzen,
Schilde, Auswahlproduktion, Joker und Ablagebau; eine zusätzliche neue
Effektfamilie außerhalb Olympia ist bisher nicht nachgewiesen.

## Bestehende Fehler, die bei der Migration berücksichtigt werden müssen

### P0: Halikarnassos ist für die Policy unvollständig

`SevenWondersEnv._make_command()` sendet immer `extra=""`. Der Aktionsraum
enthält 28 Handaktionen, aber keine Auswahl einer Ablagekarte. Die Fähigkeit
ist damit im üblichen Python-Spielablauf nicht nutzbar, obwohl C++ beim
Wunderbau `DiscardFree(true)` setzt.

Zusätzlich hat die Engine mehrere konkrete Probleme im vorhandenen Pfad:

- `main.cpp` hält nur eine Variable `extra`, die beim Einlesen jedes Sitzes
  überschrieben und anschließend für alle Spieler benutzt wird.
- `BuildDiscardFree()` erhält den Ablagestapel als Kopie. Eine erfolgreiche
  Auswahl wird dadurch nicht aus dem zentralen Stapel entfernt.
- `BuildStructure(c, discard_pile, true)` benutzt den Fundindex aus dem
  übergebenen Stapel am Ende zum Entfernen einer Karte aus `cards_hand`.
  Ein Ablageindex gehört nicht zur Hand; dieser Pfad ist vor Freischaltung
  der Agentenentscheidung zu reparieren.
- Der Berechtigungsmarker wird vor dem Ergebnis des Bauversuchs verbraucht.

Empfehlung: Nach Abschluss aller regulären Aktionen eine eigene
`discard_choice`-Phase veröffentlichen. Der Agent sieht die dann tatsächlich
verfügbaren Karten und wählt eine davon. Keine Kartenwahl vor Kenntnis der
Ablage dieses Zuges erzwingen. Handentnahme, Ablageentnahme und Effektanwendung
werden getrennt. Entscheidungen und Bestätigungen müssen im Protokoll,
Collector und Beobachtungsgedächtnis ausdrücklich abgebildet werden.

### P0: Wissenschaftsjoker werden nicht immer optimal gewertet

`CalculateScientificScore()` verteilt Joker mit einer Heuristik. Ein isolierter
C++-Prüffall mit Workshop, Laboratory, Apothecary und Scientists Guild ergibt
**10 statt 13 Punkten**: Zwei Zahnräder, ein Zirkel und ein Joker sollten zu
2/1/1 Symbolen führen, also `4 + 1 + 1 + 7 = 13`. Die Implementierung macht
daraus 3/1/0 und erhält 10. Das ist ein bestehender Wertungsfehler und keine
Änderung der neuen Edition. Derselbe Algorithmus behandelt Babylons Joker.

Empfehlung: Alle Verteilungen der wenigen Joker auf die drei Symbole
durchprobieren und das Maximum wählen. C++-Endwertung und Python-
Punktevorschau müssen für identische Symbolbestände dasselbe Ergebnis liefern.

### P1: Exporter sind aktuell keine sichere Regenerationsquelle

| Befund | Nachweis | Konsequenz |
| --- | --- | --- |
| Wunder-IDs werden um eins verschoben | `parse_enum()` erkennt `enum{` ohne Leerzeichen nicht als Syntax; es wird als erster Eintrag gezählt | Regeneration erzeugt IDs 1–14 statt 0–13; aktuelle JSON ist noch richtig zugeordnet |
| Spielerzahlen sind falsch beschriftet | C++ `GetAmount(players)` nutzt Index `players-3`; Exporter nennt den ersten Index `2` und behält nur 2–5 | Häufigkeiten um einen Spieler versetzt, Einträge für 6 und 7 fehlen |
| Produktionsmengen gehen verloren | Sawmill/Quarry/Brickyard/Foundry geben in C++ 2 Ressourcen; JSON speichert nur einen Ressourcennamen | `_extract_fixed_production()` interpretiert einen String als Menge 1; Schätzwerte können abweichen |
| veraltete Zusatzflags | Regeneration entfernt bei Forum/Caravansery `manuf_extra`/`raw_extra` | aktuelle Daten sind nicht vollständig mit Exporter reproduzierbar; kein Nachweis einer Editionsmigration |

Die Wunder-Regeneration ist daher zuerst zu reparieren oder durch eine
gemeinsame deklarative Datenquelle abzulösen. Ein ungeprüftes Ausführen des
heutigen Skripts würde eine funktionierende ID-Zuordnung zerstören.

### P1: Simultane Effekte und Nebenwirkungen gezielt absichern

Die Engine führt Spieleraktionen nacheinander aus. Vineyard/Bazar zählen
Nachbarkarten bereits während `BuildStructure()`. Die offizielle Effektübersicht
zählt auch passende Nachbarkarten desselben Zuges. Das Ergebnis kann deshalb
von der Sitzreihenfolge abhängen. Dies ist ein konkreter Prüf- und
Reparaturbedarf; ein vollständiger dynamischer Sitzpermutationstest wurde im
Audit noch nicht ausgeführt.

Für den Umbau sinnvoll: Legalität und Zahlbarkeit am Zustand zu Zugbeginn
prüfen, Ausgaben getrennt erfassen, Gebäude und unmittelbare Effekte auflösen,
anschließend zugabhängige Erträge und Ablageentscheidungen. Auch im selben
Zug erhaltene Handelsmünzen dürfen nicht vorzeitig ausgabefähig werden.
Diese Abgrenzung wird besonders wichtig, sobald Halikarnassos tatsächlich
Karten aus der Ablage baut.

## Empfohlene Umsetzung

1. **Sollkatalog vervollständigen und alten Stand einfrieren.** Alle Karten
   einschließlich Kosten, Kopien, Ketten und Erträgen sowie alle 14 Tafeln
   direkt belegen. Stabile IDs erhalten; neue IDs für neue Karten vergeben.
   Alte Karten nicht still unter ihren IDs umdeuten. Historische Simulator-
   und Datendateien mit Hash erhalten. `ruleset_id` und Datenhash festlegen.
2. **Datenquelle vereinheitlichen.** Empfehlung: ein deklarativer, geprüfter
   Katalog für Kosten, Mengen, Ketten und Wunderstufen; daraus C++-Tabellen
   und Python-Daten erzeugen. Effektfunktionen bleiben Code. Als kleinerer
   Zwischenschritt können reparierte Exporter dienen. Kosten sollen bei
   Legalitätsprüfung und Ausführung dieselbe Definition verwenden; momentan
   sind die Wunderbedingungen in `CanAddStage()` und `AddStage()` dupliziert.
3. **Bestehende Fähigkeiten reparieren.** Joker exakt maximieren,
   Halikarnassos' Datenfluss und Entnahme korrigieren, kritische simultane
   Effekte absichern. Das können bereits vor der Edition getrennt prüfbare
   Änderungen sein.
4. **2020er Karten und Wunder umstellen.** Ludus und Decorators Guild als
   Effekte ergänzen, Castrum in das Militärschema aufnehmen. Olympia-Regeln
   pro Karte/Zug auswerten; Babylon-Stufen und alle hart codierten Schwellen
   aktualisieren. Die 28 Handaktionen können für normales Bauen erhalten
   bleiben; Halikarnassos braucht einen gesonderten Entscheidungsvertrag.
5. **Python-Verträge und Evaluation aktualisieren.** Neue Fähigkeiten in
   `public_state.ABILITIES`, Produktionsmengen, Kartenidentitäten, Masken,
   Deltafeatures, Gegner und Verlauf erfassen. Ruleset und Datenhash in
   Checkpoints, Resume-Prüfung, Pool und Reports verwenden.
6. **Gezielt abnehmen, dann neu trainieren.** Vollständige deterministische
   Regelfälle und kurze Spiele prüfen. Erst danach einen neuen langen Lauf
   starten. Ein Training auf gemischten alten und neuen Regeln erschwert
   die Interpretation der Ergebnisse.

`public_schema_hash()` enthält bereits Karten- und Wunderdaten. Bei Modellen
mit öffentlicher Zustandsrepräsentation werden geänderte Daten dadurch beim
Laden erkannt. Das ist eine gute vorhandene Absicherung, ersetzt aber keine
allgemeine Ruleset-Kompatibilitätsprüfung für sämtliche Modelle, Pools und
Engine-Versionen. Alte Checkpoints bleiben historische Artefakte; ihre Werte
und bisherigen Ergebnisse gelten zunächst für den bisherigen Simulator.

## Abnahmekriterien und Aufwand

| Paket | Mindestnachweis |
| --- | --- |
| Daten | vollständige Quellzuordnung; stabile IDs; reproduzierbare Generierung; Deckgrößen für 3–7; richtige Produktionsmengen |
| Wunder | jede Stufe legal/illegal, korrekte Zahlung und Erträge, keine weitere Stufe nach Abschluss |
| Olympia | fehlende/vorhandene Farben; erster/sechster Zug; kein nachholbarer Gratisbau; Aktivierungsgrenzen |
| Babylon | zwei Nachtstufen, Zusatzkarte bereits nach Stufe 1, Joker nach Stufe 2, Jokerkombinationen |
| Halikarnassos | passende aktuelle Ablage, gleiche-Runde-Verkäufe, Duplicate-Regel, einmalige Entnahme, Ende von Ära III |
| Simultane Effekte | Erträge bei unterschiedlichen Sitzreihenfolgen gleich; Einnahmen dieses Zuges erst danach verfügbar |
| Integration | Engine-Masken und Ausführung stimmen überein; vollständige Spiele; klare Ablehnung inkompatibler Checkpoints |

Grobe Planungsschätzung für Umsetzung **einschließlich Prüfung**, kein
gemessener Zeitaufwand: Sollkatalog und Export/Datenquelle 1–2 Arbeitstage,
Kosten/Ketten/Punkte und einfache Wunder 1–2 Tage, Sonderfähigkeiten und
Phasen 2–4 Tage, Integration und Abnahme 1–2 Tage. Insgesamt eher **5–10
Arbeitstage sorgfältiger Entwicklung** als ein kurzer Kostenpatch; die
Pakete überschneiden sich. Unentdeckte Fehler im Handelsalgorithmus können
den Umfang erhöhen. Training ist darin nicht enthalten.

Die nächste konkrete Arbeit sollte der unabhängige 2020er Sollkatalog sein,
begleitet von der Reparatur der Exporter. Anschließend Sonderregeln und
Engine/Python gemeinsam migrieren. So ist jede Änderung anhand einer
festgelegten Regel und eines gezielten Prüffalls reviewbar.
