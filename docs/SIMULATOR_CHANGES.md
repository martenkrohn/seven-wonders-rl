# Änderungsprotokoll des C++-Simulators

Stand: 23. Oktober 2026. Diese Datei dokumentiert den gesamten aktuellen Unterschied des Simulatorcodes zur festen Ursprungsfassung, einschließlich Änderungen, die bereits vor der heutigen Self-Play-Arbeit vorhanden waren. Frühere Änderungen werden nicht nachträglich mit einem erfundenen Implementierungsdatum versehen.

## Vergleichsbasis und exakter Nachweis

Ursprung: [dmag-ufsm/7Wonders](https://github.com/dmag-ufsm/7Wonders), Commit [`a0bc333a7a7f21f83156de8a2d522e7baa913f0a`](https://github.com/dmag-ufsm/7Wonders/tree/a0bc333a7a7f21f83156de8a2d522e7baa913f0a). Verglichen wird mit diesem Commit, nicht mit einem veränderlichen GitHub-Branch.

Der vollständige, maschinenlesbare Zeilendifferenz liegt in [simulator/upstream.patch](simulator/upstream.patch). [simulator/manifest.json](simulator/manifest.json) enthält für jede C++-Quelldatei und jeden Header unter `src/`, `include/`, `tests/` sowie `main.cpp` und `CMakeLists.txt` die SHA-256-Prüfsummen und Dateimodi der Ursprungs- und aktuellen Fassung. Auch neue, noch nicht mit Git erfasste C++-Tests sind enthalten. Unveränderte Dateien werden ausdrücklich aufgeführt. Python-Dateien sind eine getrennte Erweiterung und gehören nicht zu diesem Simulatorpatch.

Aus `python/` regenerieren und prüfen:

```sh
.venv/bin/python scripts/audit_simulator.py
.venv/bin/python scripts/audit_simulator.py --check
```

Regeneration benötigt den bereits lokal vorhandenen Upstream-Commit. Die Prüfung benötigt keinen Netzwerkzugriff und keine alte Git-Historie; sie vergleicht den Arbeitsstand mit dem erzeugten Inventar und prüft die Integrität des Patches. Sie ersetzt keine fachliche Prüfung des Erläuterungstextes. Bei jeder weiteren Simulatoränderung müssen dieser Text, Patch und Inventar gemeinsam aktualisiert werden. Ein automatischer CI-Schritt weist veraltete Inventare zurück. Ein bereits vorhandener Patch allein dokumentiert noch nicht den aktuellen Arbeitsstand.

## ENG-001 – Auswahlproduktion nur einmal pro Kostenberechnung

Bereits vor der Dokumentation vom 19. Oktober 2026 vorhanden; genauer Implementierungszeitpunkt unbekannt. Dateien: `include/player.h`, `src/player.cpp`.

Vorher enthielt `Player` die Flags `raw_extra` und `manuf_extra`. Alexandria-Auswahlproduktion konnte innerhalb einer Ressourcenberechnung mehrfach herangezogen werden. Außerdem wurde zusätzliche Rohstoffproduktion über `raw_extra` neben der kartenspezifischen Auswahl gezählt.

Aktuell werden `used_wonder_choice_raw` und `used_wonder_choice_manuf` im Konstruktor mit `-1` initialisiert und in `ResetUsed()` zurückgesetzt. `ChooseExtraRaw()` beziehungsweise `ChooseExtraManuf()` tragen die erste Auswahl ein und verweigern eine weitere Verwendung bis zum Reset. `BuildStructure()` aktiviert für Forum und Caravansery keine parallelen Extra-Flags mehr; deren Auswahl wird über `AvailableCard()` und die bestehenden `used_*`-Felder behandelt. Die zusätzliche `if (raw_extra) cont++`-Zählung entfällt.

Zweck: Eine einzelne Auswahlquelle kann nicht beliebig viele verschiedene Anforderungen derselben Entscheidung bezahlen. Prüfungen: `tests/test_choice_simple.cpp`, `tests/test_choice_once_per_turn.cpp` (CTest `choice_simple`, `choice_once_per_turn`). Die genaue Abgrenzung zwischen Entscheidungen unterschiedlicher Spieler wird zusätzlich durch ENG-006 korrigiert.

## ENG-002 – Caravansery-Auswahl nicht vom Nachbarn beziehen

Bereits vor dem 19. Oktober vorhanden; genauer Zeitpunkt unbekannt. Datei: `src/player.cpp`, Funktion `IncrementOnDemand()`.

Die Caravansery-Abfragen in den Zweigen für Holz, Lehm und Stein verwenden zusätzlich `!is_neighbor`. Die Einschränkung für Erz sowie die vorhandenen Beschränkungen anderer Sonderproduktion sind im exakten Patch beziehungsweise in der Vergleichsbasis nachvollziehbar; sie werden hier nicht als neue Änderung ausgegeben. Kartenspezifische Produktion bleibt über die bestehenden Auswahlmechanismen verfügbar. Zweck ist die Trennung eigener Sonderproduktion von handelbarer Nachbarproduktion. Dies ist keine vollständige Neufassung des Handelsalgorithmus.

## ENG-003 – Vierte Stufe von Gizah B erreichbar machen

Bereits vor dem 19. Oktober vorhanden; genauer Zeitpunkt unbekannt. Datei: `src/wonder.cpp`, Funktionen `Gizah_b::AddStage()` und `Gizah_b::CanAddStage()`.

Beide behandeln die vierte Baustufe jetzt unter `case 3` statt `case 4`: Nach drei errichteten Stufen ist der aktuelle Stufenzähler drei. Kosten, Erträge und die übrigen Wunderdefinitionen wurden dadurch nicht an eine neue Edition angepasst. Der Patch weist genau die beiden Falländerungen aus. Die heutigen Self-Play-Arbeiten ändern diese Wunderdefinitionen nicht. Die bestehenden CTests sind kein vollständiger Test sämtlicher Wunderstufen.

## ENG-004 – Reproduzierbarer Zufall und kontrolliertes Warten

23. Oktober 2026, bereits vor dem aktuellen Self-Play-Umbau. Datei: `main.cpp`.

`Game` erhält einen lokalen `std::mt19937_64`, den effektiven `engine_seed` und `poll_us`. Der Konstruktor akzeptiert Seed und Warteintervall. `GiveCards()` und die Gildenauswahl benutzen `std::shuffle(..., rng)` statt `random_shuffle`; `GiveWonders()` benutzt eine Gleichverteilung von 0 bis 13 statt `rand() % 14`. Das zeitbasierte `srand()` in der Kartenerstellung entfällt.

`main()` akzeptiert `--seed`, `--poll-us` (0 bis 1.000.000 Mikrosekunden) und `--quiet`; Zahlen werden geprüft, Fehler führen zu einer Fehlermeldung auf stderr und Rückgabewert 1. Ohne vorgegebenen Seed wird ein Seed über `random_device` gebildet. `--quiet` deaktiviert stdout-Ausgaben. Die Warteschleife auf die Spielerdateien schläft standardmäßig 1 ms; mit 0 bleibt ein Vergleich mit aktivem Warten möglich.

Auswirkung: Wiederholungen desselben Seeds sind innerhalb derselben Toolchain reproduzierbar; identische alte Zufallsfolgen werden nicht erhalten. Prüfungen: `python/tests/test_engine_protocol.py`, insbesondere Wiederholung mit verschiedenen Austauschpfaden und Warteintervallen. Messungen zum früheren Runtime-Stand: `docs/RUNTIME_BENCHMARK_2026-10-23.json`.

## ENG-005 – Vollständiger Endzustand und atomarer Dateiaustausch

23. Oktober 2026, bereits vor dem aktuellen Self-Play-Umbau. Dateien: `main.cpp`, `src/filer.cpp`.

`WriteGameStatus()` veröffentlicht zusätzlich `protocol_version=2`, `game.phase`, `game.engine_seed` und `last_actions`. Letzteres dokumentiert je Sitz angeforderten/ausgeführten Befehl, Runde und Status `applied`, `skipped` oder `fallback_discard`. Wenn ein Befehl eine Rundennummer enthält, wird eine falsche Runde zurückgewiesen. Das Überspringen der siebten Karte bleibt erhalten und wird jetzt ausdrücklich quittiert.

Nach der letzten Aktion und den bereits in `NextTurn()` ausgeführten Schlusskämpfen werden zuerst die Gildenkopien aller Spieler ausgeführt. Danach wird ein abschließender JSON-Status mit Runde 21, Ära 4 und Phase `finished` veröffentlicht. Erst anschließend schreibt die Engine ihre Ergebnisdatei. Vorher endete die JSON-Beobachtung vor der abschließenden Auswertung; Gildenkopien erfolgten einzeln während des Schreibens der Ergebnisdatei.

`Filer::WriteMessage()` schreibt zunächst `path + ".tmp"`, prüft den Schreibstatus, schließt die Datei und benennt sie mit `std::rename` um. Fehler werfen eine Ausnahme. Dadurch sieht der Leser entweder den alten oder den vollständig veröffentlichten neuen Status. Die Dateinamen und die grundlegende JSON/Ready-Kommunikation bleiben erhalten; dieser Protokollzusatz verlangt den passenden Python-Wrapper.

Prüfungen: `tests/test_final_status.cpp`, Python-Protokolltests einschließlich Prozessende, Timeout, ungültigem JSON und abschließender Veröffentlichung. `friend struct GameTestAccess` in `Game` erlaubt den C++-Tests kontrollierte Zustände ohne zusätzliche öffentliche Spielschnittstelle.

## ENG-006 – Auswahlbelegung zwischen Spielerentscheidungen zurücksetzen

23. Oktober 2026, während der ersten rein neuronalen Trainingsversuche. Datei: `main.cpp`, Funktion `Game::Loop()`, unmittelbar vor dem Ausführen einer nicht übersprungenen Spielerentscheidung.

Beobachtung: Die neuen Läufe mit Trainings-Seeds 7 und 19 brachen ab, weil die Engine eine zuvor als legal veröffentlichte Aktion durch Abwerfen ersetzte. Ein reproduzierter Fall betraf Engine-Seed `17091312795664974005`, Runde 8: Spieler 0 baute eine Alexandria-B-Stufe und benutzte seine Clay-Pit-Auswahl; Spieler 2 konnte anschließend sein zuvor legales Forum nicht mehr ausführen.

Ursache: Die `used_*`-Markierungen der Auswahlproduktion blieben nach der ersten Entscheidung für die weiteren Spieler desselben globalen Zeitschritts belegt. Die eigene Nutzung einer Produktionsquelle durch ihren Besitzer blockierte dadurch einen anschließenden Kauf derselben Quelle durch einen anderen Spieler.

Korrektur: Vor jeder tatsächlich ausgeführten Spielerentscheidung wird für **alle** Spieler `ResetUsed()` aufgerufen. Innerhalb einer Entscheidung bleiben die Markierungen erhalten; eine Auswahlquelle wird nicht für jede einzelne Ressourcenanforderung neu freigegeben. Der bestehende Reset am Rundenende bleibt bestehen. Keine Änderung an Wunderkosten, Wunderfähigkeiten oder der gewünschten Edition.

Regression: `tests/test_joint_action_resources.cpp` stellt den Konflikt in einer kontrollierten letzten Runde her. Alexandria B benutzt eine Clay Pit zum Wunderbau; ein anderer Spieler benötigt dieselbe handelbare Auswahlquelle für sein Forum. Der Test prüft die anfängliche Legalität, den erfolgten Wunderbau und die Quittierung/Errichtung des Forums. Er ist vor der Korrektur nachweislich mit `fallback_discard` fehlgeschlagen. Der Python-Self-Play-Integrationstest verwendet außerdem Seed 7 und einen ausreichend langen ersten Rollout, der den ursprünglichen Fehler erreicht.

Grenze der Korrektur: Sie löst die nachgewiesene Verschleppung der Auswahlbelegung. Sie macht aus der weiterhin sequenziellen Engine keine vollständig neu implementierte simultane Auflösung und garantiert nicht die vollständige Korrektheit sämtlicher Handels-/Sonderregeln.

## ENG-007 – Build, Tests und Herkunftshinweise

`CMakeLists.txt` behält den bereits in der Vergleichsbasis vorhandenen Sprachstandard C++17 bei, ergänzt die Auswahlproduktionstests, die Endzustandsregression und die Regression für gemeinsame Ressourcenentscheidungen und registriert diese bei CTest. Die beiden älteren assert-basierten Tests behalten bei GCC/Clang ihre Assertions auch im Release-Build durch `-UNDEBUG`. Neue Tests mit `require()` werfen auch im Release-Build bei verletzten Bedingungen.

Die Tests für Endzustand und gemeinsame Ressourcenentscheidungen erhalten getrennte Arbeitsverzeichnisse und eine CTest-Zeitgrenze. Die zusätzlichen C++-Tests sind Downstream-Dateien; der Patch enthält sie vollständig. Herkunftskommentare am Anfang von `main.cpp`, `include/player.h`, `src/player.cpp`, `src/wonder.cpp` und `src/filer.cpp` verweisen auf Ursprung und Dokumentation und verändern kein Spielverhalten. Die gebündelten JSON-Header behalten ihre ursprünglichen Hinweise.

## Abgrenzung

Es findet keine pauschale Behauptung vollständiger Regelkorrektheit statt. Insbesondere werden die Wunder-Ausbauschritte nicht zur neuen Edition migriert. Auch Kartenwerte werden durch den aktuellen Self-Play-Umbau nicht geändert. Falls frühere Kartenänderungen nicht im exakten Unterschied zu diesem Upstream-Commit erscheinen, werden sie nicht ohne Nachweis als Änderungen dieser C++-Fassung behauptet. Für jede zukünftige Änderung ist ein neuer Eintrag mit Anlass, genauer Funktion, Wirkung, Test und Grenzen anzulegen und der exakte Nachweis neu zu erzeugen.
