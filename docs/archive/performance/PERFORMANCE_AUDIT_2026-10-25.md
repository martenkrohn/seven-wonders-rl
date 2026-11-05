# Performance-Prüfung vom 25. Oktober 2026

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Ziel: mehr verwertbare Spielerfahrung und PPO-Training pro realer Stunde. Dieser Bericht untersucht den aktuellen lokalen Arbeitsstand einschließlich der bereits vorhandenen, noch nicht eingecheckten Änderungen. Es wurden keine Trainings-, Simulator- oder Auswertungsimplementierungen geändert. Neue Messläufe liegen getrennt von bestehenden Experimenten.

## Aufbau und bereits vorhandene Optimierungen

Der produktive Pfad ist `python/scripts/ppo_training/train_unified.py`: Ein CPU-PPO-Lerner steuert einen Sitz, zwei eingefrorene neuronale Snapshots die anderen. Für jedes Spiel startet `SevenWondersEnv.reset()` einen C++-Prozess. Pro Spiel gibt es 21 Protokollübergänge; das ist nicht dasselbe wie 21 tatsächlich ausgeführte Kartenaktionen jedes Spielers. Python berechnet Beobachtungen, Masken und Aktionen aller Sitze, schreibt drei Kommandodateien und ein Bereitschaftssignal und liest den nächsten JSON-Status. Mehrere Engines laufen gleichzeitig, die Python-Verarbeitung der Umgebungen erfolgt überwiegend nacheinander.

Die Rollouts behalten die Zeit- und Umgebungsachsen für GAE; danach folgen PPO-Minibatches. Die Auswertung spielt einen festen Plan vollständig durch und erzeugt optional detaillierte Aktions- und Zeitalterdaten. HTML-Berichte können aus gespeicherten Daten erneut erzeugt werden. Die älteren Trainer sind für diese Untersuchung nicht der maßgebliche Optimierungspfad.

Bereits vorhanden und deshalb kein neuer Verbesserungsvorschlag:

- RAM-Austausch unter `/dev/shm/sevenwonders` mit automatischem Disk-Fallback.
- Gebündelte Gegner-Inferenz, wenn dieselbe gecachte Policy mehrere Anfragen erhält.
- Ein Torch-Rechenthread im Trainer und im Self-Play-Holdout-CLI.
- Seltenere Trainingsevaluationen und ein eigener Modus zur Berichtserzeugung ohne neue Spiele.
- Der tatsächlich verwendete C++-Build ist bereits mit `-O3 -DNDEBUG` kompiliert.
- Isolierte Zufallsströme, Trajektorien-Hashes, Zeitmessungen und Regressionstests für zentrale Verträge.

## Was längere vorhandene Läufe zeigen

Ausgewertet wurden die fünf abgeschlossenen lokalen Läufe mit Feature-Semantik v3 unter `python/runs/`: der moderate Lauf und die vier Science-Varianten. Sie verwenden zwei Umgebungen, RAM-Austausch und gebündelte Gegner-Inferenz. Ihre gemessenen Durchsätze liegen bei etwa 97–116 Lernerübergängen/s, also 351.000–418.000/h einschließlich der im Trainer erfassten Auswertungen. Die Läufe unterscheiden sich in Konfiguration, Diagnoseumfang und Laufbedingungen; ihre Unterschiede sind kein kontrollierter Optimierungsvergleich.

| Zeitbereich | Anteil an der Trainerlaufzeit |
| --- | ---: |
| Sammlung der Spielerfahrung insgesamt | 78–82 % |
| PPO-Updates einschließlich GAE | 14–15 % |
| Trainingsevaluation | 4–8 % |
| Vorbereitung der Gegnerbeobachtungen | 15–16 % |
| Gegner-Netzauswertungen | 16–17 % |
| Zusätzliche Bootstrap-Netzauswertung | 4–5 % |
| Spielabschluss, Gegnerauswahl und Neustart | 7–9 % |

Die letzten vier Zeilen liegen innerhalb der Sammlung und dürfen nicht zu deren Anteil addiert werden. Auch Transport, Engine-Rechenzeit und Arbeit an anderen Engines können sich überlappen. Insbesondere bedeutet `collection.wait` nicht ausschließlich Warten: Der Bereich enthält auch Beobachtungsaufbau.

Im Science-Baseline-Lauf wurden 1.048.576 Gegnerentscheidungen in 955.756 Inferenzgruppen verarbeitet: nur **1,10 Entscheidungen pro Gruppe**. Ein vielfältiger Snapshot-Pool begrenzt also die Wirkung der vorhandenen Bündelung erheblich. Gegnerbeobachtungen plus Aufbau der Lernerbeobachtung belegen zusammen ungefähr ein Viertel der Laufzeit.

## Neue kurze Skalierungsmessung

Acht echte Self-Play-Trainingsläufe: je 2.048 Übergänge, vier PPO-Updates mit jeweils 512 Übergängen, vier Minibatch-Epochen und Batchgröße 256. Zwei Seeds (71/72), Reihenfolge 2/4/8/16 und anschließend umgekehrt. Snapshots alle 512 Übergänge, reguläre Zwischenevaluation deaktiviert, abschließende Evaluation mit einem Seed-Block. Alle Läufe verwendeten automatisch RAM-Austausch und wurden erfolgreich abgeschlossen. CPU: Intel i7-7700, vier physische Kerne, acht logische CPUs unter WSL.

| Umgebungen × Horizont | Durchsatz Seed 71 | Durchsatz Seed 72 | Median beider Läufe |
| --- | ---: | ---: | ---: |
| 2 × 256 | 134,9/s | 133,9/s | 134,4/s |
| 4 × 128 | 166,9/s | 139,3/s | 153,1/s |
| 8 × 64 | 178,6/s | 153,7/s | 166,2/s |
| 16 × 32 | 154,5/s | 149,0/s | 151,8/s |

Die Trainerlaufzeit enthält Sammlung, PPO, Checkpoints und Abschlussevaluation. Import und Prozessinitialisierung vor dem internen Startzeitpunkt sind darin nicht vollständig enthalten. Die extern gemessenen Prozesslaufzeiten lagen bei 14,8–31,7 s; der erste Prozess hatte einen deutlichen Kaltstarteffekt. Für kurze Einmalläufe ist deshalb die externe Zeit maßgeblich. Für lange Trainings ist der Durchsatz innerhalb des Trainers aussagekräftiger.

Mit acht Umgebungen ergaben sich innerhalb der Wiederholungen **15–32 % mehr Übergänge/s** gegenüber zwei Umgebungen; nur die Sammlung war 29–49 % schneller. Vier und sechzehn Umgebungen verbesserten den Gesamtdurchsatz weniger. Das ist ein Hinweis für einen längeren Versuch mit acht Umgebungen, keine allgemeine Geschwindigkeitszusage. Es gab keinen gesonderten ungemessenen Warm-up-Lauf, nur zwei Wiederholungen und einen kleinen Gegnerpool. Unterschiede in GAE-Horizont, Spielverteilung und Modellentwicklung sind enthalten; identische Trajektorien zwischen unterschiedlichen Umgebungszahlen werden nicht behauptet. Es wurde kein belastbarer Spielstärkevergleich durchgeführt.

Die Rohdaten einschließlich Konfigurationen, Logs und neuen Modellen liegen in `/tmp/sevenwonders-performance-audit/`. Die kompakte Messaufzeichnung ist in `PERFORMANCE_AUDIT_2026-10-25.json` gespeichert. Zur Wiederholung jeweils ein frisches Ausgabeverzeichnis verwenden:

```sh
cd python
.venv/bin/python scripts/ppo_training/train_unified.py \
  --mode selfplay --seed 71 --total-timesteps 2048 \
  --n-envs 8 --rollout-horizon 64 --batch-size 256 --minibatch-epochs 4 \
  --eval-every 100000 --eval-games-per-matchup 1 \
  --save-every 100000 --snapshot-every 512 \
  --out-dir runs/performance-check-8env-seed71
```

Für eine belastbarere Nachmessung Budget und Wiederholungszahl erhöhen und einen separaten Warm-up ausschließen. Das vorhandene `scripts/benchmark_runtime.py` verwendet für seinen Trainingsvergleich ausdrücklich **fixed**-Gegner; es erfasst den hier interessierenden neuronalen Self-Play-Pfad nicht direkt.

## Profiling des Merkmalsaufbaus und Kritiker-Mikromessung

Zusätzlich wurden 10.506 vorhandene JSON-Fixtures nach einem Durchlauf zum Aufwärmen für alle drei Sitze profiliert: **31.518 Beobachtungen**, ohne laufendes Training im Hintergrund. `cProfile` erfasste 53,77 s; diese instrumentierte Laufzeit darf nicht mit normalen Trainingszeiten gleichgesetzt werden. Die Aufrufzahlen und relativen Schwerpunkte stützen jedoch die Codebefunde:

| Funktion/Bereich | Kumulative Profilzeit |
| --- | ---: |
| `build_obs` insgesamt | 53,66 s |
| Aktionsmerkmale `build_action_feats` | 41,34 s |
| `estimate_trade_cost` einschließlich Datenaufbereitung | 14,15 s |
| `_build_player_context` | 11,63 s |
| Tabellenmerkmale `build_table_features` | 10,47 s |
| Auflösung gebauter Karten `built_ids` | 8,54 s |
| Eigentlicher Min-Cost-Flow-Algorithmus | 1,45 s |

Auch diese Zeiten überlappen. Es gab rund 3,08 Millionen Namensauflösungen und 7,87 Millionen Karten-ID-Lookups. Der eigentliche Handelsgraph-Solver macht nur etwa ein Zehntel der kumulativen Handelsberechnung aus. Deshalb zuerst Kontextwiederverwendung, feste Indizes und weniger wiederholte Datenumwandlung untersuchen, bevor der Handelsalgorithmus neu geschrieben wird. Die historischen Fixtures sind eine breite technische Stichprobe, keine exakte Gewichtung der Zustände einer aktuellen trainierten Policy.

Eine getrennte Torch-Mikromessung verglich denselben unveränderten ActorCritic mit vollständigem Forward und ausschließlich `value_head(global_mlp(...))`: ein Thread, fünf Wiederholungen mit wechselnder Reihenfolge, je 500 Aufrufe nach Warm-up, Median pro Aufruf. Die Werte waren für die getesteten Eingaben bitgenau gleich.

| Batchgröße | Vollständiger Forward | Nur Kritiker |
| --- | ---: | ---: |
| 2 | 346 µs | 96 µs |
| 8 | 676 µs | 101 µs |
| 16 | 1.132 µs | 107 µs |

Das bestätigt den billigeren Wertpfad lokal. Es ist keine Messung eines umgebauten Trainers; End-to-End-Gewinn und Verhaltensgleichheit eines solchen Umbaus müssen noch geprüft werden.

## Priorisierte Vorschläge

### 1. Anzahl Umgebungen gezielt abstimmen

Zuerst vier und acht Umgebungen prüfen. Mehr Umgebungen verteilen die Kosten eines Lerner-Netzaufrufs auf mehr Entscheidungen, überdecken Engine-Latenz und erhöhen die Chance auf gemeinsame Gegner-Inferenz. Sie parallelisieren jedoch nicht automatisch die Python-Featureberechnung.

Bei Vergleichen die Übergänge pro PPO-Update konstant halten: bisher `2 × 256 = 512`, alternativ `4 × 128` oder `8 × 64`. Minibatch-Größe und Epochenzahl bleiben gleich. Dadurch bleibt die Anzahl der Optimierungsschritte pro Übergang vergleichbar. Trotzdem verändern kürzere Trajektorien den GAE-Horizont und mehr Umgebungen die Spiel-/Seed-Verteilung. Ein schnellerer Lauf muss anschließend auf einem gemeinsamen Holdout-Plan geprüft werden.

Aufwand: gering, zunächst reine Konfiguration. Die Kurzversuche dieser Prüfung sind im Messabschnitt dokumentiert. Vor einem neuen Standard mindestens längere Läufe mit gewachsenem Gegnerpool und mehreren Seeds messen.

### 2. Bootstrap ausschließlich über den Kritiker berechnen

`collect_rollout()` führt nach jedem Simulationsschritt erneut `tensor_observations(following)` und `model.forward_batch(...)` aus. Verwendet werden nur die Zustandswerte. Das Modell berechnet trotzdem sämtliche 28 Aktions-Embeddings und Policy-Logits; zusätzlich werden Aktionsfeatures und Masken erneut gestapelt.

Ein eigener Wertpfad benötigt nur `flatten_global`, `global_mlp` und `value_head`. Für terminale Zustände muss überhaupt kein Wert berechnet werden. Das ändert weder Architekturparameter noch Beobachtungssemantik. Als zweite Stufe lassen sich die nächsten Werte für gewöhnliche, nicht zurückgesetzte Zustände aus der folgenden Lerner-Inferenz übernehmen. Nur Rollout-Ende und gültige externe Trunkierungen benötigen einen gesonderten Bootstrap. Ein Wert aus dem zurückgesetzten Folgespiel darf niemals dafür verwendet werden.

Aufwand: gering für einen reinen Wertpfad, mittel für Wiederverwendung zwischen Schritten. Priorität: hoch als überschaubare, semantisch gut prüfbare Änderung. Der bisherige Bootstrap-Bereich kostet nur 4–5 % des Gesamtlaufs: selbst vollständiges Entfernen allein ergäbe ungefähr 4–5 % mehr Durchsatz. Eine große Beschleunigung dieses einzelnen Netzaufrufs ist kein ebenso großer Trainingsgewinn.

Prüfung: identische Werte, Returns und GAE in gemischten terminalen/trunkierten Batches; gleicher Zustand vor Reset; gleiche Trajektorien- und Gewichts-Hashes bei unveränderter Konfiguration. Vorhandene Collector- und GAE-Tests gezielt erweitern.

### 3. Öffentliche Spielmerkmale einmal berechnen und für alle Sitze verwenden

`build_obs()` baut Tabellenmerkmale und anschließend Aktionsmerkmale. Beide Wege rekonstruieren Wissenschaft und Handelsrabatte aus gebauten Karten. `_build_player_context()` löst zusätzlich für alle drei Spieler Kartenlisten und Nachbarproduktion auf. Das wiederholt sich für die beiden Gegner und den Lerner desselben Spielzustands.

Vorschlag: pro eingelesenem Status einmal unveränderliche Daten für die drei absoluten Sitze erzeugen: Karten-IDs, Wissenschaft/Joker, Rabatte, feste und wählbare Produktion, handelbares Angebot und Wunderstand. Die jeweilige Beobachtung rotiert diese Daten in die vorhandene Reihenfolge `[self, left, right]`. Unveränderliche Kartenmerkmale wie Kosten, Effekte und Folgekettenzahl können beim Laden der Kartendaten vorbereitet werden.

Zusätzlich befüllt `build_action_feats()` dieselben Zustandsmerkmale in einer Schleife für jede legale Aktion. Einmalige Berechnung und anschließendes Broadcasting auf die legalen Zeilen reduziert Python-Schleifen und String-Lookups. Die bestehenden Header und die bisherigen Nullwerte illegaler bzw. leerer Slots müssen exakt erhalten bleiben. Handelsberechnungen nur bei identischen vollständigen Eingaben wiederverwenden; keine vereinfachte Schätzung einführen.

Aufwand: mittel. Priorität: hoch, weil dieser Bereich ungefähr ein Viertel des Gesamtlaufs betrifft. Ein hypothetisch halbierter Aufwand dieses Bereichs entspräche ungefähr 14 % mehr Gesamtdurchsatz, keiner gemessenen Zusage. Beobachtungsgleichheit über den Fixture-Pool und identische Trainings-Trajektorien sind hier die entscheidenden Abnahmekriterien.

### 4. Auswertung über mehrere Spiele bündeln oder auf Prozesse verteilen

`evaluate_model()` verarbeitet den Spielplan mit einer seriellen Schleife. Die vorhandene Gegnerbündelung verbindet höchstens Anfragen innerhalb eines Spiels. Anders als Trainingsepisoden können die einzelnen Auswertungsspiele unabhängig bearbeitet werden.

Zuerst eine Auswertung mit mehreren gleichzeitig aktiven Umgebungen und gemeinsamer Lerner-/Gegner-Inferenz prototypisieren. Alternativ einen kleinen Prozesspool verwenden, etwa zwei bis vier Worker auf der hier vorhandenen CPU mit vier physischen Kernen. Jeder Worker lädt Modelle einmal, verwendet einen Torch-Thread und eigene Austauschverzeichnisse. Datensätze anhand ihrer ursprünglichen Spiel-ID geordnet zusammenführen, damit Plan, Seed-Blöcke und gepaarte Vergleiche erhalten bleiben. Fehlerhafte Spiele weiter sichtbar erfassen.

Aufwand: mittel. Hohe Priorität für umfangreiche Holdout- und Modellvergleiche. Bei den beobachteten Trainingsläufen kostet die Evaluation dagegen nur 4–8 %: Selbst kostenlose Evaluation würde dort nur etwa 4–9 % mehr Gesamtdurchsatz ermöglichen. Die ursprüngliche Anzahl Auswertungsspiele und deren Unsicherheitsintervalle beibehalten.

Berichte und Dashboard nach Möglichkeit aus vorhandenen Datensätzen neu erzeugen; dafür existieren bereits `--report-only` und `build_dashboard.py`. Das beschleunigt wiederholte Analysen ohne neue Simulationen.

### 5. Gegnerpool und Modellcache auf lange Läufe ausrichten

Der Pool kann 80 zulässige Snapshots umfassen, der LRU-Modellcache ist in `sample_selfplay_opponents()` auf 16 begrenzt. Bei Spielneustarts müssen ältere gezogene Snapshots deshalb möglicherweise erneut von Disk geladen und als Torch-Modell aufgebaut werden. Ein bereits eviktiertes, aber noch aktives Modell kann zusätzlich ein zweites Policy-Objekt für denselben Checkpoint erzeugen; die Bündelung nach Objektidentität verbindet diese dann nicht.

Cache-Größe konfigurierbar machen und bei hinreichendem Speicher mindestens alle zulässigen und noch aktiven Referenzen abdecken. Das aktuelle Netz hat nur ungefähr 0,43 MB reine Float32-Parameter je Snapshot; 80 Netze liegen größenordnungsmäßig bei 35 MB reinen Parametern, zuzüglich Python-/Torch-Verwaltungsaufwand. Trainingscheckpoints enthalten zusätzlich Optimizerzustand und sind ungefähr 1,3 MB groß. Eigene Inferenz-Snapshots ohne Optimizerzustand können das Einlesen und Speichern reduzieren.

`OpponentPool._list_checkpoints()` durchsucht und sortiert außerdem bei jeder Gegnerziehung das ganze Verzeichnis. Der Trainer kennt die neuen eigenen Snapshots bereits. Ein aktualisierter In-Memory-Index kann diese wiederholte Arbeit vermeiden, ohne Poolverteilung oder Zufallsziehungen zu ändern. Eine externe Refresh-Funktion kann für andere Poolnutzer erhalten bleiben.

Aufwand: gering bis mittel. Gewinn erst mit großem Pool messen; die kurzen Skalierungsversuche dieser Prüfung haben nur fünf Snapshots und quantifizieren diesen Effekt nicht. Die gesamten 7–9 % unter `collection.reset` enthalten auch Prozessstart, Featurebau und Episodenprotokollierung und dürfen nicht vollständig dem Cache zugerechnet werden.

### 6. Dateiprotokoll und Prozessstart schrittweise ersetzen

Jeder Übergang veröffentlicht drei JSON-Kommandos plus `ready.txt`, die Engine veröffentlicht einen JSON-Status. Python und C++ prüfen die Bereitschaft mit voreingestellten Warteintervallen von jeweils 1 ms. RAM vermeidet einen Teil der Speicherkosten, beseitigt aber weder Dateisystemaufrufe noch JSON-Aufbau, Parsing oder Polling.

Ein persistenter Engine-Worker mit blockierenden Pipes oder einem lokalen Socket könnte alle drei Aktionen in einer Nachricht empfangen und einen Status zurückgeben. Anschließend einen Reset-Befehl ergänzen, um den Prozess über mehrere Spiele zu verwenden. Als größerer Ausbau kommt eine direkte C++-Bibliothek mit `reset(seed)` und `step(actions)` und später numerischen Datenarrays statt JSON infrage. Das Spiel ist derzeit unmittelbar in `main.cpp` mit Dateiausgabe und Ablaufsteuerung verbunden; ein sauberer Kern muss dafür erst herausgelöst werden.

Aufwand: hoch, für direkte Einbettung sehr hoch. Potenziell der nachhaltigste Architekturgewinn; einen konkreten Faktor geben die vorliegenden Messungen nicht her. Bei direkter Einbettung können Mehrkernverarbeitung und GIL-Freigabe relevant werden. Vorhandene Protokollgarantien, Reihenfolge der drei Aktionen, Ressourcen-/Handelsregeln, Seeds und echte Endwertung müssen mit festen Aktionsfolgen gegen den bestehenden Simulator geprüft werden. Der bisherige Dateipfad kann als Referenz erhalten bleiben.

`--engine-poll-us 0` ist kein Ersatz: Das erzeugt eine C++-Warteschleife und lässt das Python-Polling unverändert. Kleinere Warteintervalle nur anhand von Wandzeit **und** CPU-Verbrauch vergleichen.

### 7. Doppelte Simulatorarbeit entfernen

`--quiet` setzt derzeit nur den Fehlerstatus von `std::cout`. In `Game::Loop()` werden trotzdem alle Ausdrücke für die Debugausgabe ausgewertet. Dazu gehören pro Sitz ein weiterer Aufruf von `GetPlayableCards()` und `CanBuildWonder()`, obwohl `WriteGameStatus()` diese Informationen unmittelbar zuvor berechnet hat. `GetPlayableCards()` prüft Ressourcen und Handel und sichert/wiederherstellt Spieler- und Nachbarzustände; das ist mehr als eine billige Abfrage.

Die gesamten Debugblöcke hinter einem tatsächlichen Quiet-Schalter überspringen oder die bereits ermittelten Resultate wiederverwenden. Daneben ruft `WriteGameStatus()` alle Score-Kategorien einzeln auf und danach `CalculateScore()`, das dieselben Kategorien erneut berechnet. Ein gemeinsam erzeugtes Score-Ergebnis kann diese Dopplung vermeiden. Karten und Ressourcen werden außerdem häufig als vollständige Container kopiert; das ist eine spätere native Profiling-Aufgabe.

Aufwand: gering bis mittel für die klaren Dopplungen. Noch keine isolierte native CPU-Messung, deshalb keine bezifferte Beschleunigung. Insbesondere Legalitätsprüfungen haben interne Zustandswirkungen: Vorher/nachher alle Statusdaten und Aktionsbestätigungen über identische Aktionsfolgen vergleichen, statt nur Endpunkte zu prüfen.

## Was zunächst nachrangig ist

- **GPU-Portierung:** Der Trainer ist CPU-only, und die Sammlung dominiert. Kleine, häufige Netzaufrufe mit vielen verschiedenen Gegnergewichten müssen zuerst sinnvoll gebündelt werden. Eine GPU beschleunigt weder die Python-Merkmalsableitung noch das Dateiprotokoll automatisch. Auch kostenloses PPO würde bei 14–15 % Zeitanteil allein nur rund 16–18 % mehr Gesamtdurchsatz ergeben.
- **Mehr Torch-Threads:** Bereits ein Thread; auf dem kleinen Netz konkurrieren zusätzliche Threads mit den Engines. Separat messen, nicht pauschal erhöhen.
- **Kleinerer Gegnerpool, weniger PPO-Epochen, kleineres Netz oder weniger Evaluation:** Das spart Arbeit, verändert aber Gegnerdiversität, Lernverhalten oder Messgenauigkeit. Erst später als kontrollierte Lernexperimente untersuchen.
- **Aus allen drei Sitzen PPO-Daten gewinnen:** Strategisch interessant, aber die Gegner verwenden eingefrorene andere Policies. Ihre Aktionen dürfen nicht als gewöhnliche On-Policy-Samples des Lerners behandelt werden. Eine Variante mit drei aktuellen Lerner-Policies würde das Trainingssetting ändern.
- **Flächendeckend Low-Level-Arrays oder neue Compilerflags:** Zuerst die gemessenen Python-Dopplungen, Bündelung und Austauschkosten angehen. Der verwendete Engine-Build ist schon optimiert.

## Empfohlene Reihenfolge und Erfolgskriterien

1. Längeren Konfigurationsvergleich mit 2/4/8 Umgebungen ausführen, pro Update dieselbe Samplezahl; 4 oder 8 anschließend auf identischem Holdout beurteilen.
2. Reinen Kritikerpfad für Bootstrap und gemeinsame Spielmerkmale implementieren, getrennt messen und auf Verhaltensgleichheit prüfen.
3. Auswertung parallelisieren; Cache und Poolindex mit einem großen, realistischen Snapshot-Pool untersuchen.
4. Die kleinen C++-Dopplungen entfernen und native Engine-Zeit gesondert erfassen.
5. Erst danach einen persistenten Pipe-/Socket-Prototyp gegen den Dateipfad messen und über direkte C++-Einbettung entscheiden.

Primäre Kennzahl: abgeschlossene Lernerübergänge pro realer Stunde einschließlich PPO und regulärer Evaluation. Zusätzlich Sammlung, PPO, Evaluation und Kaltstart getrennt ausweisen, abgeschlossene Spiele und Fehler zählen sowie CPU-Zeit und Speicher erfassen. Messungen mit Warm-up, alternierender Reihenfolge und mindestens drei Wiederholungen auf ansonsten ruhiger Maschine durchführen. Einen kleinen Anfangspool und einen realistischen späten Pool getrennt betrachten.

Bei rein technischen Optimierungen gleiche Seeds, Trajektorien und Endgewichte erwarten; bei geänderter Parallelitätskonfiguration stattdessen den gleichen Evaluationsplan und mehrere Trainingsseeds verwenden. Mehr simulierte Übergänge pro Stunde sind erst dann ein praktischer Trainingsgewinn, wenn die Spielstärke bei gleichem Zeitbudget mindestens erhalten bleibt.
