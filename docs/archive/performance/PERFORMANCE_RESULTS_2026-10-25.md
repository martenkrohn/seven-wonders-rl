# Runtime-Optimierungen und Vorher-Nachher-Messung, 25. Oktober 2026

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Die Maßnahmen mit geringem bis mittlerem Aufwand aus dem [Audit](PERFORMANCE_AUDIT_2026-10-25.md) wurden umgesetzt. Die Engine verwendet weiterhin das JSON-/Dateiprotokoll und einen Prozess pro Spiel. Direkte C++-Einbettung, Pipes und persistente Engine-Worker wurden nicht implementiert.

## Implementierung

- **Bootstrap:** `ActorCritic.forward_values()` berechnet nur den Kritiker. Der Collector stapelt dafür ausschließlich globale Beobachtungen und überspringt vollständig terminale Batches. Gemischte Batches behalten ihre ursprüngliche Größe und Rechenfolge, um Rundungsunterschiede durch eine geänderte Matrixgröße zu vermeiden. Terminale Werte bleiben null; externe Trunkierungen nutzen weiterhin den echten Zustand vor Reset. Der zusätzliche Umbau zur Wiederverwendung der Werte aus der nächsten Schleifeniteration war nicht nötig.
- **Gemeinsame Beobachtungen:** `ObservationContext` bereitet Karten-IDs, Wissenschaft/Joker, Rabatte, Produktion, Handelsangebot, Wunderstand und Tabellenmerkmale einmal pro eingelesenem Status vor. Lerner und Gegner erhalten daraus gecachte, relativ zu ihrem Sitz gedrehte Beobachtungen. Der Kontext wird weder in das Engine-JSON geschrieben noch auf spätere Statusobjekte übertragen. Zustandsmerkmale werden einmal vorbereitet und als Vektor in die legalen Aktionszeilen übernommen. Statische Kartenproduktion und Karteneffekte werden begrenzt gecacht. Schema, Typen, Aktionsmasken und Header-/Nullwerte bleiben erhalten.
- **Gegnerverwaltung:** Der Trainer registriert eigene atomar gespeicherte Snapshots im Poolindex. Andere Nutzer behalten automatische Verzeichniserkennung beziehungsweise können `refresh()` aufrufen. Der Modellcache umfasst standardmäßig mindestens 16 Modelle beziehungsweise den zulässigen Pool plus zwei aktive Sitze je Umgebung; `--opponent-cache-size` überschreibt die Grenze. Eigene `checkpoint_*.pt` enthalten keine Optimizerzustände mehr; `latest.pt` behält diese. Das implementiert keinen exakten Resume.
- **Parallele Evaluation:** Ein CPU-Prozesspool verteilt unabhängige Spiele. Jeder Worker verwendet einen Torch-Thread, einen eigenen Austauschpfad und einen über seinen Spielblock erhaltenen Gegnercache. Ergebnisse werden nach ursprünglicher Spiel-ID geschrieben; Seeds, Gegnerplan und Diagnoseaufzeichnungen bleiben gleich. Bereits abgeschlossene Versuche werden auch bei einem Fehler eines anderen Workers erhalten. Die Self-Play- und Green-Priority-CLIs akzeptieren `--workers` mit Standard 2; Training akzeptiert `--eval-workers` mit Standard 1, da Prozessstart bei kleinen Zwischenmessungen teuer sein kann.
- **C++:** `--quiet` überspringt die Debugblöcke einschließlich ihrer zusätzlichen Legalitätsprüfungen tatsächlich. `CalculateScores()` liefert die bisherigen sechs Kategorien und die Summe gemeinsam; die Statusausgabe berechnet diese Kategorien nicht mehr doppelt. Der vollständige Upstream-Patch und das Prüfsummen-Inventar wurden aktualisiert; siehe [ENG-008](../../SIMULATOR_CHANGES.md#eng-008--doppelte-arbeit-bei-status-und-unterdrückten-debugausgaben).
- **Konfiguration:** `--throughput-preset` wählt acht Umgebungen und Horizont 64, weiterhin 512 Samples pro PPO-Update. Die bisherigen Standardwerte 2 × 256 bleiben für bestehende Experimentreihen erhalten. Acht statt zwei Umgebungen ändern Trajektorien, Seed-/Spielverteilung und GAE-Horizont und sind deshalb eine ausdrücklich auswählbare Variante.

## Messverfahren

Vor den Änderungen wurde der komplette relevante lokale Quellstand einschließlich der vorher schon vorhandenen, nicht eingecheckten Arbeit und des benutzten Engine-Binarys unter `python/runs/performance-implementation-20261025/before-source/` eingefroren. Erste Vorabmessungen liefen während der Umsetzung und sind **nicht** die Grundlage des abschließenden Geschwindigkeitsvergleichs.

Der abschließende Benchmark verwendet getrennte Prozesse und Austauschverzeichnisse, RAM-Austausch, dieselbe virtuelle Python-Umgebung und dieselbe CPU. Builds und Tests liefen dabei nicht gleichzeitig. Gemessene Varianten werden je Wiederholung in umgekehrter Reihenfolge ausgeführt. Es gibt drei Wiederholungen und einen separaten Trainings-Warm-up je Quellstand. Ausgewiesen werden Median-Durchsatz, interne Trainerzeit und externe Prozesslaufzeit; Kaltstart und Python-Import sind in der externen Zeit enthalten.

Training: 4.096 Übergänge je Lauf; 512 Übergänge pro PPO-Update; Batchgröße 256; vier Minibatch-Epochen; Snapshots alle 512 Übergänge; keine Zwischenevaluation, eine abschließende Evaluation mit zwölf Spielen. Je Quellstand werden zwei und acht Umgebungen bei gleichen Seeds untersucht. Ein Vergleich unterschiedlicher Umgebungszahlen ist keine Verhaltensgleichheitsprüfung.

Zusätzliche Messungen: Aufbau von jeweils drei Beobachtungen für 512 gleichmäßig über den vorhandenen Fixture-Pool ausgewählte Zustände; Transport mit 20 vollständig simulierten Spielen und vollständigen Status-Fingerprints; detaillierte neuronale Evaluation mit 96 Spielen je Versuch auf identischem Seed-/Sitz-/Gegnerplan. Für letztere werden dieselben bereits vorhandenen Checkpointdateien für beide Quellstände verwendet. Prozessstartkosten der parallelen Evaluation sind eingeschlossen.

Die automatisierte Prüfung vergleicht bei gleichem Trainingssetting die Trajektorien-Hashes jeder PPO-Sammlung und die finalen Modellparameter. Beobachtungen, Engine-Statusfolgen und detaillierte Auswertungssätze werden zusätzlich über vollständige Inhalts-Fingerprints verglichen. Nur die Laufzeitfelder der Auswertung werden vor dem Vergleich entfernt.

Die einzelnen Beiträge der technischen Änderungen wurden nicht separat isoliert; gemessen wird das Paket. Die Trainingsläufe haben nur neun Pool-Snapshots. Die Wirkung des größeren Modellcaches auf einen großen späten Pool wird deshalb nicht als gemessener Anteil am Trainingsgewinn ausgewiesen. Ein gesonderter Regressionstest prüft die Wiederverwendung von mehr als sechzehn geladenen Snapshots.

## Ergebnisse

Alle 33 gemessenen Versuche wurden erfolgreich abgeschlossen. Sämtliche Verhaltensvergleiche des Benchmarks stimmen überein: finale Gewichte und Rollout-Hashes in allen sechs passenden Trainingspaaren; Beobachtungsdaten; vollständige Engine-Statusfolgen; detaillierte Auswertungssätze einschließlich Aktionen, Kategorien und Seed-/Sitzdaten. Die zusätzlichen Auswertungsläufe umfassen 864 vollständig abgeschlossene Spiele ohne Trunkierungen oder Fehler.

| Trainingssetting | Vorher, Median | Nachher, Median | Verhältnis der Median-Durchsätze |
| --- | ---: | ---: | ---: |
| 2 Umgebungen × Horizont 256 | 130,5 Übergänge/s | 147,8 Übergänge/s | **+13,3 %** |
| 8 Umgebungen × Horizont 64 | 161,2 Übergänge/s | 186,9 Übergänge/s | **+16,0 %** |
| Bisheriges 2er-Setting → optimierte 8er-Variante | 130,5 Übergänge/s | 186,9 Übergänge/s | **+43,3 %** |

Der interne Trainingsabschnitt enthält Sammlung, PPO, Checkpoints und die abschließende Evaluation. Die externen Prozesszeiten inklusive Import/Initialisierung betragen im Median 34,87 s vorher und 31,19 s nachher mit zwei Umgebungen beziehungsweise 25,50 s nachher mit acht Umgebungen. Das entspricht bei diesen kurzen Einmalläufen rund 11 % beziehungsweise 27 % weniger realer Gesamtzeit. Die relativen Durchsatzgewinne innerhalb des Trainingsabschnitts dürfen deshalb nicht unverändert auf kurze CLI-Einmalläufe übertragen werden.

| Zusätzliche Messung | Vorher | Nachher | Einordnung |
| --- | ---: | ---: | --- |
| Gemeinsamer Merkmalsaufbau | 1.258 Beobachtungen/s | 1.851 Beobachtungen/s | **+47 % Durchsatz** |
| Detaillierte Evaluation, seriell, 96 Spiele | 15,27 s | 13,49 s | +13 % Spieldurchsatz |
| Detaillierte Evaluation, nachher mit 2 Workern, 96 Spiele | 15,27 s | 10,22 s | **33 % weniger Zeit**, +49 % Spieldurchsatz |
| Transport und Beobachtungsbau ohne neuronale Policies | 529 Übergänge/s | 517 Übergänge/s | Kein belastbarer Gewinn; Median etwa 2 % niedriger |

Die parallele Evaluationszeit enthält den Start und das Ende des Prozesspools. Vollständige externe CLI-Prozesszeiten der Evaluationsmessung liegen im Median bei 18,08 s vorher und 13,29 s nachher mit zwei Workern. Kleine Pläne können wegen des Prozessstarts dennoch seriell schneller sein.

Beim reinen Transport reicht die beobachtete Streuung nicht für eine belastbare Beschleunigungsbehauptung: In den drei Paaren ändert sich der Durchsatz ungefähr um −2,3 %, −0,1 % und +4,8 %. Ein Gewinn der isolierten C++-Änderungen wurde nicht separat ermittelt. Der weiterhin vorhandene Dateiaustausch und das Polling bleiben bestehen.

Ein repräsentativer Snapshot aus derselben Updateposition schrumpft von 1.329.189 auf 446.181 Bytes, weil der nur für Inferenz verwendete Snapshot keinen Optimizerzustand mehr enthält. Daraus folgt keine zusätzlich isoliert gemessene Trainingsbeschleunigung.

Die [kompakte Messaufzeichnung](PERFORMANCE_RESULTS_2026-10-25.json) enthält alle Einzelversuche, Versionsdaten, Hashes und Quell-/Engine-Prüfsummen. Drei kurze Wiederholungen auf dieser Maschine sind kein universelles Leistungsversprechen. Für eine Lernqualitätsaussage mit acht Umgebungen ist weiterhin ein längerer gemeinsamer Holdout-Vergleich nötig; die rein technischen Änderungen wurden dagegen bei unveränderter Konfiguration auf identische Trainingsresultate geprüft.

## Wiederholung und Verwendung

Aus `python/`, jeweils mit frischem Ausgabeverzeichnis:

```sh
.venv/bin/python scripts/benchmark_optimizations.py \
  --before-root runs/performance-implementation-20261025/before-source \
  --out-dir runs/performance-repeat \
  --checkpoint runs/performance-implementation-20261025/before-n2-r0/latest.pt \
  --references runs/performance-implementation-20261025/before-n2-r0/checkpoint_1024.pt \
               runs/performance-implementation-20261025/before-n2-r0/checkpoint_2048.pt \
  --repeats 3 --steps 4096 --blocks 8
```

Der Benchmark beendet sich mit Fehler, wenn die Verhaltens-Fingerprints zwischen vorher und nachher bei gleichem Setting abweichen. Rohdaten, Prozesslogs, Modelle und die eingefrorene Referenz liegen lokal unter `python/runs/performance-implementation-20261025/`; diese großen Artefakte werden nicht eingecheckt. Die kompakte Ergebnisaufzeichnung wird neben diesem Bericht gespeichert.

Für ein neues Training mit höherer Parallelität:

```sh
.venv/bin/python scripts/ppo_training/train_unified.py \
  --mode selfplay --throughput-preset --out-dir runs/new-throughput-run
```

Bei langen Evaluationen sind zwei Worker der neue CLI-Standard; für kleine Evaluationspläne lässt sich mit `--workers 1` seriell arbeiten. Eine gesteigerte Anzahl Übergänge pro Stunde ist kein zusätzlicher Nachweis höherer Spielstärke. Die Konfigurationsvariante mit acht Umgebungen sollte auf demselben Holdout-Plan wie bisherige Trainings verglichen werden.

## Validierung

Die bestehende Python-Testsuite bestand nach dem Kernumbau. Abschließend wurden
17 gezielte Tests für Runtime-Optimierungen, Collector, parallele Auswertung,
Green-Priority-CLI und Self-Play-Snapshots erfolgreich ausgeführt. Sie prüfen auch
den Fehlerfall eines Workers, RNG-/Gewichtserhaltung und Optimizerzustände in
regulären Checkpoints gegenüber Inferenz-Snapshots. Alle vier CTests und die
Prüfung des regenerierten Inventars mit 49 C++-/Build-/Testdateien bestanden.
`git diff --check` meldet keine Whitespace-Fehler.
