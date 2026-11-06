> Historical audit before the cleanup. Findings describe the pre-cleanup tree.
> [Implemented changes](../reference/CLEANUP.md).

# Projekt-Audit: Aufräumen, GitHub und Präsentation

Stand: 6. November 2026. Untersucht wurden der lokale Arbeitsbaum, Git-Index und
Commitverlauf, direkte Imports des aktiven RL-Pfads, Tests und CI, Dokumentation,
öffentliche GitHub-Refs und Repository-Metadaten sowie die veröffentlichte
Demo-Snapshotbeschreibung. Dies ist ein Audit mit Handlungsempfehlungen; Code,
Trainingsläufe und öffentliche Inhalte wurden dafür nicht verändert.

**1. Gesamturteil**

Eine deutliche Bereinigung ist sinnvoll. Die größte Altlast ist das gleichzeitige
Vorhalten zweier Trainingssysteme: Protokoll 2 mit altem Simulator und Forschungs-
werkzeugen sowie Protokoll 3 mit Second-Edition-Engine und neuem Agenten.
Hinzu kommen getrennte serielle und parallele Trainingsabläufe innerhalb des
neuen Systems. Dokumentation und Prüfabläufe behandeln diese historischen Wege
weiterhin als zu pflegende Software.

Empfohlenes Ziel: **ein aktiv unterstützter Second-Edition-Pfad mit Gedächtnis,
einer Trainingsimplementierung und einer Evaluation.** Historische Ergebnisse
bleiben zitierbar; alte Trainer müssen deshalb nicht weiter ausführbar bleiben.
Die bestehende Git-Historie kann alte Quellstände bewahren. Dafür ist kein
zusätzliches Archiv sämtlicher alter Implementierungen im aktuellen Baum nötig.

Eine Ausnahme ist die Dashboard-Pipeline: Die öffentliche Demo und der lokale
Renderer sind weiterhin nützlich. Deren Ablösung oder Anpassung muss vor dem
Entfernen der dafür benötigten alten Berichtskomponenten erfolgen.

**2. Tatsächlicher Stand: lokal und öffentlich**

| Ebene | Beobachtung | Konsequenz |
| --- | --- | --- |
| Öffentlicher Defaultbranch `main` | `1c9ea06`, Commit vom 24. Oktober 2026 | Die Startseite erklärt noch die alte 28-Aktionen-Umgebung und das alte Setup. |
| Öffentlicher Entwicklungsbranch | `refactor/separate-engine-20261105`, `ab38603` | Engine-Auslagerung, Second Edition und erste Protokoll-3-Anbindung sind bereits öffentlich. |
| Abstand zu `main` | Sieben zusätzliche Commits, kein abweichender Commit auf `main` im geprüften Vergleich | Die Veröffentlichungslücke besteht bereits unabhängig von den uncommitteten Änderungen. |
| Lokaler Arbeitsbaum | Bei Bestandsaufnahme 16 geänderte und 34 unversionierte Dateien | Wesentliche neue Funktionen sind noch nicht Bestandteil eines veröffentlichten Commits. Zahlen vor Anlage dieses Audits. |
| Öffentlicher RL-Tag | `v0.2.0-bga` verweist auf `86d4866` | Der Tag enthält nicht automatisch den späteren lokalen Stand. |
| Neuester CI-Lauf des Entwicklungsbranches | Erfolgreich für `ab38603` | Das validiert diesen Commit, nicht die danach entstandenen lokalen Funktionen. |
| Öffentliche Demo | Seed 31, Feature-Semantik v3, Export vom 24. Oktober | Kein öffentlicher Nachweis der aktuellen Second-Edition-Lernleistung. |

Direkt verifiziert über `git ls-remote --heads --tags origin`, GitHub-API und
die Live-Datei `snapshot.json`. Quellen:
[Repository](https://github.com/martenkrohn/seven-wonders-rl),
[öffentlicher Main-Stand](https://github.com/martenkrohn/seven-wonders-rl/tree/1c9ea06a06513f1f7b1ff46385d49d8bc996d944),
[Entwicklungsstand](https://github.com/martenkrohn/seven-wonders-rl/tree/ab3860311cbd03f22a63ec242929e10f7d626e11),
[erfolgreicher CI-Lauf](https://github.com/martenkrohn/seven-wonders-rl/actions/runs/42188248577),
[Live-Snapshot](https://martenkrohn.github.io/seven-wonders-rl/snapshot.json).

Wichtig für die Veröffentlichung: Dateien können schon über einen öffentlichen
Branch erreichbar sein, obwohl sie auf der GitHub-Startseite unter `main` fehlen.
Ein sauberer Defaultbranch ist daher etwas anderes als ein vollständig
bereinigtes öffentliches Repository einschließlich Historie und anderer Refs.

**3. Code: Was bleiben sollte und was entfallen kann**

Der aktive Pfad ist relativ klar abgrenzbar. Eine statische Verfolgung direkter
Python-Imports ab `DecisionEnv`, Decision-PPO und Decision-Evaluation erreicht
13 der 62 aktuell vorhandenen Python-Module unter `python/src/`. Das ist ein
Orientierungswert für den Kern, keine Aussage, dass alle anderen Module tot
sind: Renderer, Diagnosewerkzeuge und weitere öffentliche APIs zählen dabei
nicht als Einstiegspunkte.

| Bestandteil | Empfehlung | Begründung und Bedingung |
| --- | --- | --- |
| `envs/decision_env.py`, `features/decision_features.py`, `features/decision_memory.py` | Behalten als alleinigen aktiven Umgebungs-/Beobachtungspfad | Implementieren den aktuellen Vertrag mit Gedächtnis. |
| `utils/decision_ppo.py`, `utils/decision_parallel.py` | Zusammenführen | Enthalten getrennte Trainingsschleifen, Resume-/Liga-Vorbereitung, Metriken und Speichern. |
| `utils/decision_workers.py` | Behalten und in gemeinsamen Trainer einbinden | Prozessbesitz, Shared Memory und Fehlerbehandlung sind weiterhin erforderlich. |
| `selfplay/decision_league.py` | Behalten | Aktive Gegnerliga, Snapshot-Verwaltung und Wiederaufnahme. |
| `selfplay/pair_pfsp.py`, `selfplay/opponent_pool.py` | Behalten, erst danach gezielt vereinfachen | Beide werden vom neuen Ligapfad direkt benutzt; sie sind trotz alter Herkunft keine toten Dateien. |
| `evaluation/decision_evaluation.py`, `decision_intermediate.py` | Behalten | Separate feste Vergleichspläne und laufende Trainingskontrolle erfüllen unterschiedliche Zwecke. |
| `utils/decision_control.py`, `decision_console.py` | Behalten | Aktiver Stopp-/Resume-Betrieb und Terminalausgabe. |
| `scripts/train_decision_ppo.py`, `scripts/eval_decision.py` | Zu den einzigen normalen Training-/Evaluation-Einstiegen machen | Aktuelle Bedienung wird dadurch eindeutig. |
| `scripts/train_ppo.py`, `scripts/train_ppovppo.py` | Entfernen, nachdem verbliebene Imports geprüft sind | Ausführung ist bereits ausdrücklich deaktiviert; der alte Code enthält verworfene GAE-/Evaluationspfade. Git bewahrt ihn. |
| `scripts/ppo_training/run_long_training*.py`, `run_multihead_experiment.py`, `run_fresh_pair_pfsp.py`, Spezialisten-/Prior-Rezepte | Aus dem aktiven Baum entfernen, sobald die Entscheidung zur Aufgabe des Legacy-Betriebs umgesetzt wird | Historische Versuchssteuerung ist kein notwendiger Teil des aktuellen Trainers. |
| `scripts/ppo_training/train_unified.py`, `envs/sevenwonders_env.py`, alte Feature-/Modell-/Policy-Pfade | Legacy-Laufzeit als zusammenhängenden Bestand stilllegen | Nicht einzeln blind löschen: alte Evaluation, Diagnose und Dashboard-Zulieferung hängen teilweise daran. |
| Alte `configs/long_*`, `multihead_*`, `fresh_pair_pfsp_*` | Mit den zugehörigen Legacy-Rezepten entfernen oder nur ausgewählte Versuchsparameter im Ergebnisarchiv erhalten | Keine parallel empfohlenen Startkonfigurationen pflegen. |
| `evaluation/dashboard.py`, `evaluation/assets/`, `scripts/build_dashboard.py`, `demo/` | Vorerst behalten | Sichtbarer und nützlicher Projektbestandteil; aktive Evaluation liefert noch nicht dessen alte Berichtsschnittstelle. |
| `data/bga_cards-v1.json`, `bga_wonders-v1.json`, `scripts/export_bga_data.py` | Behalten | Aktuelle Datengrundlage und reproduzierbarer Export. |
| Alte `card_db.json`, `wonder_db.json`, alte Exporte und Fixture-Pools | Zusammen mit Legacy-Verbrauchern prüfen und reduzieren | Kein pauschaler Löschkandidat, solange alte Berichte/Werkzeuge noch laufen sollen. |
| `scripts/benchmark_decision_performance.py` | Auf kompakten Benchmark des aktuellen Trainers reduzieren | Enthält auf 749 Zeilen auch isolierte Vergleichsprototypen; verworfene Implementierungen gehören nicht dauerhaft zum normalen Werkzeugbestand. |
| `scripts/edition_comparison/report.html` und große Editionsvergleichsdaten | Aus aktivem Werkzeugbestand herausnehmen oder als abgeschlossenen Migrationsnachweis bündeln | Die Migration ist umgesetzt; ein generierter Bericht unter `scripts/` verwischt Quelle und Ergebnis. |

Das ist eine Entfernungsempfehlung für zusammengehörige Funktionsbereiche,
keine bereits vollständig geprüfte Liste einzeln gefahrlos löschbarer Dateien.
Beim Umsetzen muss der Import- und Dokumentationsgraph des gewählten Zielbestands
mitgeführt werden. Eine bloße Umbenennung aller alten Dateien in `legacy/`
würde die Test- und Wartungslast nicht ausreichend beseitigen.

**4. Gedächtnis: alten Modus wirklich weiterpflegen?**

Für das gewünschte Produktziel gibt es keinen erkennbaren Grund, den Agenten ohne
Gedächtnis als gleichberechtigten Trainingsmodus dauerhaft zu unterstützen.
Der aktive Protokoll-3-Agent hat bereits einen verbindlichen Gedächtnisvertrag.
Die umfangreiche optionale Gedächtniserweiterung des alten Trainers ist ein
anderer Implementierungspfad, keine notwendige zweite Variante des neuen Agenten.

Mehr verfügbare legitime Information erweitert grundsätzlich die Möglichkeiten
einer Policy. Bei endlicher Modellgröße und Trainingszeit garantiert sie aber
keinen empirischen Spielstärkegewinn. Diese Unterscheidung rechtfertigt eine
saubere Aussage in Ergebnissen; sie verpflichtet nicht dazu, die alte Software
weiter zu betreiben. Ein Vergleich ohne Gedächtnis wäre nur dann zusätzliche
Arbeit wert, wenn der isolierte Nutzen des Gedächtnisses eine konkrete
Forschungsfrage werden soll. Für bloßes Aufräumen ist so ein Experiment unnötig.

Alte Checkpoints müssen anschließend nicht mehr ladbar sein. Eine klare
Ablehnung inkompatibler Modelle sollte bleiben, damit niemand versehentlich
Gewichte mit falschen Regeln oder Beobachtungen verwendet.

**5. Seriell und parallel: tatsächliche Doppelarbeit reduzieren**

`decision_ppo.train()` verzweigt bei `workers > 1` zu `train_parallel()`;
sonst läuft `_train()`. Beide Wege pflegen wesentliche Teile des Trainingsablaufs
selbst. Das ist ein konkretes Beispiel für vermeidbaren Wartungsaufwand.

Empfehlung: eine Trainingsschleife und einen Collector verwenden. Die normale
Bedienung kann vier Worker vorauswählen. Ein Worker kann eine Einstellung
desselben Collectors bleiben, ohne eine zweite Trainerimplementierung zu
rechtfertigen. Ob die aktuelle Callback-API bei einem Worker erhalten bleiben
soll, muss beim Zusammenführen explizit entschieden werden: Der parallele Worker
übernimmt gegenwärtig nicht automatisch beliebige externe Python-Callbacks.

Paralleles Training ist gemessen schneller, aber keine identische Lernprozedur:
Bei 64 gesammelten Entscheidungen je Update verteilen vier Worker diese auf
mehrere Spiele. Damit ändern sich die pro Spiel zusammenhängenden Abschnitte
für GAE und Bootstrapping. Ein echter Lernqualitätsvergleich wäre eine gezielte
Methodenfrage; er gehört nicht zur Validierung einer englischen Textausgabe.

Für Evaluation ist die Lage einfacher: Ein fester Plan lässt sich auf mehrere
Prozesse verteilen. Dass Seeds, Sitze, Gegner und Ergebnisse dabei erhalten
bleiben, prüft man bei Änderungen an der Parallelisierung gezielt. Den Vergleich
muss man bei späteren reinen Textänderungen nicht wiederholen.

Die vorhandenen Performanceberichte belegen umfangreiche damalige Messungen,
aber keinen Bedarf, diese Messkampagne bei jeder späteren Änderung neu zu starten.

**6. Tests: konkrete Ursachen und angemessener Umfang**

Der lokale Baum enthält bei der Bestandsaufnahme 52 `test_*.py`-Dateien mit
231 Testfunktionen; davon neun `test_decision*`-Dateien mit 47 Testfunktionen.
Parametrisierungen sind darin nicht als zusätzliche Fälle gezählt. Viele Tests
betreffen noch das historische System.

Zwei konkrete Strukturprobleme:

- `python/tests/conftest.py::legacy_runtime` ist eine automatische Session-
  Fixture. Sobald die ausgewählten Tests bestimmte Legacy-Bedingungen erfüllen,
  klont sie die lokale Engine, checkt `f39c1d7...` aus und baut eine historische
  Engine mit CMake. Das ist echter zusätzlicher Aufwand. Es geschieht nicht
  zwangsläufig bei einem einzeln ausgewählten Console-Test.
- Keine der neun `test_decision*`-Dateien hat die registrierten Markierungen
  `unit`, `component`, `integration` oder `engine`. Die dokumentierten
  Marker-Auswahlen erfassen damit die neuen Tests nicht. Auch `make test-all`
  ist nur eine Vereinigung dieser Marker und damit derzeit kein vollständiger
  Testlauf. Das unverfilterte `pytest` in CI erfasst sie dagegen.

Zusätzlich führt CI immer das gesamte Python-Testset, Engine-Prüfungen, CTest,
einen zusätzlichen aktiven PPO-Smoke und mehrere Legacy-CLI-Aufrufe aus.
Für eine vollständige Integrationsprüfung ist das nachvollziehbar. Als lokale
Standardantwort auf jede kleine Änderung ist es unangemessen breit.

| Änderung | Sinnvolle lokale Prüfung | Normalerweise nicht nötig |
| --- | --- | --- |
| Deutsche Terminaltexte durch englische ersetzen | Betroffene Formatierungs-/Console-Tests; bei Bedarf kurzer Aufruf des betroffenen Ausgabewegs | Neue Trainingskampagne, seriell-parallel-Vergleich, Engine-Neubau, historische Suite |
| README oder Navigation | Relative Links und Befehle gegen Zielbestand prüfen | PPO-Training und Simulatorregressionen |
| Einzelne Umgebungs-/Feature-Semantik | Zugehörige Vertrags-/Featuretests, bei Enginekontakt kleine Integration | Alle alten Versuchsrezepte und Benchmarks |
| Collector, Parallelisierung oder GAE | Trajektoriengrenzen, Budget, Terminal-Bootstrap, Workerfehler und kurzer echter Trainingslauf | Unabhängige Langläufe ohne konkrete Lernqualitätsfrage |
| Speichern, Resume oder Liga-Journale | Gezielte Wiederaufnahme-/Abbruchtests und Snapshot-Konsistenz | Historische Modellfamilien, wenn deren Unterstützung entfällt |
| Abschluss einer größeren Bereinigung | Ein vollständiger Testlauf des verbleibenden aktiven Systems und frischer Setup-Smoke | Weiterhin jeden abgeschafften Legacy-Weg lauffähig halten |

Kleine Formatierungstests wie `test_decision_console.py` sind bereits vorhanden.
Die Prozess-/Crash-Tests in `test_decision_parallel.py` prüfen hingegen echten
Betriebsschutz. Diese Tests sind bei Worker-/Resume-Änderungen sinnvoll und
sollten nicht pauschal gelöscht werden.

Empfehlung für die Umsetzung: aktive Tests sauber markieren, Legacy-Tests aus
dem normalen Prüfumfang entfernen und `test-all` tatsächlich vollständig machen.
Ein Test für eine verworfene Funktion wird zusammen mit der Funktion entfernt.
Ein Test für GAE, fehlende private Informationen, legale Aktionen oder sichere
Wiederaufnahme bleibt, auch wenn er historisch entstanden ist.

Die konkrete Laufzeit und Befehlsfolge des parallel arbeitenden Übersetzungs-
agenten wurden hier nicht beobachtet. Deshalb lässt sich dessen Verzögerung
nicht sicher einer bestimmten Prüfung zuschreiben. Die strukturellen Ursachen
für unnötig breite Prüfungen sind unabhängig davon nachweisbar.

**7. Dokumentation: höchste Aufräumpriorität**

Git führt aktuell 76 Dateien unter `docs/`, davon 42 im Archiv. Das Archiv ist
bereits von aktuellen Anleitungen getrennt; das Problem ist die zusätzliche
Mischung historischer und aktueller Aussagen außerhalb des Archivs.

| Datei/Bereich | Konkreter Befund | Änderungsempfehlung |
| --- | --- | --- |
| `README.md` lokal | Mischt aktuellen Agenten, Legacy-Forschung, alten Trainingsbetrieb, GUI und lange Bedienhinweise; erster Link ist deutsch und lokalbetrieblich | Als englischen Projekteinstieg neu ordnen, Bedienung auslagern. |
| `docs/README.md` | Mehrere zeitlich verschiedene „Stand“-Abschnitte und Legacy-Anweisungen; Handover wird mit „Gedächtnislücke“ beschrieben | Einen aktuellen Einstieg und klar getrennten Ergebnisnachweis anbieten. |
| `docs/reference/ARCHITECTURE.md` | Architekturhauptreferenz beginnt weiterhin mit Protokoll 2 | Durch Architektur des aktiven Pfads ersetzen; alte Fassung nur historisch verlinken. |
| `docs/guides/DEVELOPMENT.md` | Nennt sich aktuelle Arbeitsanleitung, führt aber primär den alten Unified Trainer aus | Auf aktuelle Installation, CLI und angemessene Prüfungen umstellen. |
| `python/scripts/README.md` | Beschreibt den aktiven Einstieg oben, führt danach den alten Trainer als „Main PPO training“ | Nur wenige unterstützte Tools prominent führen; alte Werkzeugliste entfernen. |
| `docs/research/RL_RESTART_HANDOVER.md` | Nachtrag sagt noch, Evaluation bleibe ein weiterer Baustein, obwohl neue Evaluation implementiert ist | Als abgeschlossenen Migrationsaudit archivieren; nicht weiter als aktuelle Aufgabenliste ergänzen. |
| `docs/research/PERFORMANCE_RESULTS_2026-11-06.md` | Nachtrag meldet Umsetzung, Haupttext beschreibt Prototypen und einen gestoppten Lauf | Als abgeschlossenen Messbericht erhalten; aktuelle Betriebsinformation ausschließlich im Guide. |
| `OBSERVATION_MEMORY.md`, `PAIR_PFSP_TRAINING.md`, `CONTINUOUS_TRAINING.md`, Teile von `EVALUATION.md` | Alte Pfade bleiben unter normalen Guide-Namen, teilweise mit Legacy-Hinweis | Historische Bedienung entfernen oder archivieren; nur nötige Dashboard-Anleitung aktuell behalten. |
| `research/OVERVIEW.md`, `AGENDA.md`, `CONTENT_BANK.md` | Methodische Inhalte bleiben nützlich, Implementierungsstand teilweise Legacy | Forschung und Quellen behalten; Aussagen zur aktuellen Implementierung neu prüfen. Keine neue Literaturrecherche zum Aufräumen nötig. |
| `docs/experiments/001-fresh-pair-pfsp/PROTOCOL.md` | Vorbereiteter Versuch des alten Pfads | Als nicht durchgeführten historischen Plan kennzeichnen oder entfernen, wenn ohne weiteren Nutzen. |
| `docs/reference/*validation*.json`, `restart-step1-20261106.json` | Viele einzelne Umsetzungs-/Abnahmeprotokolle | Ein kompakter Release-Nachweis mit relevanten Versionen, Prüfungen und Grenzen; nur einzigartige Evidenz separat erhalten. |
| Editionsaudit und Editionsvergleich | Abgeschlossener Migrationsbedarf, keine aktuelle Aufgabenliste | In einen abgeschlossenen Migrationsnachweis überführen und aus dem normalen Einstieg entfernen. |

Eine Linkprüfung relativer Markdown-Dateiziele außerhalb von Codeblöcken fand
drei fehlende Ziele: `docs/ORIGIN.md` verlinkt `simulator/upstream.patch` und
`simulator/manifest.json`; `CONTENT_BANK.md` verlinkt `../../src/player.cpp`.
Die entsprechenden Engine-Dateien liegen nach der Auslagerung unter `engine/`.
Die Prüfung deckt keine externen URLs oder Abschnittsanker ab.

`ORIGIN.md`, `NOTICE`, Lizenztexte und Herkunftsnachweise sind kein beliebiger
Clutter. Sie sollten kompakt und konsistent erhalten bleiben. Aussagen, wonach
GUI-Artwork nicht verteilt werde, müssen den indirekt eingebundenen Engine-
Bestand korrekt abgrenzen; die lokale README beschreibt inzwischen eine GUI
mit im Engine-Repository enthaltenem Artwork.

Historische Messergebnisse sind ebenfalls anders zu behandeln als alte
Startanleitungen: Sie dokumentieren Arbeit und Grenzen. Ausgewählte aussagekräftige
Berichte behalten, redundante Zwischenprotokolle reduzieren. Eine zusätzliche
Git-Datei pro kleiner Umsetzung ist keine notwendige Nachweispflicht.

**8. Was auf GitHub landet und was lokal bleibt**

Der geprüfte Index enthält 395 Einträge. Die vorhandenen regulären Dateien
umfassen zusammen etwa 5,5 MB; die Engine ist zusätzlich ein Git-Submodule mit
eigenem Repository. Große lokale Verzeichnisse dürfen nicht mit dem Umfang des
öffentlichen aktuellen Quellbaums verwechselt werden.

| Bestand | Im aktuellen Git-Bestand? | Einordnung |
| --- | --- | --- |
| Python-Quellcode, Tests, Fixture-Pool, Konfigurationen | Ja, soweit bereits getrackt | Wird mit entsprechenden Commits veröffentlicht. |
| Dokumentation, ausgewählte Ergebnis-JSONs und Vergleichsberichte | Ja | Ein Teil der vermeintlichen Ausgabedaten ist ausdrücklich öffentlich versioniert. |
| `demo/index.html`, `demo/snapshot.json`, `.nojekyll` | Ja | Bewusst ausgewählte öffentliche Demonstration. |
| `python/runs/`, `.venv`, Builds, Checkpoints `*.pt` | Ignoriert; im geprüften Index keine `.pt`-Dateien oder Python-Caches | Der große lokale Trainingsbestand gehört nicht zum aktuellen versionierten Quellbaum. |
| Neue Decision-Module und zugehörige Tests/Guides | Teilweise noch unversioniert | Noch nicht Teil eines Commits; `.gitignore` schützt sie nicht vor einem späteren `git add`. |
| `TRAINING.md` | Neu, unversioniert und nicht ignoriert | Würde beim Hinzufügen veröffentlicht; enthält derzeit absoluten lokalen Nutzerpfad und konkreten Lauf. |
| `engine/` | Öffentlicher Submodule-Verweis | GitHub zeigt das eigene Engine-Repository und den gepinnten Commit. |

Der Pages-Workflow lädt ausdrücklich nur drei Demo-Dateien hoch. Er publiziert
keinen kompletten Laufordner. Repository-Veröffentlichung und Pages-Deployment
sind zwei verschiedene Wege: Die erste umfasst alle gepushten getrackten Dateien,
die zweite den ausgewählten statischen Snapshot.

Eine Suche in getrackten Textdateien fand `/home/` in zwölf Dateien, überwiegend
historischen Berichten, und lokale Referenznamen wie `00_marten` in sieben Dateien.
Das sind konkrete Aufräum-/Portabilitätsbefunde, kein Nachweis veröffentlichter
Zugangsdaten. Neue Validierungsdateien sollten ebenfalls vor Aufnahme auf lokale
Pfade geprüft werden. Ein vollständiger Secret- oder Historienaudit wurde nicht
durchgeführt; `.gitignore` beweist keine Geheimnisfreiheit.

Empfohlener Veröffentlichungsschritt: Dateien gezielt auswählen, dann den
vorgesehenen Commit mit `git diff --cached --name-status` und
`git diff --cached` ansehen. Für den gesamten versionierten Zielbestand zusätzlich
`git ls-files` bzw. den Commitbaum prüfen; ein Commit-Diff zeigt nur die Änderungen.
Keine pauschale Aufnahme aller neuen JSONs und Arbeitsnotizen.

Das Entfernen einer Datei aus dem aktuellen Baum entfernt frühere öffentliche
Versionen nicht. Für das gewünschte saubere Projektbild genügt gewöhnlich der
bereinigte aktuelle Stand; eine Umschreibung der Git-Historie ist dafür nicht
erforderlich.

**9. Lokalen Speicher getrennt vom Projektbild behandeln**

Gemessene lokale Größen: `python/runs/` etwa 7,3 GB, `.venv/` etwa 7,0 GB,
Root-Build etwa 22 MB, `python/build/` etwa 784 KB. Das sind Momentaufnahmen.
Die lokale `.git`-Datenbank ist etwa 349 MB groß; daraus folgt nicht, dass dieselbe
Objektmenge öffentlich gepusht wurde.

Build-/Wheel-Ausgaben und Caches lassen sich bei Bedarf neu erzeugen.
Trainingsverzeichnisse dagegen enthalten nicht ersetzbare Checkpoints und
Auswertungen. Vor deren Reduktion eine Übersicht pro Lauf erstellen und
mindestens den aktiven Lauf, benötigte eingefrorene Ligagegner, Referenzpanels und
ausgewählte Resultate erhalten. Keine Löschung von Snapshots allein aufgrund
ihres Alters: Pool, Resume und Evaluation können sie referenzieren.

Der aktive Lauf meldete bei einer lesenden Stichprobe `running`, vier Worker,
305.792 Entscheidungen und 16.699 abgeschlossene Partien. Das widerlegt die
Verwendung älterer Aussagen „weiterhin gestoppt“ als aktuelle Betriebsinformation.
Diese Werte sind keine neue Leistungsauswertung und ändern sich während des
Betriebs. Für diesen Audit wurde kein Training angehalten oder gestartet.

**10. README und Arbeitgeber-Präsentation**

Der technische Bestand erfüllt wesentliche Voraussetzungen für ein interessantes
Portfolio: nachvollziehbare eigene Erweiterung eines bestehenden Simulators,
Umgang mit partieller Information und legalen Aktionen, PPO, eingefrorene
Gegnerpopulationen, reproduzierbare Auswertung, Fehler-/Resume-Behandlung und ein
sichtbares Analysewerkzeug. Herkunft und Einsatz von Coding-Assistenten sind
transparent dokumentiert.

Der öffentliche Einstieg vermittelt davon zurzeit nur den älteren Stand.
Die lokale README enthält mehr aktuelle Technik, ist aber überladen mit
Migrationsgeschichte und historischen Bedienwegen. Interessierte müssen zu lange
suchen, um Problem, Eigenanteil, Ergebnis und aktuelle Entwicklungsfrage zu erkennen.

Aus öffentlichen Repository-Metadaten: Beschreibung vorhanden, aber kein
Homepage-Eintrag und keine Topics. Die bereits vorhandene Demo ist damit weniger
gut auffindbar, als sie sein könnte. Empfehlung: Demo als Website eintragen und
wenige passende Topics wie `reinforcement-learning`, `pytorch`, `ppo` und
`self-play` setzen. Das wurde hier nicht verändert.

Empfohlene Reihenfolge für eine spätere englische README:

1. Zwei bis drei Sätze zu Problem, Ziel und technischem Eigenanteil.
2. Screenshot oder aussagekräftige Grafik mit direktem Demo-Link und klarer
   Kennzeichnung des gezeigten Experimentstands.
3. Aktuelles System: Second Edition, beobachtete Historie für Actor/Critic,
   legale angebotene Entscheidungen, eigene eingefrorene Gegner und PPO.
4. Ein knappes Ergebnis mit Testbedingungen. Historische Lernfortschritte
   ausdrücklich dem alten Regel-/Featurestand zuordnen; neue Betriebsfähigkeit
   und neue Spielstärke getrennt benennen.
5. Drei konkrete Engineering-/Forschungsentscheidungen mit Begründung, zum
   Beispiel Informationsgrenzen, festes Vergleichspanel und sicheres Resume.
6. Kurzes Setup und genau ein kurzer Trainings-/Evaluationsweg; vertiefende
   Bedienung per Link.
7. Nächste tatsächliche Forschungsfrage, Herkunft, Lizenz und technische Details.

Das Ziel ist keine größere Sammlung von Fachbegriffen. Ein Arbeitgeber sollte
erkennen können, welche Probleme verstanden wurden, wie Entscheidungen begründet
sind und wie belastbar die Ergebnisse sind. Umfangreiche historische
Trainerkompatibilität ist dafür kein notwendiges Qualitätsmerkmal.

Die Demo sollte vorerst als historische Demonstration erhalten bleiben. Eine
aktuelle Second-Edition-Demo benötigt einen passenden Berichtsexport aus
`decision_evaluation.py` oder eine Anpassung des Renderers. Gegenwärtig schreibt
der neue Evaluator unter anderem `summary.json` und `report.txt`; die alte
Dashboard-Bedienung erwartet ihren `analysis.json`-/Berichtsvertrag. Ein Austausch
der öffentlichen Zahlen ohne diesen Abgleich wäre keine fertige Migration.
Eine neue Website ist dafür nicht erforderlich.

**11. Empfohlene Umsetzung in überschaubaren Schritten**

1. **Aktuellen Stand sichern und Unterstützung festlegen.** Laufende Änderungen
   anderer Arbeit zusammenführen. Second Edition mit Gedächtnis als einziges
   normales Trainingssystem festlegen. Alten Stand durch bestehenden Commit
   oder bewusst gewählten Archiv-Tag referenzierbar lassen.
2. **Prüfaufwand begrenzen.** Markierungen reparieren, normalen aktiven Prüfumfang
   definieren, automatischen Legacy-Build daraus entfernen. Texte und
   Dokumentation künftig mit gezielten Prüfungen validieren.
3. **Leichte Altlasten entfernen.** Deaktivierte Trainer, ungenutzte historische
   Rezepte und überholte Betriebsanleitungen samt Verweisen bereinigen. Geteilte
   Liga-Komponenten und Dashboard vorerst erhalten.
4. **Trainer vereinheitlichen.** Gemeinsamer Collector/Trainingsablauf mit
   Workerzahl als Parameter. Einmal gezielt Budget, GAE-Grenzen, Stop/Resume und
   Liga-Konsistenz prüfen; kein neuer Langlauf allein zum Aufräumen.
5. **Dokumentation konsolidieren.** Aktuelle Architektur, ein Trainingsguide,
   eine Evaluationsanleitung und ein kurzer Status. Migration und ausgewählte
   Ergebnisse als abgeschlossene Belege behandeln; redundante Abnahmen bündeln.
6. **GitHub-Inhalt bewusst auswählen.** Den vollständigen Zielbaum und den
   Veröffentlichungsdiff prüfen, aktives Setup einmal aus sauberem Checkout
   ausführen und den bereinigten Stand auf `main` bringen. Keine Release-Tags
   still auf neue Commits verschieben.
7. **Präsentation separat abschließen.** README an Zielgruppe ausrichten,
   Demo-/Metadaten besser sichtbar machen und danach die aktive
   Dashboard-Anbindung sowie einen neuen öffentlichen Snapshot umsetzen.

Die ersten fünf Schritte reduzieren tatsächlichen Pflegeaufwand. Eine lediglich
neue README über dem unveränderten Doppelbetrieb würde das Kernproblem bestehen
lassen. Umgekehrt muss die Präsentationsüberarbeitung nicht warten, bis ein neuer
Agent eine größere Spielstärke nachgewiesen hat: Der aktuelle technische Stand
kann bereits klar und ehrlich gezeigt werden.

**12. Grenzen dieses Audits**

Es wurden keine Testsuiten, Benchmarks oder neuen Trainingsversuche ausgeführt.
Die Aussagen zur Teststruktur folgen aus Quellcode und CI-Konfiguration; der
erfolgreiche Remote-Lauf wurde separat gelesen. Historische Messwerte wurden
nicht reproduziert. Die Demo-Snapshotmetadaten wurden live geprüft, die
Browserdarstellung nicht visuell abgenommen. Ein exakter Löschpatch mit
abschließender Import-/Setup-Prüfung ist ein nachfolgender Umsetzungsschritt.
Der Arbeitsbaum war während der Prüfung in Benutzung; einzelne spätere Änderungen
können den dokumentierten Momentstand überholen.
