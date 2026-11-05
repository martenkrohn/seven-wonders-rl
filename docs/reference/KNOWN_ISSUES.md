# Bekannte Grenzen

Stand: **29. Oktober 2026**. Diese Datei beschreibt gegenwärtige Grenzen;
historische Befunde stehen im [Archiv](../archive/README.md), technische
Verträge in der [Architektur](ARCHITECTURE.md). Funktionsfähige Tests und
Lernfortschritte belegen keine vollständige Regelkorrektheit oder Optimalität.

## Informationen und Modell

Aktuelle Modelle unterstützen öffentliche Kartenidentitäten, Wunderidentitäten
und Stufenpläne über `--public-information` (645 globale und 121 Merkmale je
Aktionsposition). Die Basis ohne Erweiterung hat 102/99 Merkmale. Die
Startressource des Wunders ist bereits in den Produktionssummen enthalten.
Die frühere Aussage, Wunderidentitäten und Stufeninformationen seien generell
nicht vorhanden, ist damit überholt.

Mit `--no-observation-memory` verarbeitet der Critic weiterhin nur den bisherigen
globalen Vektor und das Modell erhält keinen Verlauf. Die neue optionale
[Gedächtniserweiterung](../guides/OBSERVATION_MEMORY.md) stellt vollständige eigene
Hand, legale Aktionsinformationen und beobachteten Verlauf beiden Netzteilen
bereit. Die historischen Checkpoints erhalten dadurch nicht nachträglich eine
gelernte Erinnerungsfähigkeit; ein Nutzen muss durch neues Training geprüft werden.
Auch die Erweiterung arbeitet als Feedforward-Netz über einer begrenzten
expliziten Historie, nicht als rekurrentes Netz. Sie rekonstruiert keine sichere
aktuelle Gegnerhand aus verdeckten Entscheidungen.

Vier Köpfe teilen Encoder und Training. Ihre festen zufälligen Priors wirken auf
Actor-Logits, nicht auf Value-Ziele. Daraus folgen weder unabhängige
Trainingswiederholungen noch Garantien für Deep Exploration. Einordnung und
Quellen stehen in der [Content-Bank](../research/CONTENT_BANK.md#exploration).

Historische Architektur-v1-Modelle behalten für ausdrückliche Legacy-Inferenz
ihren linearen Kopf. Sie werden nicht stillschweigend in die neue Architektur
konvertiert. Checkpoint-, Feature- und Datenbankschemata müssen kompatibel sein.

## Simulator und Aktionsschnittstelle

Die aktuelle Engine ist keine vollständige Migration auf jede neuere Edition.
Der [Editionsaudit](EDITION_AUDIT.md) belegt, dass die Kartendeklarationen noch
mit dem festgehaltenen Upstream übereinstimmen, und dokumentiert insbesondere
die ursprünglichen Exportfehler bei Wunder-IDs, Spielerzahlen und
Produktionsmengen. Diese Exportfehler sind seit Paket 2 vom 5. November 2026
[behoben und geprüft](../guides/LEGACY_DATA_EXPORT.md). Der Audit beschreibt
auch die Migration auf das Grundspiel von 2020; umgesetzt ist sie noch nicht.
Weitere geerbte Regelfehler außerhalb gezielter Regressionen sind möglich.
Insbesondere bleibt die zusätzliche Auswahl aus dem Ablagestapel bei
Halikarnassos über `extra` unvollständig unterstützt. Handelsdetails werden von
der Engine bestimmt und sind kein frei auswählbarer vollständiger
Bezahlaktionsraum der Policy.

`Player::CalculateScientificScore()` teilt Wissenschaftsjoker gierig zu. In der
Spezialistenprüfung hätte eine erschöpfende Zuordnung in 52/480 Wissenschafts-,
11/480 Militär- und 6/480 Ausgangsmodell-Partien 1–3 zusätzliche mögliche
Wissenschaftspunkte geliefert. Die Analyse änderte die Engine nicht; hypothetische
Punktgewinne sind ohne Neuberechnung aller Plätze keine korrigierten Siege.
Siehe [Beleg und Grenzen](../archive/specialists/SPECIALIST_RESULTS_2026-10-26.md#zusätzlich-gefundene-simulatorgrenze).

Gewinn- und Ressourcenmerkmale im Bericht sind teilweise Schätzungen. Münzdeltas
sind wegen Nachbarzahlungen keine exakten Handelskosten; Kartenfarbe und
Wertungskategorie sind nicht identisch. Fehlende Telemetrie wird nicht durch
vermeintlich exakte Werte ersetzt.

## Training, Wiederaufnahme und Reproduzierbarkeit

Der vereinheitlichte Trainer ist CPU-basiert. Schrittbudgets müssen zur Zahl der
Umgebungen passen. Transportfehler liefern keinen gültigen Übergang; der
betroffene Rollout wird verworfen, Diagnosen und bereits gespeicherte Modelle
bleiben erhalten.

Milestone-Läufe unterstützen `--resume`: abgeschlossene Updates, Optimizer und
Zufallszustände werden fortgesetzt. Laufende Engine-Partien werden als
aufgegeben protokolliert und mit neuen Episodenseeds ersetzt. Das ist **keine
bitgenaue Fortsetzung jedes Spielzustands**. Die Option ist nur für die vom
Trainer unterstützten Konfigurationen verfügbar.

`--init-checkpoint` startet dagegen eine neue Phase mit vorhandenen Gewichten
und frischem Optimizer. `--restart-incomplete` archiviert einen passenden
unfertigen Lauf und startet neu. Diese drei Verfahren dürfen nicht gleichgesetzt
werden. Details: [Entwicklung](../guides/DEVELOPMENT.md).

Gleiche Seeds sichern keine identischen Resultate über andere Betriebssysteme,
Compiler/STL- oder Bibliotheksversionen. Abhängigkeiten sind nicht vollständig
festgeschrieben. Ältere Läufe vor korrigierter GAE und Endwertung sind keine
Evidenz für den aktuellen Trainer.

## Gegnerpool und Bewertung

Der letzte lange Lauf nutzt jüngere Snapshots, ein zeitlich gestreutes Archiv
und feste Anker mit 60/20/20-Gruppenanteilen je Gegnerplatz. Das ist eine
Projektheuristik, keine leistungsabhängige Liga und kein implementiertes
PSRO/JPSRO. Zeitlich verschiedene Modelle garantieren keine verschiedenen
Strategien. Die zwei Gegner werden unabhängig gezogen.

Die neu eingerichtete [Paar-PFSP-Variante](../guides/PAIR_PFSP_TRAINING.md)
zieht stattdessen geordnete Gegnerpaare anhand diskontierter Trainingsresultate.
Ihre Schätzungen mitteln über sich verändernde Lerner und Kopf-Mischungen;
sie sind keine exakten aktuellen Policy-Auszahlungen. Pseudo-Spielzahlen,
Alterung und Kontrollauswahlen begrenzen Fehlsteuerung durch wenige oder alte
Beobachtungen, garantieren aber keine ausreichende Abdeckung oder neue Strategien.
Der neue lange Lauf ist vorbereitet; ein Leistungsvorteil ist noch nicht belegt.

Das Standardziel bleibt Punkteabstand am Spielende, nicht unmittelbar Siegchance.
Das regelmäßig betrachtete Dashboardpanel ist eine Entwicklungskontrolle,
kein dauerhaft unberührter Abschlusstest. Neue Spielseeds, neue Gegner und
unabhängige Trainingswiederholungen beantworten unterschiedliche Fragen.
Siehe [Evaluationsgrundlage](../research/CONTENT_BANK.md#evaluation).

## Laufzeit und Speicherung

Dateibasierter Austausch und ein Engineprozess pro Partie bleiben bestehen.
RAM-Austausch und gebündelte Gegnerinferenz reduzieren Aufwand, beseitigen ihn
aber nicht. Der Pool begrenzt die zugelassenen Gegner; er löscht keine alten
Checkpointdateien. Der tatsächlich erreichte Durchsatz hängt von Konfiguration,
Hardware, Gegnermenge und Evaluation ab.
