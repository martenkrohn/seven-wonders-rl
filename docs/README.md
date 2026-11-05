# Projektdokumentation

Stand: **29. Oktober 2026**. Das Projekt wechselt von explorativen Versuchen
zu gezielten, wissenschaftlich begründeten Vergleichen. Dieser Einstieg zeigt,
welche Dokumente heute maßgeblich sind und welche frühere Versuche belegen.

## Hier anfangen

| Frage | Dokument | Zweck |
| --- | --- | --- |
| Wo stehen wir wissenschaftlich? | [Forschungsüberblick](research/OVERVIEW.md) | Kurze Einordnung und Verweise zum Vertiefen |
| Welche Fragen wollen wir klären? | [Forschungsagenda](research/AGENDA.md) | Die drei vereinbarten Leitfragen |
| Was sagen Literatur, Code und Daten? | [Content-Bank](research/CONTENT_BANK.md) | Ausführliche Recherche mit Quellen und Übertragungsgrenzen |
| Wie halten wir kommende Versuche fest? | [Versuchsablauf und Vorlage](experiments/README.md) | Eine Prüffrage, ein nachvollziehbares Protokoll, eine Auswertung |

Die Recherche ist abgeschlossen. Als nächster Baustein ist die
[gemeinsame Hand- und Verlaufseingabe für Actor und Critic](guides/OBSERVATION_MEMORY.md)
implementiert; ihr Nutzen für die Spielstärke ist noch zu prüfen. Die neue
[Liga mit leistungsabhängiger Gegnerpaar-Auswahl](guides/PAIR_PFSP_TRAINING.md)
ist eingerichtet. Der [frische Ausgangslauf](experiments/001-fresh-pair-pfsp/PROTOCOL.md)
ist vorbereitet, aber noch nicht gestartet. Frühere Startbefehle
im Archiv sind keine aktuelle Empfehlung für einen weiteren langen Lauf.

Für den nun gewünschten Betrieb ohne Endbudget sind
[VS-Code-Tasks zum Starten, Stoppen und Fortsetzen](guides/CONTINUOUS_TRAINING.md)
eingerichtet. Der lange Lauf wird erst durch den Start-Task begonnen.
Das [bereinigte Dashboard](guides/EVALUATION.md#bereinigte-dashboard-auswahl)
zeigt nur diese neuen Auswertungen und die letzte Evaluation des festgelegten
bisherigen Endmodells als Referenz.

Engine und alte GUI sind jetzt getrennt vom RL-Projekt wartbar. Die weiteren
Schritte zur BGA-Edition stehen im [Engine-Arbeitsplan](../engine/docs/MIGRATION_PLAN.md).
Der [nächste Agentenauftrag](../engine/docs/NEXT_AGENT_TASK.md) beschränkt sich
auf das erste offene Arbeitspaket; Regellogik und neue Grafik folgen getrennt.
Paket 2 macht die [alten Datenexporte](guides/LEGACY_DATA_EXPORT.md) mit
geprüftem Daten-Diff reproduzierbar; die aktiven Regeln bleiben Legacy.

## Aktueller Bezugspunkt

Der letzte abgeschlossene Lauf ist `long-mixed-wonders-v2-seed231`: vier Köpfe,
feste Actor-Logit-Priors, öffentliche Karten-/Wunderinformationen und ein
60/20/20-Pool eigener jüngerer, archivierter und fester Modelle. Der Standardreward
bewertet den Punkteabstand zum stärksten Gegner am Spielende.

Die [frische Prüfung des ersten langen Laufs](archive/long-training/LONG_FRESH_RESULTS_2026-10-26.md)
belegt Fortschritt auf ihrem Testplan. Der
[zusätzliche Fortschritt des zweiten Laufs](archive/long-training/LONG_TRAINING_V2_RESULTS_2026-10-29.md)
ist noch unsicher. Das sind historische Befunde, keine allgemeine
Optimalitätsaussage. Die nächste Methodenentscheidung soll auf einer klaren
Prüffrage und passenden Erfolgskriterien beruhen.

## Nach Aufgabe geordnet

| Bereich | Was gehört hinein? | Einstieg |
| --- | --- | --- |
| `research/` | Aktuelle Fragen, Literatur und fachliche Einordnung | [Überblick](research/OVERVIEW.md) |
| `experiments/` | Neue, gezielt formulierte Versuche mit Protokoll und Ergebnissen | [Arbeitsweise](experiments/README.md) |
| `reference/` | Tatsächliche technische Verträge und gegenwärtige Grenzen | [Architektur](reference/ARCHITECTURE.md), [bekannte Grenzen](reference/KNOWN_ISSUES.md) |
| `guides/` | Praktische Bedienung ohne historische Versuchsentscheidungen | [Entwicklung](guides/DEVELOPMENT.md), [Beobachtungsgedächtnis](guides/OBSERVATION_MEMORY.md), [Evaluation und Dashboard](guides/EVALUATION.md), [öffentliche Demo](guides/DASHBOARD_DEMO.md) |
| `archive/` | Protokolle, Ergebnisse und Messzusammenfassungen der explorativen Phase | [Kommentiertes Archiv](archive/README.md) |
| `engine/` (Submodule im Projektstamm) | Eigenständige Engine, Tests und Herkunftsnachweis | [Engine-Arbeitsweise](guides/ENGINE.md) |

Der [Editionsaudit zu Karten und Wundern](reference/EDITION_AUDIT.md) prüft den
aktuellen Simulator gegen das Grundspiel von 2020 und beschreibt die nötige
Migration. Er enthält bestätigte Abweichungen, Exportfehler und noch offene
Einzelkosten; die Spielregeln wurden durch den Audit noch nicht umgestellt.
Die [vollständige Gegenüberstellung vom 5. November](reference/edition-comparison/README.md)
enthält alle Karten und Wunderseiten sowie einen lokalen grafischen Bericht
mit den vorhandenen alten Bildern und den neuen Bildern aus der lokalen
BGA-Referenz. Die zugehörigen neuen Definitionen sind abgeglichen; vier
Kostenwerte der früheren lokalen Vorarbeit wurden dabei korrigiert.

[ORIGIN.md](ORIGIN.md) und [SIMULATOR_CHANGES.md](SIMULATOR_CHANGES.md) bleiben
wegen der Verweise in Herkunftshinweisen und Quellcode an ihren festen Orten.
Sie dokumentieren Herkunft und Änderungen, nicht die aktuelle Forschungsplanung.
Die ausführliche Werkzeugliste steht in
[python/scripts/README.md](../python/scripts/README.md).

## Regeln für die weitere Pflege

- Eine aktuelle Aussage hat einen maßgeblichen Ort: Forschungsrichtung in
  `research/`, Bedienung in `guides/`, technische Verträge in `reference/`.
- Neue Versuche erhalten einen eigenen Ordner in `experiments/`. Protokoll,
  Ergebnis und strukturierte Zusammenfassung bleiben zusammen. Ein Datum
  allein ist kein ausreichender Dokumentzweck.
- Historische Befunde werden nicht nachträglich zu aktuellen Anweisungen
  umgeschrieben. Korrekturen und Grenzen werden ausdrücklich gekennzeichnet.
- Große Rohlogs, Checkpoints und erzeugte Dashboards bleiben in `python/runs/`.
  Dokumente nennen ihre Herkunft und relevante Prüfsummen; das Archiv enthält
  ausgewählte Auswertungen, nicht sämtliche Rohdaten.
- Überholte Aufgabenlisten werden ersetzt statt unbegrenzt ergänzt. Vor dem
  Löschen prüfen wir, ob ein Dokument noch einzigartige Evidenz oder ein
  notwendiges Versuchsprotokoll enthält.

## Bei dieser Bereinigung entfernt

| Dokument | Warum entfernt? | Maßgeblicher Ersatz |
| --- | --- | --- |
| `NEXT_STEPS.md` | Verkettete frühere Aufgaben und widersprüchliche Zwischenstände | Dieser Einstieg und die Forschungsagenda |
| `LONG_TRAINING_RECOMMENDATION_2026-10-26.md` | Vorläufige Empfehlung vor inzwischen abgeschlossenen Läufen | Tatsächliche Laufprotokolle und Ergebnisberichte im Archiv |
| `TRAINING_RESEARCH_QUESTIONS_2026-10-29.md` | Früheres Brainstorming; durch die vereinbarten drei Leitfragen und die Recherche abgelöst | Forschungsagenda, Überblick und Content-Bank |

Ergebnisberichte und strukturierte Messdaten wurden erhalten. Ihre Archivierung
entzieht ihnen nicht den Wert als Nachweis; sie trennt frühere Befunde von
heutigen Entscheidungen.
