# Archiv der explorativen Phase

Stand: **29. Oktober 2026**. Diese Unterlagen belegen, was bisher versucht und
beobachtet wurde. Sie sind keine Liste aktuell auszuführender Aufgaben.
[Zur heutigen Dokumentation](../README.md).

## Was ist weiterhin wichtig?

| Themenbereich | Erkenntniswert | Wesentliche Unterlagen |
| --- | --- | --- |
| Lange Trainingsphasen | Ressourcenplanung verbesserte sich; zusätzlicher Fortschritt der zweiten Phase bleibt unsicher | [Frische Prüfung](long-training/LONG_FRESH_RESULTS_2026-10-26.md), [zweiter Lauf](long-training/LONG_TRAINING_V2_RESULTS_2026-10-29.md) |
| Köpfe und Priors | Frühe Vergleichsergebnisse und Grenzen der Zuschreibung an einen einzelnen Mechanismus | [Multi-Head-Ergebnis](exploration/MULTIHEAD_RESULTS_2026-10-26.md), [frische Prior-Prüfung](exploration/MULTIHEAD_FRESH_EVALUATION_2026-10-26.md) |
| Wissenschaft und Abwerfen | Diagnose des ursprünglichen Problems; erzwungenes Grün allein half dem getesteten Modell nicht | [Gemeinsamer Test](science/SCIENCE_HOLDOUT_RESULTS_2026-10-25.md), [Grün-Priorität](science/GREEN_PRIORITY_RESULTS_2026-10-25.md), [Abwurfanalyse](science/DISCARD_ANALYSIS_2026-10-26.md) |
| Spezialisten | Wissenschafts- und Militärreward führten zu unterschiedlichen Ergebnissen; Jokergrenze dokumentiert | [Reward-Protokoll](specialists/SPECIALIST_REWARDS.md), [Ergebnis](specialists/SPECIALIST_RESULTS_2026-10-26.md) |
| Frühes Self-Play | Erste Lernnachweise, damalige Features und Strategieeinschränkungen | [Erste Läufe](selfplay/SELFPLAY_POC_RESULTS_2026-10-23.md), [Feature-v3-Ergebnis](selfplay/SELFPLAY_FEATURE_V3_RESULTS_2026-10-24.md) |
| Geschwindigkeit | Messgrundlage und Vorher-Nachher-Nachweise technischer Optimierungen | [Audit](performance/PERFORMANCE_AUDIT_2026-10-25.md), [Umsetzungsergebnis](performance/PERFORMANCE_RESULTS_2026-10-25.md) |
| Technische Validierung | Datierte Prüfungen während der Entwicklung, kein aktueller vollständiger Teststatus | [Validierungsprotokoll](validation/VALIDATION.md) |

## Wie die Unterlagen zu lesen sind

- Protokolle dokumentieren frühere Entscheidungen und Befehle. Aussagen wie
  „als Nächstes“ oder „noch nicht gestartet“ gelten für den damaligen Stand.
  Für den heutigen Status sind [Forschung und Dokumentation](../README.md) maßgeblich.
- Ergebnisberichte einschließlich negativer und unsicherer Befunde bleiben
  erhalten. Ein später besseres Modell macht frühere Vergleiche nicht wertlos.
- JSON-Dateien sind ausgewählte strukturierte Mess- und Auswertungszusammenfassungen;
  sie sind keine vollständigen Rohdatenbestände. Ihre Inhalte wurden bei der
  Bereinigung unverändert erhalten. Darin gespeicherte Pfade können die
  ursprüngliche Ablage vor der Umsortierung beschreiben.
- Markdown-Verweise wurden an die neue Struktur angepasst. Modelle, Rohlogs und
  Laufverzeichnisse wurden nicht verschoben. Historische Befehle setzen die dort
  genannten lokalen Artefakte und die jeweils passende Software voraus.
- Dass ein Protokoll Vergleiche oder Wiederholungen vorschlug, belegt nicht,
  dass sie tatsächlich ausgeführt wurden. Dafür sind die Ergebnisberichte zuständig.

## Vollständiger Bestand

Die Gruppierung folgt dem Zweck des Versuchs, nicht bloß seinem Datum.

### Lange Trainingsphasen

- [Neue Seeds und Direktvergleich: Ergebnis](long-training/LONG_FRESH_RESULTS_2026-10-26.md)
- [Ergebnis des langen Laufs mit öffentlichen Informationen](long-training/LONG_TRAINING_RESULTS_2026-10-26.md)
- [Vorbereiteter Lauf: 2.097.152 zusätzliche Übergänge](long-training/LONG_TRAINING_RUN_V1.md)
- [Zweiter langer Lauf: eigene Gegner, 60/20/20](long-training/LONG_TRAINING_RUN_V2.md)
- [LONG_TRAINING_V2_RESULTS_2026-10-29 — strukturierte Messzusammenfassung](long-training/LONG_TRAINING_V2_RESULTS_2026-10-29.json)
- [Zweiter langer Trainingslauf: Ergebnis vom 29. Oktober 2026](long-training/LONG_TRAINING_V2_RESULTS_2026-10-29.md)
- [LONG_TRAINING_VALIDATION_2026-10-26 — strukturierte Messzusammenfassung](long-training/LONG_TRAINING_VALIDATION_2026-10-26.json)

### Mehrere Köpfe und Priors

- [One multi-head exploration experiment](exploration/MULTIHEAD_EXPLORATION.md)
- [MULTIHEAD_FRESH_EVALUATION_2026-10-26 — strukturierte Messzusammenfassung](exploration/MULTIHEAD_FRESH_EVALUATION_2026-10-26.json)
- [Frische Evaluation von v1 und v2, 26. Oktober 2026](exploration/MULTIHEAD_FRESH_EVALUATION_2026-10-26.md)
- [MULTIHEAD_PRIORS_RESULTS_2026-10-26 — strukturierte Messzusammenfassung](exploration/MULTIHEAD_PRIORS_RESULTS_2026-10-26.json)
- [Auswertung des Multi-Head-Laufs mit Priors, 26. Oktober 2026](exploration/MULTIHEAD_PRIORS_RESULTS_2026-10-26.md)
- [Next experiment: four heads with calibrated frozen priors](exploration/MULTIHEAD_PRIORS_V2.md)
- [MULTIHEAD_RESULTS_2026-10-26 — strukturierte Messzusammenfassung](exploration/MULTIHEAD_RESULTS_2026-10-26.json)
- [First multi-head experiment: results, 26 October 2026](exploration/MULTIHEAD_RESULTS_2026-10-26.md)

### Wissenschaft und Ressourcen

- [DISCARD_ANALYSIS_2026-10-26 — strukturierte Messzusammenfassung](science/DISCARD_ANALYSIS_2026-10-26.json)
- [Abwurfverhalten trainierter Modelle nach Zeitalter](science/DISCARD_ANALYSIS_2026-10-26.md)
- [Mad Scientist: dasselbe Modell mit und ohne Grün-Priorität](science/GREEN_PRIORITY_EVALUATION.md)
- [Grün-Priorität gegen freie Policy: Ergebnis](science/GREEN_PRIORITY_RESULTS_2026-10-25.md)
- [Wissenschaft: Exploration und spätes Lernsignal diagnostizieren](science/SCIENCE_DIAGNOSTICS.md)
- [Wissenschaftsdiagnose: vier Varianten, Seed 31](science/SCIENCE_DIAGNOSTIC_RESULTS_2026-10-25.md)
- [Gemeinsame Evaluation der vier Wissenschaftsvarianten](science/SCIENCE_HOLDOUT_RESULTS_2026-10-25.md)

### Spezialisten

- [SPECIALIST_RESULTS_2026-10-26 — strukturierte Messzusammenfassung](specialists/SPECIALIST_RESULTS_2026-10-26.json)
- [Spezialisten gegen eingefrorene Standardgegner: Auswertung vom 26. Oktober 2026](specialists/SPECIALIST_RESULTS_2026-10-26.md)
- [Wissenschafts- und Militärspezialisten: Plan und Reward-Vertrag](specialists/SPECIALIST_REWARDS.md)

### Grundlagen und frühes Self-Play

- [Korrigierte Spielmerkmale – 23.10.2026](selfplay/FEATURE_CORRECTIONS_2026-10-23.md)
- [Self-play assessment: feature semantics v3](selfplay/SELFPLAY_FEATURE_V3_RESULTS_2026-10-24.md)
- [Auswertung des moderaten Self-Play-Laufs vom 23.10.2026](selfplay/SELFPLAY_MODERATE_RESULTS_2026-10-23.md)
- [Self-play proof of concept](selfplay/SELFPLAY_POC.md)
- [SELFPLAY_POC_RESULTS_2026-10-23 — strukturierte Messzusammenfassung](selfplay/SELFPLAY_POC_RESULTS_2026-10-23.json)
- [Self-Play-Prototyp: Ergebnisse vom 23. Oktober 2026](selfplay/SELFPLAY_POC_RESULTS_2026-10-23.md)
- [SELFPLAY_STRATEGY_AUDIT_2026-10-23 — strukturierte Messzusammenfassung](selfplay/SELFPLAY_STRATEGY_AUDIT_2026-10-23.json)
- [Strategieauswertung und Audit der verfügbaren Spieldaten](selfplay/SELFPLAY_STRATEGY_AUDIT_2026-10-23.md)

### Geschwindigkeit und Laufzeit

- [PERFORMANCE_AUDIT_2026-10-25 — strukturierte Messzusammenfassung](performance/PERFORMANCE_AUDIT_2026-10-25.json)
- [Performance-Prüfung vom 25. Oktober 2026](performance/PERFORMANCE_AUDIT_2026-10-25.md)
- [PERFORMANCE_RESULTS_2026-10-25 — strukturierte Messzusammenfassung](performance/PERFORMANCE_RESULTS_2026-10-25.json)
- [Runtime-Optimierungen und Vorher-Nachher-Messung, 25. Oktober 2026](performance/PERFORMANCE_RESULTS_2026-10-25.md)
- [RUNTIME_BENCHMARK_2026-10-23 — strukturierte Messzusammenfassung](performance/RUNTIME_BENCHMARK_2026-10-23.json)
- [SELFPLAY_RUNTIME_OPTIMIZATION_2026-10-23 — strukturierte Messzusammenfassung](performance/SELFPLAY_RUNTIME_OPTIMIZATION_2026-10-23.json)
- [Self-play runtime optimizations: 23 October 2026](performance/SELFPLAY_RUNTIME_OPTIMIZATION_2026-10-23.md)

### Historische technische Prüfungen

- [Validation](validation/VALIDATION.md)
