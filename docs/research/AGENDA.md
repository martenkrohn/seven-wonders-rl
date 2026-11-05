# Wie wir die offenen Trainingsfragen klären können

Die drei vereinbarten Leitfragen für die weitere Arbeit. Stand: 29. Oktober 2026.
Die zugehörige Recherche ist inzwischen in der [Content-Bank](CONTENT_BANK.md)
dokumentiert; der [Überblick](OVERVIEW.md) fasst sie kurz zusammen.
Die Erweiterung der Informationsgrundlage ist inzwischen implementiert;
die neue [Gegnerpaar-Liga](../guides/PAIR_PFSP_TRAINING.md) ist eingerichtet.
Der [frische Ausgangslauf](../experiments/001-fresh-pair-pfsp/PROTOCOL.md)
ist vorbereitet; ein kontrollierter Methodenvergleich bleibt offen.

## Konkretisierter Arbeitsfokus

Nach Besprechung des Überblicks sind drei Anforderungen für die nächste Phase
festgehalten. Die ersten beiden sind inzwischen umgesetzt und im Trainer standardmäßig aktiv
([Beobachtungsgedächtnis](../guides/OBSERVATION_MEMORY.md)); ihr Nutzen ist noch zu messen:

1. **Eigene Hand für die Zustandsbewertung:** Der Critic soll die komplette
   aktuelle Hand berücksichtigen können. Ihre bisherige Auslassung ist keine
   PPO-Vorgabe und im verfügbaren Code nicht fachlich begründet.
2. **Erinnerung an beobachtete Hände:** Actor und Critic sollen die zuvor selbst
   gesehenen Kartenpakete und den relevanten Verlauf nutzen können. Im
   Dreierspiel ist insbesondere die Rückkehr eines Pakets nach drei
   Weitergaben wichtig. Die explizite Beobachtungshistorie ist implementiert.
3. **Messbare Gegnerauswahl:** Von festen, händisch gewählten Gruppenanteilen
   zu einer nachvollziehbaren Regel anhand von Ergebnissen, Unsicherheit,
   Vergessen und strategischer Abdeckung übergehen. Bei drei Spielern sind
   Gegnerpaare maßgeblich. Die erste Regel verwendet diskontierte Sieganteile,
   PFSP-Gewichte und gleichverteilte Kontrollauswahlen; ihr Nutzen ist zu prüfen.

Begründung, Abgrenzung und Prüfkriterien stehen in den
[Prioritäten der Content-Bank](CONTENT_BANK.md#prioritaeten). Die folgenden
drei Leitfragen bleiben der wissenschaftliche Rahmen.

## 1. Das bestehende Verfahren einordnen

Actor und Critic gehören bei uns zum PPO-Grundverfahren. Mehrere Köpfe,
Prior-Netze und die konkrete Gegnerzusammenstellung sind zusätzliche
Entscheidungen. Der aktuelle Reward bewertet ausschließlich den Punkteabstand
am Spielende; bestimmte Strategien werden nicht ausdrücklich belohnt.

**Vorgehen:** Eine kleine Übersicht erstellen: Was verwenden wir, warum,
seit wann und mit welcher wissenschaftlichen Grundlage? Dabei prüfen, ob
unsere Umsetzung dem jeweiligen Forschungsansatz tatsächlich entspricht.
Das trennt etablierte Grundlagen von unseren eigenen Anpassungen.

## 2. Exploration und Gegnerauswahl recherchieren

Deine beiden Sorgen hängen zusammen: Eine langfristige Strategie muss zunächst
ausprobiert werden und gegen die Trainingsgegner überhaupt lohnend sein.
Mehr aktuelle Gegner oder mehr unterschiedliche Checkpoints lösen diese
Fragen nicht automatisch. Auch der Nutzen älterer Anker sollte gemessen werden,
statt ihn allein aus ihrem Alter oder ihrer Gesamtstärke abzuleiten.

**Vorgehen:** Forschung zu langfristiger Exploration und Gegnerpopulationen
gezielt danach vergleichen, welche Belege sie für Spiele wie unseres liefert.
Wichtig sind Voraussetzungen, Vergleichsversuche, Rechenaufwand und die Frage,
ob Ergebnisse aus Zwei-Spieler-Spielen auf unser Dreierspiel übertragbar sind.
Zunächst wenige plausible Richtungen sammeln, noch keinen Sieger auswählen.

## 3. Die erreichbare Aussage festlegen

Wir möchten wissen, ob ein Modell stärker und anpassungsfähiger wird – nicht
nur, ob es gegen bekannte Gegner häufiger gewinnt oder mehr Wissenschaft baut.
Ein wissenschaftlich begründetes Verfahren wäre eine bessere Grundlage, aber
noch kein Nachweis des bestmöglichen Spielers.

**Vorgehen:** Vor dem nächsten Versuch Erfolgskriterien festlegen: bessere
Ergebnisse gegen unterschiedliche Gegner, belastbare Vergleiche auf neuen
Seeds und erkennbare Reaktionen auf verschiedene Spielsituationen. Auch
festhalten, welche Ergebnisse gegen unsere Vermutung sprechen würden.

