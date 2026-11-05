# Wissenschaftsdiagnose: vier Varianten, Seed 31

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Nachtrag: Der gemeinsame Holdout ist inzwischen ausgewertet. Siehe
[SCIENCE_HOLDOUT_RESULTS_2026-10-25.md](SCIENCE_HOLDOUT_RESULTS_2026-10-25.md)
für den Vergleich auf identischen Gegnern und Testspielen.

Ausgewertet am 25. Oktober 2026. Alle vier Läufe sind vollständig abgeschlossen:
jeweils **524.288 Transitionen**, 1.024 PPO-Updates und 24.966 abgeschlossene
Trainingspartien. Es sind rund 500.000 Transitionen pro Variante, exakt 524.288.

**Befund:** Erfolgreiche Wissenschaftsfolgen kommen bereits sehr früh vor. Nach
einer Phase nahezu vollständiger Vermeidung entwickelt auch der unveränderte
Standardtrainer später deutlich mehr frühe Wissenschaftsinvestitionen.
Höhere Entropie liefert bei diesem Seed keinen zusätzlichen Vorteil für die
späte Wissenschaftsdiagnose. GAE-Lambda 1,0 geht hier mit deutlich weniger
Wissenschaft einher. Ein allgemeiner Spielstärkesieger ist noch nicht bestimmt.

## Datenprüfung

Alle 99.864 Wissenschaftsprotokolle wurden unabhängig gegen die regulären
Episode-Protokolle geprüft: eindeutige Umgebung/Episode, Engine-Seed, Sitz,
Gegner, Endpunktzahlen und Wissenschaftswertung stimmen überein. Alle Partien
haben die drei Zeitalter-Snapshots; alle 1.801.677 erfassten Entscheidungen
haben geordnete eindeutige Züge, plausible Wahrscheinlichkeiten und konsistente
legale grüne Angebote. Die acht Fenster jeder Variante wurden aus den
Rohprotokollen neu berechnet und stimmen exakt mit der gespeicherten
Zusammenfassung überein.

Zusätzlich wurden alle 32 Checkpoint-Evaluationen geprüft: zusammen 4.608
abgeschlossene Spiele, null Fehler oder Abbrüche. Die wissenschaftlichen
Mittelwerte stimmen mit den einzelnen Spielprotokollen überein. Alle vier
Update-Dateien enthalten 1.024 Updates mit je 512 Transitionen ohne Truncations.
Es wurde keine neue Partie und kein neues Training für diese Analyse gestartet.

Die gespeicherten Konfigurationen unterscheiden sich nur in Entropiekoeffizient,
GAE-Lambda und Ausgabe-/Austausch-/League-Pfaden. Alle verwenden Seed 31,
Gamma 0,999, Merkmalssemantik v3 und Architektur v2. Die Manifeste nennen denselben
Commit `1c9ea06a06513f1f7b1ff46385d49d8bc996d944` und denselben Simulatorhash,
aber jeweils einen veränderten Arbeitsbaum (`dirty=true`). Ohne gesonderten
Quellcode-Snapshot ist damit nicht bytegenau bewiesen, dass sämtlicher Python-Code
während der vier Starts identisch war.

## Entdecken: kein vollständiges Ausbleiben erfolgreicher Folgen

Die erste Partie mit frühem Grün, mindestens zehn Wissenschaftspunkten und ohne
Punktrückstand endet in **allen vier Varianten bei Transition 294**. Sie stammt
noch vom untrainierten Modell, vor dem ersten PPO-Update, gegen zwei untrainierte
Gegner. Der Agent baut Scriptorium und Workshop in Zeitalter I, später Lodge:
ein vollständiges Symbolset, zehn Wissenschaftspunkte, VP-Vorsprung +22.

Die erste entsprechende Partie ab 20 Wissenschaftspunkten endet beim Standard
und höherer Entropie bei Transition 7.434, bei beiden Lambda-1-Varianten bei 3.906.
Alle enthalten sechs tatsächlich gebaute grüne Karten und ergeben 21
Wissenschaftspunkte. Hier sind überwiegend ebenfalls untrainierte oder sehr
frühe Gegner beteiligt. Die gleichen Seeds und Anfangsgewichte erklären die
identischen ersten Partien; dies sind keine vier unabhängigen Entdeckungsbelege.

Damit ist die starke Hypothese „erfolgreiche Wissenschaftsfolgen werden überhaupt
nie erlebt“ für diese Läufe widerlegt. Das beweist weder eine verlässlich gute
Strategie gegen starke Gegner noch, dass alle nützlichen Folgen ausreichend
exploriert werden. Einzelne günstige Partien sind kein Lernnachweis.

## Entwicklung über die Trainingszeit

Grüne Bauquote in Zeitalter I: Anteil der Entscheidungen mit legalem grünem
Gebäudebau, bei denen tatsächlich ein grünes Gebäude gebaut wurde.

| Fensterbeginn | Standard | Entropie 0,03 | Lambda 1,0 | Kombination |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 6,97 % | 6,67 % | 9,48 % | 8,18 % |
| 65.536 | 0,14 % | 0,53 % | 3,19 % | 3,64 % |
| 131.072 | 2,07 % | 0,19 % | 5,65 % | 1,50 % |
| 196.608 | 7,50 % | 0,89 % | 8,38 % | 0,72 % |
| 262.144 | 8,47 % | 10,21 % | 8,87 % | 1,83 % |
| 327.680 | 20,79 % | 31,57 % | 4,43 % | 1,39 % |
| 393.216 | 40,55 % | 36,98 % | 4,65 % | 2,59 % |
| 458.752 | 41,01 % | 38,15 % | 4,16 % | 4,36 % |

Fenster umfassen jeweils 65.536 Transitionen, gruppiert nach Partiebeginn.
Der Standard unterdrückt frühes Grün zunächst fast vollständig und entwickelt
es nach ungefähr 300.000 Transitionen deutlich häufiger. Höhere Entropie zeigt
eine ähnliche späte Entwicklung. Die tatsächliche Policy-Wahrscheinlichkeit
steigt ebenfalls: Das ist kein bloß zufällig häufigeres Ziehen grüner Aktionen.
Die besuchten Spielzustände ändern sich allerdings mit; die Mittelwerte sind
kein Vergleich derselben fixierten Beobachtungen vor und nach einem Update.

## Letztes Trainingsviertel

Für jede Variante werden die letzten beiden Fenster zusammengefasst:
Partiebeginn zwischen 393.216 und 524.288, jeweils 6.240 abgeschlossene Partien.
Quoten sind nach ihren tatsächlichen Nennern gewichtet, nicht einfach über
Fenster gemittelt.

| Variante | Grün I gebaut / legal | Bauquote | Policy-P bei legalem Grün I | Wissenschaft Ø | Spiele ≥10 / ≥20 | Frühes Grün + ≥10 + kein Rückstand |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard (.01 / .95) | 2.112 / 5.180 | 40,77 % | 40,89 % | 4,98 | 1.255 / 153 | 334 |
| Entropie (.03 / .95) | 1.847 / 4.919 | 37,55 % | 37,57 % | 3,36 | 754 / 70 | 184 |
| Lambda (.01 / 1.0) | 243 / 5.521 | 4,40 % | 4,45 % | 0,61 | 57 / 0 | 1 |
| Kombination (.03 / 1.0) | 186 / 5.350 | 3,48 % | 3,98 % | 0,35 | 17 / 0 | 1 |

Beim Standard treten in diesem Abschnitt außerdem 81 Partien mit frühem Grün,
mindestens 20 Wissenschaftspunkten und ohne Rückstand auf; bei höherer Entropie
sind es 35, bei den beiden Lambda-1-Varianten null. Das ist deutlich mehr als
eine einzelne frühe Zufallsentdeckung. Zugleich bleibt Wissenschaft unzuverlässig:
selbst im Standard erreichen 4.985 von 6.240 Partien keine zehn Wissenschaftspunkte.

### Was bedeutet hier „erfolgreich“?

Die vorab definierten 10/20 Punkte sind bewusst Beobachtungsschwellen, keine
fachlich begründeten Grenzwerte für starke Wissenschaftsstrategien. Bereits ein
Set aus drei unterschiedlichen Symbolen ergibt zehn Punkte. Der Nichtverlust
kann außerdem durch andere Punktequellen zustande kommen. Daher sollte die
Bezeichnung als „frühe Wissenschaft mit Nichtverlust“ gelesen werden und nicht
als „Wissenschaft als erfolgreich erlernte Hauptstrategie“.

Als nachträglich ergänzte Einordnung der oberen Verteilungsschwänze ergeben die
Rohdaten im letzten Trainingsviertel:

| Variante | ≥20 Wissenschaftspunkte | ≥30 | ≥40 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Standard | 153 | 15 | 1 | 40 |
| Entropie | 70 | 4 | 0 | 36 |
| Lambda | 0 | 0 | 0 | 18 |
| Kombination | 0 | 0 | 0 | 17 |

Alle Anzahlen beziehen sich jeweils auf 6.240 Partien, unabhängig von frühem Grün
und Spielausgang. Beim Standard erfüllen 13 der 15 Partien mit mindestens 30
Punkten zusätzlich frühes Grün und Nichtverlust; die eine 40-Punkte-Partie
ebenfalls. Keine Variante erreicht hier 50 Wissenschaftspunkte. Die Grenzwerte
30/40 sind eine nachträgliche Vertiefung, kein vorab definiertes Erfolgskriterium.
Hohe Wissenschaftserträge bleiben damit selten: Im Standard erreichen nur
0,24 % dieser Partien mindestens 30 Wissenschaftspunkte. Die zunehmende frühe
grüne Bauquote darf nicht mit einer zuverlässig starken Wissenschaftsstrategie
gleichgesetzt werden.

Mit frühem Grün erzielt der Standard hier durchschnittlich 9,64
Wissenschaftspunkte, ohne frühes Grün 2,63. Der mittlere VP-Abstand beträgt
−3,57 gegenüber −4,05. Dies ist eine beobachtete Gruppendifferenz und **kein
kausaler Nachweis**, dass die frühe Karte den Punktevorsprung verbessert hat.

## Entropie und Credit Assignment

Mittlere gemessene Policy-Entropie über die ersten/letzten 64 PPO-Updates:

| Variante | Anfang | Ende | Value-Loss am Ende |
| --- | ---: | ---: | ---: |
| Standard | 0,968 | 0,348 | 0,330 |
| Entropie | 1,079 | 0,463 | 0,328 |
| Lambda | 0,852 | 0,362 | 0,675 |
| Kombination | 1,044 | 0,512 | 0,717 |

Der höhere Koeffizient geht wie beabsichtigt mit höherer gemessener Entropie
einher, aber nicht mit besseren späten Wissenschaftskennzahlen als beim Standard.
Die Kombination hat am Ende sogar die höchste Entropie und fast kein frühes
Grün. Mehr allgemeine Aktionsvielfalt genügt hier offensichtlich nicht als
Wissenschaftsdiagnose. Unterschiedliche Zustands-/Maskenverteilungen beeinflussen
auch diese Entropiemittelwerte.

Lambda 1,0 ist bei diesem Seed keine erfolgreiche Abhilfe für die beobachtete
Wissenschaftsschwäche. Der etwa doppelte späte Value-Loss ist mit schwierigeren
oder stärker streuenden Lernzielen vereinbar, isoliert aber keine Ursache.
Weder das vollständige Credit-Assignment-Problem noch Deep Exploration sind mit
diesem einzelnen Parameterversuch abschließend bewertet.

## Spielstärke: noch kein fairer Variantenvergleich

Die letzten Checkpoint-Evaluationen ergeben 4,64 / 4,69 / 0,59 / 0,35
Wissenschaftspunkte im Mittel für Standard / Entropie / Lambda / Kombination.
Das bestätigt das grobe Wissenschaftsmuster auch für die eingefrorenen Endmodelle.
Die Gegner sind aber je Variante deren eigene Checkpoints 520.192 und 262.144.
Es sind verschiedene Modelle mit verschiedenen Testplan-Identitäten.

Daher darf aus den Sieganteilen von 39,24 % / 33,22 % / 42,59 % / 36,81 % **kein
Spielstärkeranking der vier Varianten** abgeleitet werden. Ebenso zeigen negative
Trainingsmargen keine allgemeine Verschlechterung: Die Gegner verändern sich.

## Nächster Schritt

Zuerst die vorhandenen Anfangs-, Halbzeit- und Endmodelle auf demselben
eingefrorenen Gegnerplan auswerten. Dafür ist kein neues Training nötig. Der
Befehl steht unter „Gemeinsamer Spielstärketest“ in
[SCIENCE_DIAGNOSTICS.md](SCIENCE_DIAGNOSTICS.md). Ein ausführbares Skript liegt
lokal unter `python/runs/science-diagnostic-v1-common-holdout.sh`.
Es spielt 5.760 Evaluationspartien und wurde bei dieser Analyse **nicht gestartet**.

Die Arbeitshypothese sollte nun lauten: Die Standard-PPO-Konfiguration kann frühe
Wissenschaft entwickeln, tut dies aber erst spät und nicht zuverlässig. Das
Budget und die Entwicklung im Self-Play sind ernst zu nehmende Erklärungsanteile.
Eine spezielle Deep-Exploration-Methode ist anhand dieser Daten noch nicht als
notwendige nächste Änderung begründet. Erst den gemeinsamen Holdout prüfen,
danach Standard versus sinnvolle Alternativen über weitere Trainingsseeds
wiederholen. Noch keinen Wissenschaftsbonus und keine allgemeine Umstellung auf
Lambda 1,0 aus diesen Befunden ableiten.

## Lokale Artefakte

- Interaktives Dashboard: `python/runs/science-diagnostic-v1-summary.html`.
- Originalzusammenfassung: `python/runs/science-diagnostic-v1-summary.json`.
- Unabhängige Rohdatenprüfung und letzte Viertel:
  `python/runs/science-diagnostic-v1-audit.json`.
- Prüfskript: `python/runs/science-diagnostic-v1-audit.py` (aus `python/` ausführen).
- Zusammenfassungs-SHA-256:
  `f99bcb66199bd7f216ac3497de6b70a2ebcaba1dfe673bdf8809740dcc32c55c`.

Es liegt nur **ein Trainingsseed** je Variante vor. Viele Spiele innerhalb eines
Laufs sind keine unabhängigen Wiederholungen des Lernverfahrens. Diese Diagnose
belegt das Verhalten dieser vier Läufe; sie liefert keine robuste Rangfolge der
Algorithmen über zufällige Initialisierungen.
