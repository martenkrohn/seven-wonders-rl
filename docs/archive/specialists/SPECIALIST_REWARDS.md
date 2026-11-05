# Wissenschafts- und Militärspezialisten: Plan und Reward-Vertrag

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Reward-Vertrag vom 25. Oktober 2026; Status ergänzt am 26. Oktober. Die beiden
v2-Trainings und ihr erster Holdout sind abgeschlossen; Ergebnisse stehen in
[SPECIALIST_RESULTS_2026-10-26.md](SPECIALIST_RESULTS_2026-10-26.md). Die breitere
Gegnerprüfung und Aufnahme verifizierter Modelle in den gemischten Pool stehen aus.

## Ziel und erste Versuchseinstellung

Zwei vorhandene neuronale Spieler werden zu unterschiedlichen, weiterhin
spielstarken Gegnern weitertrainiert. Als gemeinsamer Ausgangspunkt dient das
abgeschlossene Standardmodell mit Feature-Semantik v3 und 524.288 Übergängen:
`python/runs/science-diagnostic-v1-baseline-seed31/latest.pt`. Es ist eine bekannte
funktionsfähige Referenz, kein behaupteter global bester Spieler. Quelle und
SHA-256 werden in den neuen Läufen protokolliert.

Für einen tatsächlich abgeschlossenen Spielzustand seien:

- `D = eigene Gesamt-VP − max(Gesamt-VP der beiden Gegner)`;
- `S = eigene points.scientific`, einschließlich korrekt gewerteter Symbole,
  Sets und Joker;
- `M = eigene points.military`, also gewertete Siege minus Niederlagenpunkte
  über alle drei Zeitalter. Gewinne sind nach Zeitalter mit 1/3/5 gewichtet.

| Profil | Terminaler Trainingsreward |
| --- | --- |
| Standard/Kontrolle | `D / reward_scale` |
| Wissenschaft | `(D + alpha * S) / reward_scale` |
| Militär | `(D + alpha * M) / reward_scale` |

Erster Pilot: `alpha = 1.0`, `reward_scale = 10`. Diese Stärke ist eine
Versuchseinstellung, kein empirisch optimierter Wert. Wissenschaft und Militär
sind bereits Teil der Gesamt-VP; der Zusatzterm gewichtet die eigene Zielkategorie
noch einmal. Bei `alpha = 0` entsteht exakt der Standardreward.

Beispiel: 40 eigene VP, stärkster Gegner 45 VP, 20 Wissenschafts-VP ergeben
Standard −0,5 und Wissenschaft +1,5. Das ist eine beabsichtigte Verschiebung des
Trainingsziels: Ein Spezialist kann für eine gute Spezialisierung trotz Niederlage
positiven Reward erhalten. Deshalb entscheiden spätere echte Siege, VP-Abstand
und Kategorieerträge gemeinsam über seine Brauchbarkeit.

Militär verwendet den vorzeichenbehafteten Netto-Score, typischerweise −6 bis
+18. Niederlagen liefern also einen negativen Zusatzterm. Mehr Schilde als für
gewonnene Konflikte nötig, ungenutzte Produktion, rote/grüne Karten an sich und
bloßes Abwerfen erhalten keinen eigenen Bonus. Wissenschaft wird anhand der
finalen Engine-Wertung belohnt, nicht anhand einer Proxy-Schätzung.

## Zeitpunkt und Abgrenzung

Alle nichtterminalen Übergänge behalten Reward null. Der vollständige Reward
wird genau einmal nach letzter Aktion, letzten Kämpfen und Gildenkopien aus
`phase=finished` berechnet. Timeouts, Protokollfehler und unvollständige Partien
bekommen keinen erfundenen Terminalbonus.

Der erste Pilot verwendet bewusst keinen zusätzlichen Schritt-/Zeitalterbonus,
keine Schilde- oder Farbzwänge und keine gleichzeitige Änderung von Entropie,
Gamma oder GAE-Lambda. So wird der Effekt des neuen Lernziels getrennt von
weiteren Explorationseingriffen untersucht. Das garantiert noch keine erfolgreiche
langfristige Strategie; falls der Pilot nicht spezialisiert, zuerst Lernkurven und
Verläufe prüfen und dann Bonusstärke oder zeitliche Credit-Zuweisung einzeln ändern.

## Initialisierung und Gegner

`--init-checkpoint` lädt kompatible Modellgewichte einschließlich Kritiker.
Optimizer, Schrittzähler, Seed-Ströme und Self-Play-Pool beginnen neu. Das ist
Gewichtsinitialisierung, kein exaktes Fortsetzen des alten Laufs. Der Kritiker
passt sich dem geänderten Reward an. Unterschiedliche Hidden-Größen werden aus
dem Checkpoint übernommen; inkompatible Merkmals-/Architekturversionen werden
abgewiesen. Evaluation und Berichte kennzeichnen das Schritt-0-Modell als
vortrainierten Ausgangspunkt.

Beide Spezialisten trainieren gegen zwei eingefrorene Kopien desselben normalen
Standardmodells (`science-diagnostic-v1-baseline-seed31/latest.pt`). Die Gegner
ändern ihre Gewichte während des gesamten Laufs nicht. Der Trainer kopiert die
Gegnerdateien in das neue Laufverzeichnis und protokolliert Quellpfad und Hash.
Eigene Spezialisten-Zwischenstände werden gespeichert, aber niemals als
Trainingsgegner ausgewählt. Die Lernerposition rotiert weiterhin über alle Sitze.

Technisch bleibt `--mode selfplay` der neuronale Ausführungspfad;
`--frozen-opponents` ersetzt dessen wachsenden Gegnerpool vollständig. Werden
mehrere explizite Dateien angegeben, wird je Gegnersitz unabhängig gleichverteilt
gezogen. Die vorbereitete erste Serie verwendet genau eine Standarddatei für
beide Sitze, damit die Spezialisierung gegen gewöhnliche Gegner gelernt wird.
Trainingsevaluationen verwenden denselben festen Gegnerplan, separate
Evaluationsseeds und keinen Bonus. Der abschließende Holdout prüft andere,
nicht als Trainingsgegner eingesetzte Standard-Checkpoints.

Für den ersten längeren Versuch: 524.288 zusätzliche Übergänge je Spezialist,
Seed 31, acht Umgebungen × Horizont 64, vier PPO-Epochen, Batchgröße 256,
Entropie 0,01, Gamma 0,999, Lambda 0,95, Snapshots alle 4.096 Übergänge und
Evaluation alle 65.536. Ein optionaler Kontrolllauf startet vom selben Modell
mit Standardreward und exakt denselben übrigen Einstellungen. Die zusätzliche
Parallelität ist hier bewusst für alle Varianten identisch.

## Protokollierung und Verifikation

Konfiguration, Manifest und Checkpoints speichern Reward-Profil, Bonusstärke,
Skalierung und Herkunft der Startgewichte. Pro abgeschlossener Episode werden
normaler VP-Abstand, unskalierter Zusatzterm, Standardreward, skalierter Bonus
und gesamter Trainingsreward getrennt aufgezeichnet. Trainingsreward darf nicht
als Siegquote oder reale Punktzahl interpretiert werden.

Die Evaluatoren laden ausschließlich Modellgewichte. Ihre Umgebungen bleiben beim
Standardreward; VP, Siege, Kategorien und Unsicherheitsintervalle enthalten keinen
Trainingsbonus. Für beide Spezialisten und die Kontrolle denselben unveränderten
Holdout verwenden: normale Standard-Checkpoints 131.072 und 262.144, alle Sitze,
identische/mischte Gegnerpaare, 40 Seed-Blöcke mit insgesamt 480 Spielen je Modell,
Seed 79001. Gegen weitere unabhängige normale Modelle und den jeweils anderen
Spezialisten prüfen, sobald brauchbare Kandidaten vorliegen. Ein zusätzliches
abschließendes Testset nicht zur Auswahl von Alpha verwenden.

Erst bei nachvollziehbar erhöhtem echten Kategorieertrag **und** noch brauchbarer
Spielstärke aufnehmen. Eine feste Grenzzahl ist vor dem Pilot nicht belegt;
Sieganteil, VP-Abstand und deren gepaarte Intervalle gegenüber Ausgangsmodell und
Kontrolle transparent ausweisen. Nach einem erfolgreichen Pilot über Seeds 47
und 59 wiederholen; keine belastbare Diversität aus einem einzelnen Seed behaupten.

## Ablauf und Befehle

1. Kleine technische Testläufe beider Profile prüfen Reward-Zerlegung,
   Gewichtsinitialisierung, Logging und bonusfreie Evaluation.
2. Das vorbereitete Trainingsskript startet Wissenschaft und Militär nacheinander.
3. Das kombinierte Skript erzeugt anschließend gemeinsame Holdout-Berichte für beide.
4. Aus den Ergebnissen Bonusstärke bzw. Budget entscheiden. Erst danach den
   gemischten Generalisten-Lauf und spätere Deep Exploration angehen.

Aus dem Verzeichnis `python/`:

```sh
bash runs/train_and_evaluate_specialists_v2.sh
```

Dieser eine Befehl trainiert beide Modelle nacheinander und startet danach
für beide den gemeinsamen Holdout samt Berichten und Dashboard.

Die Quelldatei `scripts/ppo_training/specialist_experiment_commands.py` kann die
Skripte mit anderen Seeds, Budgets, Bonusgewichten oder Präfixen erneut erzeugen.
Mit `--include-control` wird zusätzlich der Standardreward-Kontrolllauf ausgegeben.
Die Option `--phase all --include-control` erzeugt eine vollständige Serie mit
allen drei Varianten.

Die neuen Modellverzeichnisse heißen `runs/specialists-v2-frozen-science-seed31` und
`runs/specialists-v2-frozen-military-seed31`. Der neue Schrittzähler misst zusätzliche
Trainingsübergänge ab dem geladenen Ausgangspunkt. Die Skripte überspringen
passende fertige Läufe; unvollständige passende Läufe werden archiviert und neu
aus den Startgewichten trainiert. Es gibt weiterhin keinen exakten Resume.
Bestehende Modelle und Ergebnisse werden nicht überschrieben.

## Technische Prüfung

Die Einrichtung wurde mit beiden Profilen in kurzen 1.024-Übergänge-Läufen vom
vorgesehenen Standardmodell geprüft: acht Umgebungen, abgeschlossene Trainings,
je drei vollständig abgeschlossene Evaluationsspiele (ein älterer Snapshot, alle Sitze). Die lokalen Ergebnisse
stehen unter `python/runs/specialists-v1-setup-check/`, einschließlich
`setup_validation.json` mit Herkunftshash, Reward-Zerlegung und Ergebnissen.
Das ist eine Funktionsprüfung und kein Nachweis erfolgreicher Spezialisierung.
Zum Zeitpunkt dieser technischen Einrichtung waren die langen Trainings noch nicht
gestartet. Inzwischen sind beide v2-Läufe und ihre erste Holdout-Evaluation
abgeschlossen; siehe [Ergebnisse vom 26. Oktober](SPECIALIST_RESULTS_2026-10-26.md).

Regressionstests prüfen die Formeln einschließlich negativer Militärpunkte und
Nullgewicht, Reward null vor Spielende und bei Timeout, exakte Übernahme der
Startgewichte ohne Optimizer-/RNG-Zustand, abweichende Dimensionen, geänderte
Quelldateien beim Neustart und die vortrainierte Beschriftung in Holdout-Berichten.


## Änderung des Gegnerplans

Der anfängliche Plan mit eigenen Spezialisten-Snapshots als Gegner wurde auf
Wunsch durch den festen normalen Gegner ersetzt. Ältere `*_v1.sh`-Skripte
verwenden weiterhin den alten Self-Play-Plan und sind für diese Serie nicht zu
verwenden. Die v2-Verzeichnisse trennen beide Versuchsanordnungen. Der kombinierte
v2-Befehl startet lange Läufe ausschließlich, wenn er vom Benutzer ausgeführt wird.
