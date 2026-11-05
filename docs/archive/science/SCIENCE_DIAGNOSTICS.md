# Wissenschaft: Exploration und spätes Lernsignal diagnostizieren

> **Historischer Stand.** Versuchsprotokoll oder Auswertung der explorativen Phase;
> keine aktuelle Handlungsanweisung. [Heutiger Einstieg](../../README.md) · [Archivübersicht](../README.md).

Diese Diagnose verändert weder Reward noch Eingaben, Aktionen, Gegnerauswahl oder
Zufallsströme. Sie ist mit `--science-diagnostics` im vereinheitlichten Trainer
zuschaltbar. Lange Trainingsläufe werden ausschließlich vom Nutzer gestartet.

## Fragestellung und Grenzen

1. Erlebt das Training überhaupt Partien mit früher grüner Investition und
   anschließend hohen Wissenschaftspunkten und konkurrenzfähigem Ergebnis?
2. Werden frühe grüne Entscheidungen und die ihnen zugewiesene
   Policy-Wahrscheinlichkeit im weiteren Training häufiger?
3. Ändert zusätzliche Entropie oder ein stärker durchgereichtes spätes Lernsignal
   diese Befunde und die Spielstärke auf einem festen Testplan?

Das sind deskriptive Diagnosen. Ein Zusammenhang zwischen früher Wissenschaft
und Endergebnis ist kein kausaler Wert der Karte: Ressourcen, Wunder, Kartenhände
und Gegner unterscheiden sich. Auch die mittlere Aktionswahrscheinlichkeit kann
sich aufgrund anderer besuchter Zustände ändern. Eine erfolgreiche Partie wird
nicht automatisch von einem einzigen unveränderten Modell gespielt; die Policy
kann zwischen Rollouts aktualisiert werden. Jede Entscheidung speichert deshalb
den zugehörigen Policy-Trainingsschritt.

Ein positiver Befund beim Entropieexperiment beweist keine Lösung von Deep
Exploration. Höheres `gae_lambda` betrifft Credit Assignment und den
Bias-/Varianz-Kompromiss, nicht die Exploration selbst. Hier ist `gamma=.999`
bereits nahe 1; es bleibt unverändert. Die vier Varianten verändern jeweils nur
Entropiekoeffizient und/oder GAE-Lambda. Die Self-Play-Gegner entwickeln sich je
Variante unterschiedlich mit, deshalb bleibt der gemeinsame Holdout wesentlich.

## Erfasste Daten

`science_episodes.jsonl` enthält ausschließlich vollständig abgeschlossene
Trainingspartien des Lerners, mit Umgebung, Episode, Engine-Seed, Sitz und
Gegneridentitäten. Pro tatsächlich ausgeführter Entscheidung werden Zeitalter,
Karte, Aktion, legale grüne Bauoption, aufsummierte Wahrscheinlichkeit aller
legalen grünen Bauaktionen und tatsächlich ausgeführter grüner Bau gespeichert.
Bezahlte und kostenlose Bauaktionen zählen zur Wahrscheinlichkeit; Ablage und
Wunderbau mit einer grünen Karte zählen nicht als grüner Bau. Übersprungene
siebte Karten zählen nicht als Entscheidungen.

Am Ende jedes Zeitalters werden Wissenschaftssymbole, Joker und die Zahl gebauter
grüner Karten festgehalten. Am Spielende folgen Wissenschaftspunkte, Punktetotale,
Abstand zum stärksten Gegner und anteiliger Sieg. Wissenschaftspunkte können auch
aus Wundern/Gilden stammen. Unfertige Partien am Budgetende fehlen bewusst;
technische Fehler werden weiter über `failure.json` protokolliert.

Die Offline-Auswertung gruppiert nach Partiebeginn in Fenstern von 65.536
Transitionen. Sie liefert Bauquote, mittlere grüne Wahrscheinlichkeit bei legaler
Option, Ergebnisse mit/ohne frühes Grün und Anzahlen ab 10/20 Wissenschaftspunkten.
Zusätzlich zählt sie Partien mit mindestens einem grünen Bau in Zeitalter I,
mindestens 10/20 Wissenschaftspunkten und VP-Abstand >=0. Das sind vorab definierte
Diagnoseschwellen, keine Definition optimalen Spiels. Sie protokolliert die erste
solche Entdeckung zur späteren Inspektion im JSONL. Fehlende Nenner werden `null`.

## Vergleichsplan

| Variante | Entropiekoeffizient | GAE-Lambda |
| --- | ---: | ---: |
| baseline | 0.01 | 0.95 |
| entropy | 0.03 | 0.95 |
| credit | 0.01 | 1.0 |
| combined | 0.03 | 1.0 |

Jede Variante startet frisch, mit 524.288 Transitionen, zwei Environments und
256 Schritten pro Rollout. Gleiche Seeds pro Variante paaren die Ausgangsbedingungen,
ohne identische spätere Trajektorien zu versprechen. Seed 31 ist der erste
Vergleich; Seeds 47 und 59 dienen der Wiederholung. Ein einzelner Seed reicht
nicht für robuste Aussagen über die Trainingsmethode.

Alle Befehle aus `python/`. Der Generator druckt ausschließlich Befehle, startet
selbst nichts. Die erzeugten Befehle enthalten `--restart-incomplete` und
`--skip-completed`: Vollständig abgeschlossene Varianten mit identischen
Einstellungen werden übersprungen. Abgebrochene Varianten werden nach
`runs/archive/<Runname>-<Zeitstempel>-<Kennung>/` verschoben und frisch gestartet.
Vorhandene Dateien werden nicht gelöscht; Modelle werden nicht aus dem Archiv
fortgesetzt. Ein Lauf ohne `result.json`, etwa nach Prozessabbruch, gilt ebenfalls
als unvollständig. Veränderte Trainingseinstellungen erfordern einen neuen
`--prefix`, damit verschiedene Experimente nicht versehentlich vermischt werden.
Sperren verhindern gleichzeitige neue Trainer im selben Ausgabeverzeichnis;
lokale ältere Trainer werden zusätzlich über ihre Prozessargumente erkannt.

Nach einem Abbruch dieselben beiden Befehle erneut ausführen: **das Shellskript
zuerst neu erzeugen**, falls es noch aus der Version ohne Neustartoptionen stammt.
Ein aktuell laufender Prozess darf nicht parallel erneut gestartet werden.
Archive können später nach Sichtung manuell gelöscht werden. Eine externe
Self-Play-League außerhalb des Run-Verzeichnisses wird nicht automatisch
zurückgesetzt; hierfür einen frischen Run samt frischer League verwenden.

Bei direktem Aufruf von `train_unified.py` die beiden Optionen an den bisherigen
Trainingsbefehl anhängen. Ohne diese Optionen bleibt die bisherige Ablehnung
nichtleerer Verzeichnisse bestehen. Dies ist **kein exaktes Resume**: Zufallsströme,
Collector-Zustand und laufende Simulatorpartien werden derzeit nicht gespeichert.

```bash
# Zunächst vier längere Läufe, nacheinander. Anschließend automatische Offline-Auswertung.
.venv/bin/python scripts/ppo_training/science_experiment_commands.py --seeds 31 > /tmp/science-seed31.sh
bash /tmp/science-seed31.sh
```

Die Offline-Auswertung erzeugt automatisch auch
`runs/science-diagnostic-v1-summary.html`. Diese Datei direkt im Browser öffnen.
Sie verwendet das bestehende Dashboard mit Sprache, Farbschema, sortierbaren
Tabellen und CSV-Export. Im Bereich **Wissenschaftsdiagnose** sind alle Varianten
als Auswertung auswählbar. Die Verlaufskurve vergleicht die geladenen Varianten;
Kennzahl und Zeitalter lassen sich wechseln. Die Tabellen zeigen Bauangebote,
Bauquoten, Policy-Wahrscheinlichkeiten, Wissenschaftspunkte, Gruppen mit/ohne frühe
Investition und erste erfolgreiche Wissenschaftsverläufe. Ein fehlender
Entdeckungszeitpunkt wird als „nicht erfasst“ angezeigt, nicht als Null.

Diese Ergänzung erfordert keinen Trainingsneustart. Ein bereits gestartetes
Experiment-Shellskript ruft die aktualisierte Offline-Auswertung am Ende auf.
Das Dashboard ist eine lokale Momentaufnahme, keine automatisch aktualisierte
Live-Ansicht. Neue Daten werden durch erneute Offline-Auswertung übernommen.

Alternativ eine vorhandene JSON-Zusammenfassung ohne erneutes Training darstellen:

```bash
.venv/bin/python scripts/build_dashboard.py \
  --science-summary runs/science-diagnostic-v1-summary.json \
  --out-dir runs/science-dashboard
```

Dann `runs/science-dashboard/dashboard.html` öffnen. Über **Auswertung hinzufügen**
können weitere Diagnose-Zusammenfassungen oder bisherige `analysis.json`-Dateien
lokal geladen werden. Eine neu erzeugte Version des bisherigen Dashboards
akzeptiert Diagnose-JSON ebenfalls; bereits früher erzeugte HTML-Dateien müssen
dafür mit `scripts/build_dashboard.py --evaluation-dir ...` neu erzeugt werden.
Mit zusätzlichem `--science-summary ...` lassen sich beide Datentypen direkt in
eine HTML-Datei einbetten. Trainingsdiagnosen bleiben von Holdout-Ergebnissen
getrennt; es werden keine Testspielstärke oder Konfidenzintervalle erfunden.
Die öffentliche Demo wird dadurch nicht aktualisiert.

Danach mit `runs/science-diagnostic-v1-summary.json` und den Run-Verzeichnissen
wieder melden. Weder dieser erste Vergleich noch die Wiederholungen werden bei
der Einrichtung automatisch ausgeführt.

```bash
# Weitere acht längere Läufe, nach Sichtung des ersten Vergleichs.
.venv/bin/python scripts/ppo_training/science_experiment_commands.py --seeds 47 59 > /tmp/science-replications.sh
bash /tmp/science-replications.sh

# Alle bereits abgeschlossenen Varianten gemeinsam auswerten; startet keine Spiele.
.venv/bin/python scripts/ppo_training/analyze_science.py \
  runs/science-diagnostic-v1-*-seed{31,47,59} \
  --out runs/science-diagnostic-v1-summary.json
```

## Gemeinsamer Spielstärketest nach dem Training

Trainings-Evaluationen verwenden wechselnde Vorgängermodelle und sind daher kein
direkter Spielstärkevergleich zwischen Varianten. Der folgende optionale längere
Test verwendet für alle vier Varianten dieselben zwei eingefrorenen Gegner,
Seeds und Sitzrotationen. Er vergleicht jeweils Anfang, Mitte und Ende; 480 Spiele
pro Modell, 1.440 pro Variante. Weitere Trainingsseeds können mit denselben
Referenzen geprüft werden. Gegner aus einer einzelnen Trainingslinie begrenzen
auch hier die Aussage über allgemeine Spielstärke.

```bash
set -euo pipefail
for variant in baseline entropy credit combined; do
  run="runs/science-diagnostic-v1-${variant}-seed31"
  .venv/bin/python scripts/eval_selfplay.py \
    --run-dir "$run" \
    --reference runs/science-diagnostic-v1-baseline-seed31/checkpoint_131072.pt \
                runs/science-diagnostic-v1-baseline-seed31/checkpoint_262144.pt \
    --compare "$run/checkpoint_262144.pt" \
    --blocks 40 --seed 59001 --out-dir "${run}-holdout"
done
```

Interpretation: Bleiben die Entdeckungszähler bei null, wurde in diesem Budget
kein Verlauf nach der jeweiligen Definition beobachtet. Sind Entdeckungen da,
aber frühe Bauquoten/Wahrscheinlichkeiten steigen nicht nachhaltig, ist das ein
Anlass, Credit Assignment und das Beibehalten gefundener Strategien genauer zu
untersuchen. Eine erfolgreiche Änderung sollte sich über Trainingsseeds
wiederholen und auf dem gemeinsamen Holdout konkurrenzfähig sein. Mehr grüne
Karten allein ist kein Erfolgskriterium.
